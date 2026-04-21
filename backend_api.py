# -*- coding: utf-8 -*-
# @Time    : 16.04.2026
# @File    : backend_api.py
# @Software: PyCharm

"""
backend_api.py - 文件处理回调接口 + OSS文件批量向量化入库
"""

import os
import re
import json
import uuid
import time
import asyncio
import subprocess
from typing import List
import pandas as pd
import uvicorn
import requests
import pdfplumber
import pymysql
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pymilvus import MilvusClient

from config import DB_CONFIG, MILVUS_MAIN_URI, MILVUS_MAIN_TOKEN, PORT
from utils import vector_4b, request_qwen_async
# 🌟 新增导入：KMeans, numpy 和 JSON提取工具
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity  # 🌟 新增相似度计算库
from utils import vector_4b, request_qwen_async, extract_json_array_from_text, get_cached_vector, extract_text_from_image
from xmindparser import xmind_to_dict
import glob
from multi_pdf_to_json_queue import PDFParsing, LAYOUT_PATH
from matcher import QwenEmbeddingMatcher
from audit import AgenticPipeline


app = FastAPI(title="文件处理回调接口")

MILVUS_COLLECTION_MAIN = 'evidence_card_4B'

# ================= 配置区 =================
BASE_DIR   = os.path.join(os.getcwd(), "api_data")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
STATE_DIR  = os.path.join(os.getcwd(), "task_states")  # 🌟 新增这一行
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)  # 🌟 新增创建目录

# Milvus 客户端
milvus_client = MilvusClient(uri=MILVUS_MAIN_URI, token=MILVUS_MAIN_TOKEN)

# 全局预编译正则
ZH_PATTERN     = re.compile(r'[\u4e00-\u9fa5]')
SYMBOL_PATTERN = re.compile(r'[#\|\$_\~\^\/\\<>\*\}]')

pdf_parser_engine = None
gpu_lock = asyncio.Lock()  # 保护显卡不被并发撑爆的核心！

# ================= 数据模型 =================
class FileCallbackItem(BaseModel):
    # 注意：前端传来的是 "0000001"，我们先用 str 接收，后续存库时再转为 int
    id: int = Field(..., description="文件ID")
    file_url: str = Field(..., description="文件存储链接（OSS地址）")
    file_name: str = Field(..., description="文件名称")
    user_id: str = Field(..., description="用户ID")


# ================= 启动事件 =================
@app.on_event("startup")
async def startup_event():
    global pdf_parser_engine
    # print("正在加载 YOLO 和 OCR 模型入显存...")
    pdf_parser_engine = PDFParsing(layout_path=LAYOUT_PATH)
    print("模型加载完毕，API 准备就绪！")


# ================= 辅助函数 =================
def is_garbage_text(text: str) -> bool:
    """判断是否为乱码/无意义文本（OCR失败、纯公式、全是乱码符号）"""
    if len(text) < 50:
        return True
    text_length = len(text)

    # 符号乱码检测（中英文通用）
    weird_symbols = SYMBOL_PATTERN.findall(text)
    symbol_ratio = len(weird_symbols) / text_length
    if symbol_ratio > 0.1:
        return True

    # 空格过多检测（中英文通用）
    space_ratio = text.count(' ') / text_length
    if space_ratio > 0.4:
        return True

    # 判断语言类型，再用对应规则
    zh_chars = ZH_PATTERN.findall(text)
    zh_ratio = len(zh_chars) / text_length

    en_chars = len([c for c in text if 'a' <= c <= 'z' or 'A' <= c <= 'Z'])
    en_ratio = en_chars / text_length

    if zh_ratio >= 0.1:
        # 中文文档：中文占比需 >= 10%
        return False
    elif en_ratio >= 0.3:
        # 英文文档：英文字母占比 >= 30% 则认为有效
        return False
    else:
        # 既不像中文也不像英文，判为乱码
        return True

# ================= 🌟 修改点 2：新增获取历史卡片函数 =================
def get_user_history_titles(user_id: int) -> list:
    """从数据库中获取该用户之前生成的所有卡片标题"""
    conn = None
    cursor = None
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT card_name FROM evidence_file_card_name WHERE user_id = %s", (user_id,))
        rows = cursor.fetchall()
        return [row[0] for row in rows] if rows else []
    except Exception as e:
        print(f"[DB ERROR] 获取用户历史卡片失败: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# ================= 修改mysql状态 =================
def update_file_parse_status(file_id: int, status: int):
    """
    更新 evidence_file_info 表的解析状态
    status: 1-解析完成；2-解析失败
    """
    conn = None
    cursor = None
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()
        sql = "UPDATE evidence_file_info SET file_status = %s WHERE id = %s"
        cursor.execute(sql, (status, file_id))
        conn.commit()
        print(f"[DB LOG] 文件ID {file_id} 状态更新为 {status}")
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"[DB ERROR] 更新文件状态失败: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ================= 🌟 修改点 2：新增分批存库函数 =================
def insert_file_card_names(file_id: int, user_id: int, card_names: list):
    """
    分批将大模型提炼的卡片标题写入 evidence_file_card_name 表
    """
    if not card_names:
        return
    conn = None
    cursor = None
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # 1. 生成当前时间戳
        # int(time.time()) 生成的是 10 位数的秒级时间戳。
        # 如果你的 bigint 字段设计用来存储 13 位数的毫秒级时间戳，请改为：int(time.time() * 1000)
        # current_time = int(time.time())
        current_time = int(time.time() * 1000)

        sql = """
            INSERT INTO evidence_file_card_name (file_id, card_name, name_info, user_id, status, name_repeat, status_time)
            VALUES (%s, %s, %s, %s, 0, %s, %s)
        """
        # 修改：元组中增加 item.get("name_repeat", 0)
        insert_data = [
            (
                file_id,
                item["title"],
                item["reason"],
                user_id,
                item.get("name_repeat", 0),
                current_time
            )
            for item in card_names
        ]

        # 构造批量插入的数据元组
        # insert_data = [(file_id, item["title"], item["reason"], user_id) for item in cards]
        cursor.executemany(sql, insert_data)
        conn.commit()
        print(f"[DB LOG] 成功为文件ID {file_id} 插入 {cursor.rowcount} 个新卡片标题！")
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"[DB ERROR] 插入卡片标题失败: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def is_chinese_or_english(text, threshold=0.05):
    """判断文本主要是中文还是英文"""
    chinese_count = 0
    english_count = 0
    total_count   = 0
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            chinese_count += 1
            total_count += 1
        elif 'a' <= char <= 'z' or 'A' <= char <= 'Z':
            english_count += 1
            total_count += 1
    if total_count == 0:
        return "其他"
    chinese_ratio = chinese_count / total_count
    english_ratio = english_count / total_count
    if chinese_ratio >= threshold and chinese_ratio > english_ratio:
        return "中文"
    elif english_ratio >= threshold and english_ratio > chinese_ratio:
        return "英文"
    else:
        return "其他"




def detect_pdf_type(file_path: str) -> str:
    """
    分析文件类型
    返回: 'xmind' | 'image' | 'text'
    """
    try:
        print()
        print(f"文件路径：{file_path}")

        # 【新增逻辑】：直接通过后缀名拦截原生 Xmind 文件
        if file_path.lower().endswith('.xmind'):
            return "xmind"

        # 如果不是 .xmind 后缀，再尝试用 pdfplumber 按 PDF 解析
        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) == 0:
                return "text"

            # 1. 元数据检测
            metadata   = pdf.metadata or {}
            meta_string = " ".join([str(v) for v in metadata.values()]).lower()
            if 'xmind' in meta_string or 'mindmap' in meta_string:
                return "xmind"

            # 2. 排版特征检测
            page  = pdf.pages[0]
            words = page.extract_words()
            if not words:
                return "image"

            text        = page.extract_text() or ""
            text_length = len(text.strip())
            if text_length < 30:
                return "image"

            unique_y_positions = []
            for w in words:
                y = w['top']
                if not any(abs(y - uy) < 3 for uy in unique_y_positions):
                    unique_y_positions.append(y)

            avg_word_len    = sum(len(w['text']) for w in words) / len(words)
            is_scattered    = len(unique_y_positions) > (len(words) * 0.4)
            is_short_phrases = avg_word_len < 10

            if is_scattered and is_short_phrases:
                return "xmind"

            return "text"

    except Exception as e:
        print(f"检测文件类型失败: {e}")
        # 这里如果报错，可能是不受支持的格式或损坏的文件
        return "text"




def extract_xmind_to_text(pdf_path):
    """从 xmind 类型的 PDF 中提取节点坐标数据，供大模型还原结构"""
    nodes = []
    with pdfplumber.open(pdf_path) as pdf:
        page  = pdf.pages[0]
        words = page.extract_words()
        for w in words:
            nodes.append({
                "text": w['text'],
                "x": round(w['x0'], 1),
                "y": round(w['top'], 1)
            })
    return json.dumps(nodes, ensure_ascii=False)


def extract_text_from_pdf_with_plumber(pdf_path: str) -> list:
    """
    使用 pdfplumber 逐页提取 PDF 文本
    返回 chunk_text_data 所需的 [{"type": "text", "text": "..."}, ...] 格式
    """
    content_list = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and text.strip():
                    content_list.append({"type": "text", "text": text.strip()})
    except Exception as e:
        print(f"[pdfplumber] PDF 文本提取失败: {e}")
    return content_list

#
# def chunk_text_data(content_list, split_to_length: int = 200):
#     """核心分块逻辑：输入解析列表，输出分块后的列表"""
#     if not content_list:
#         return []
#
#     require_items = [x for x in content_list if x.get("type") in ("text", "title")]
#     if not require_items:
#         return []
#
#     require_list  = [x.get("text", "") for x in require_items]
#     combined_text = "".join(require_list)
#
#     lang_detect = is_chinese_or_english(combined_text)
#     if lang_detect == "中文":
#         separators = ("。", "！", "!")
#     else:
#         separators = (". ", ".", "!", "！", "。")
#
#     new_block_list = []
#
#     for i, con in enumerate(require_items):
#         t = (con.get("text") or "").strip()
#         if lang_detect == "英文" and t:
#             t += " "
#
#         current_bbox = con.get("bbox", [])
#
#         if not new_block_list:
#             new_block_list.append({
#                 "text": t,
#                 "bbox": [current_bbox] if current_bbox else [],
#                 "end_oldblock_type": con["type"]
#             })
#             continue
#
#         last           = new_block_list[-1]
#         last_text      = last["text"]
#         end_with_sep   = last_text.rstrip().endswith(separators)
#         is_last_required = (con is require_items[-1])
#
#         if (
#             len(last_text) < split_to_length
#             or last["end_oldblock_type"] == "title"
#             or (not end_with_sep)
#             or is_last_required
#         ):
#             last["text"] += t
#             if current_bbox:
#                 last["bbox"].append(current_bbox)
#             last["end_oldblock_type"] = con["type"]
#         else:
#             new_block_list.append({
#                 "text": t,
#                 "bbox": [current_bbox] if current_bbox else [],
#                 "end_oldblock_type": con["type"]
#             })
#
#     # 结尾小块并入前一块
#     if len(new_block_list) >= 2 and len(new_block_list[-1]["text"]) < 300:
#         new_block_list[-2]["text"] += new_block_list[-1]["text"]
#         new_block_list[-2]["bbox"].extend(new_block_list[-1]["bbox"])
#         new_block_list.pop()
#
#     final_res_list = []
#     for index, block in enumerate(new_block_list):
#         final_res_list.append({
#             "text": block["text"],
#             "index": index,
#             "length": len(block["text"]),
#             "bbox": block["bbox"]
#         })
#
#     return final_res_list


def chunk_text_data(content_list, split_to_length: int = 200):
    """优化后的核心分块逻辑：支持深度拆分超长字符串，按标点断句组装"""
    if not content_list:
        return []

    # 1. 过滤出需要处理的文本块
    require_items = [x for x in content_list if x.get("type") in ("text", "title") and x.get("text")]
    if not require_items:
        return []

    new_block_list = []
    current_text = ""
    current_bboxes = []
    last_type = ""

    # 🌟 核心改进 1：定义通用分句正则，支持中英文句号、感叹号、问号和换行符
    # 使用 () 捕获组可以保留分隔符本身，不会在 split 时丢失标点
    sentence_pattern = r'([。！？!\?\n]+)'

    for con in require_items:
        t = con.get("text", "").strip()
        if not t:
            continue

        current_bbox = con.get("bbox", [])
        item_type = con.get("type")

        # 🌟 核心改进 2：强制将传入的文本（即使是几千字的整块）按标点打碎成句子列表
        # 例如："你好。世界！" 会被 split 成 ["你好", "。", "世界", "！", ""]
        pieces = re.split(sentence_pattern, t)
        sentences = []

        # 将句子文本与后面的标点符号重新拼合
        for i in range(0, len(pieces) - 1, 2):
            sentence = pieces[i] + pieces[i + 1]
            if sentence.strip():
                sentences.append(sentence)
        # 把结尾没有标点的一点残留加上
        if len(pieces) % 2 != 0 and pieces[-1].strip():
            sentences.append(pieces[-1])

        if not sentences:
            sentences = [t]

        # 🌟 核心改进 3：遍历打碎后的短句，根据长度上限(split_to_length)重新组装
        for sentence in sentences:
            # 判断：如果当前积累的长度 + 新句子的长度 > 目标长度，且当前积累器里有东西
            # （例外：如果上一个是 title，则强行跟下一段连着，不截断）
            if (len(current_text) + len(sentence) > split_to_length) and len(current_text) > 0 and last_type != "title":
                # 保存上一个块
                new_block_list.append({
                    "text": current_text,
                    "bbox": current_bboxes.copy(),
                    "end_oldblock_type": last_type
                })
                # 清空累加器，开启新块
                current_text = sentence
                current_bboxes = [current_bbox] if current_bbox else []
            else:
                # 长度还没超，继续累加拼接到当前块
                current_text += sentence
                if current_bbox and current_bbox not in current_bboxes:
                    current_bboxes.append(current_bbox)

            last_type = item_type

    # 4. 循环结束后，把最后留在累加器里的尾巴内容存入 block
    if current_text:
        new_block_list.append({
            "text": current_text,
            "bbox": current_bboxes,
            "end_oldblock_type": last_type
        })

    # 5. 结尾小块并入前一块的逻辑优化
    # 为了防止把一个本来 180 字的块和 150 字的块合并成 330 字（超标）
    # 我们只将非常短的尾巴（例如小于目标长度的 1/3）合并到上一块
    min_tail_length = split_to_length // 3
    if len(new_block_list) >= 2 and len(new_block_list[-1]["text"]) < min_tail_length:
        new_block_list[-2]["text"] += new_block_list[-1]["text"]
        new_block_list[-2]["bbox"].extend(new_block_list[-1]["bbox"])
        new_block_list.pop()

    # 6. 构造最终返回的数据格式
    final_res_list = []
    for index, block in enumerate(new_block_list):
        final_res_list.append({
            "text": block["text"].strip(),
            "index": index,
            "length": len(block["text"].strip()),
            "bbox": block["bbox"]
        })

    return final_res_list


def convert_office_to_pdf_sync(docx_path: str, output_dir: str):
    """调用系统 LibreOffice 将 docx/doc/ppt/pptx 转换为 PDF"""
    command = [
        "soffice",
        "--headless",
        "--convert-to", "pdf",
        docx_path,
        "--outdir", output_dir
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    except subprocess.CalledProcessError as e:
        raise Exception(f"DOCX 转 PDF 失败: {e.stderr.decode('utf-8', errors='ignore')}")
    except FileNotFoundError:
        raise Exception("找不到 soffice 命令，请确保服务器安装了 LibreOffice 并配置了环境变量")


def process_and_insert_to_milvus(task_id: str, chunked_data: list, batch_id: str = None, type: str = 'pdf',
                                 user_id: str = '', file_id: int = 0):
    """
    遍历分块数据，调用 Embedding 接口，批量写入 Milvus。
    """
    if batch_id is None:
        batch_id = task_id
    if not chunked_data:
        return {"status": "error", "msg": "没有有效的分块数据可供插入"}

    filtered_count = 0
    valid_embeddings = []
    valid_texts = []
    valid_metadata = []

    # 批量入库的数组
    to_insert_batch = []

    try:
        for index, item in enumerate(chunked_data):
            text = item.get("text", "").strip()

            # 🌟 严控条件 1：如果切块文本为空，直接判定为解析失败
            if not text:
                return {"status": "error", "msg": f"解析失败：分块 {index} 文本为空"}

            if type == 'pdf':
                if len(text) > 600:
                    continue

            # （保留原有逻辑）如果是乱码或过短，先跳过，由循环结束后的 to_insert_batch 兜底判断
            if is_garbage_text(text):
                filtered_count += 1
                continue

            try:
                embeddings_list = vector_4b(text)
            except Exception as e:
                # 🌟 严控条件 2：Embedding 报错，直接判定为解析失败
                return {"status": "error", "msg": f"解析失败：分块 {index} 请求 Embedding API 报错: {str(e)}"}

            if not embeddings_list or not isinstance(embeddings_list, list) or len(embeddings_list) == 0:
                # 🌟 严控条件 3：向量获取为空或无效，直接判定为解析失败
                return {"status": "error", "msg": f"解析失败：分块 {index} 获取向量为空或无效"}

            doc_id = f"{task_id}_{index}"

            to_insert_batch.append({
                "id": doc_id,
                "vector": embeddings_list,
                "batch_id": batch_id,
                "text": text,
                "date": str(int(time.time())),
                "user_id": user_id,
                "file_id": file_id,
            })

            valid_embeddings.append(embeddings_list)
            valid_texts.append(text)
            valid_metadata.append(f"来源文件ID: {file_id}")

            # 如果全是乱码被过滤，导致没有有效数据，返回 error
        if not to_insert_batch:
            return {"status": "error", "msg": "解析失败：文本内容过短(小于50字)或全为无效乱码，无有效数据入库"}

        # 执行统一批量插入
        milvus_client.insert(collection_name=MILVUS_COLLECTION_MAIN, data=to_insert_batch)

    except Exception as e:
        return {"status": "error", "msg": f"Milvus 入库失败: {str(e)}"}

    return {
        "status": "success",
        "filtered_count": filtered_count,
        "embeddings": valid_embeddings,
        "texts": valid_texts,
        "metadata": valid_metadata
    }


# ================= 🌟 修改点 4：新增卡片提炼核心逻辑 =================
async def mine_card_titles_for_file(file_id: int, user_id: int, embeddings_list: list, texts: list, metadata: list):
    """基于向量聚类与大模型，智能提炼卡片标题"""
    TARGET_TOTAL_TITLES = 10  # 目标提取总数
    BATCH_INSERT_SIZE = 5  # 每满 5 个插入一次数据库

    num_chunks = len(embeddings_list)
    if num_chunks == 0:
        return

    embeddings = np.array(embeddings_list)
    # 根据文本分块数量动态决定聚类数量 (不用太多)
    N_CLUSTERS = max(1, min(num_chunks // 2, 5))

    print(f"[{file_id}] 开始进行 KMeans 聚类 (簇数量: {N_CLUSTERS})...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init="auto")
    kmeans.fit(embeddings)

    clusters_pool = {i: [] for i in range(N_CLUSTERS)}
    for idx, label in enumerate(kmeans.labels_):
        clusters_pool[label].append({
            "text": texts[idx],
            "source": metadata[idx],
        })

    # 1. 提前拉取该用户在数据库中已有的所有卡片标题
    existing_db_titles = get_user_history_titles(user_id)
    history_topics = list(set(existing_db_titles))  # 作为防重名黑名单

    # 2. 将历史标题转化为向量，利用 utils 的 get_cached_vector (会自动读本地文件)
    history_embeddings = []
    print(f"[{file_id}] 正在加载用户历史卡片的向量用于防重叠比对...")
    for t in history_topics:
        vec = get_cached_vector(t)
        if vec:
            history_embeddings.append(vec)

    accumulated_for_db = []
    total_generated = 0

    system_prompt = '你现在是“医学证据卡片选题总编 + 临床知识编辑 + 循证医学研究员 + 医生知识资产工程师”。'

    for cluster_id, items in clusters_pool.items():
        if total_generated >= TARGET_TOTAL_TITLES:
            break  # 已达成总目标，跳出循环

        current_materials_str = ""
        # 提取簇内的精华内容作为素材
        for rank, item in enumerate(items[:20]):
            current_materials_str += f"片段{rank + 1} [{item['source']}]: {item['text']}\n"

        history_str = "\n".join(history_topics) if history_topics else "无"

        dynamic_user_prompt = f"""
        【历史选题黑名单】（绝对禁止重复以下方向或题目）：
        {history_str}

        【本次挖掘素材】：
        {current_materials_str}

        【你的任务】：
        你的任务是基于上述【本次挖掘素材】，为医生生产证据卡片【设计选题池】。
        你的最高目标有五个：
        1. 每张卡片必须服务于一个明确的临床决策。
        2. 吸收医生资料、思维导图、医案等。
        3. 必须有明显循证属性（量化指标、证据等级）。
        4. 产出未来可外挂给大模型用于辅助决策的知识资产。
        5. 绝不能与【历史选题黑名单】在临床决策点上重复！如果你发现本次素材只能生成与黑名单高度相似的选题，请放弃生成，直接输出空数组 []。

        一、必须使用的资料范围
        你必须优先并显式整合以下来源来挖掘选题：
        A. 我上传的医生资料（若有）包括但不限于：决策树/思维导图/医案/书籍/证据卡片示例/研究资料/指南整理稿/症状评分表/胃镜病理评分表
        B. 联网搜索
        你必须检索最新公开资料寻找选题灵感，尤其包括：最新国内外指南/专家共识/系统综述/Meta分析/RCT/高质量观察性研究/真实世界研究/疾病管理规范/近3-5年高质量综述
        C. 学术搜索
        优先检索：PubMed/Google Scholar/Cochrane/国内核心数据库可公开获得部分/指南发布机构官网/学协会官网

        二、你对上传资料的使用原则（挖掘选题的依据）
        1. 决策树 / 思维导图的作用：把它视为“医生显性化诊疗逻辑”。提取风险因素分层、症状识别路径、辅助检查路径等，这决定“临床推理顺序”。
        2. 医案与书籍的作用：把它们视为“医生隐性经验的文本化载体”。提取核心病机、理法方药对应关系、误治/漏治等，这决定“为什么这样治”。
        3. 指南与文献的作用：把它们视为“校准器”和“证据增强器”。提取推荐意见、证据等级、与医生经验一致和不一致之处。
        4. 既有证据卡片样例的作用：借用其成品感、医生阅读友好度、栏目意识。

        三、选题机制：先选题，再写卡片
        在正式写证据卡片前，你必须先做“选题推演”。
        每次选题时，先在我上传资料、联网结果和学术搜索结果中提取选题池，然后按照以下九类角度生成候选题目：

        1. 决策冲突型:围绕医生最容易犹豫的点来选题。例如：什么时候以清热化湿为先，什么时候先健脾和胃；某症状背后究竟更偏气滞、湿热、阴虚还是瘀阻；同类表现如何区分不同治法路径。
        2. 鉴别诊断型:围绕“看起来像，但处理不同”的问题来选题。重点写：相似主诉、关键分叉点、舌脉/胃镜/病理/病程/诱因的区别、误判的临床后果。
        3. 病机拆解型:围绕一个复杂病机或关键病机来选题。重点写：核心病机是什么、当前主要病机是什么、为什么不能只盯一个证型、病机转化后治法如何变化。
        4. 症状切入口型:围绕临床高频、但背后病机不单一的症状来选题。例如：烧心、胃脘胀满、胃脘刺痛、嗳气、咽堵、纳呆、背沉背痛。但必须写出“同一症状，不同处理路径”。
        5. 检查结果驱动型:围绕胃镜、病理、幽门螺杆菌、胆汁反流、萎缩、肠化、异型增生等结果来选题。重点写：某检查发现如何改变辨证权重、某病理结果如何改变随访和干预优先级、某结果出现后治法是否应升级或调整。
        6. 用药决策型:围绕“什么情况下加什么、减什么、避什么”来选题。重点写：加减触发条件、药物配伍逻辑、与基础方的关系、剂量侧重、安全边界。不能只罗列药物。
        7. 预后与随访型:围绕“什么时候需要更密切随访、什么时候可观察、什么指标提示风险上升”来选题。重点写：病程风险、复查触发条件、干预有效性的判定指标。
        8. 生活方式干预型:围绕饮食、情志、作息、分餐、戒烟、低盐、劳逸等内容来选题。必须把它做成“能影响决策质量的内容”，而不是泛泛养生建议。
        9. 经验-证据碰撞型:围绕医生经验与现代循证之间一致的地方、不一致的地方、互补的地方来选题。这类题目最容易形成差异化内容和讨论价值。

        四、选题去同质化规则
        你必须把“避免千篇一律”当作硬约束。每个专区内连续生成多张卡片时，禁止出现以下情况：
        - 连续多张都用同一题目句式
        - 连续多张都以“某某证治疗方案”命名
        - 连续多张都只是在症状、证型、方药上做替换
        - 连续多张都采用同一种切入角度
        - 连续多张的核心段落结构几乎一致
        - 连续多张都只讲治疗，不讲鉴别与边界
        - 连续多张都没有明确的证据指标

        你必须主动拉开差异，至少从以下维度错开：题目句式、切入角度、问题层级、病机层级、检查层级、决策层级、证据层级、适用人群层级、写作重心。
        题目允许的风格应该是多样的，例如：决策判断式、鉴别分析式、证据比较式、病机拆解式、治疗时机式、误区纠偏式、随访管理式、药物加减式、检查结果解读式。但无论题目风格如何变化，都必须最终落到可执行的临床决策上。

        【标题字数与精炼红线】（最高优先级）：
        医生在手机上浏览专区时，标题必须极度精简、一目了然。
        1. 每个卡片标题的字数必须严格控制在 10 到 15 个字之间！
        2. 剔除所有不必要的连接词、助词和冗长的状语，提炼核心医学实体。
        3. 严禁使用提问式的长复句。
        【错误示范（35字）】：肝硬化腹水患者体位变化对心输出量和血管阻力的影响是否影响利尿剂使用时机
        【正确示范（13字）】：肝硬化腹水体位与利尿剂时机
        【错误示范（22字）】：幽门螺杆菌根除后针对不同病理类型的随访策略
        【正确示范（14字）】：Hp根除后不同病理随访策略
        
        五、最终输出要求（仅输出选题清单）
        每次先给出一组候选题目（建议5-8个）。
        
        【本轮任务】
        1. 输出 10 个证据卡片候选选题。
        2. 绝对禁止出现任何中文或英文标点符号。
        3. 严格遵守 10-15 字的字数限制。
        4. 请仔细对比【历史选题黑名单】，主动拉开差异。
        5. 🌟 为每个卡片标题生成一个简短的原因说明（主要解释为什么从临床决策角度提取这个标题）。
        6. 请严格输出一个标准的JSON数组格式，单占一行，必须使用双引号。
        🌟 格式必须严格如下：
        [{{"title": "标题1", "reason": "提取原因说明1"}},{{"title": "标题2", "reason": "提取原因说明2"}}]
        如果实在没有新角度，输出 []。
        """

        try:
            # 请求大模型
            topics_raw = await request_qwen_async(system_prompt, dynamic_user_prompt)

            # 解析 JSON 数组
            try:
                new_topics = extract_json_array_from_text(topics_raw)
            except Exception:
                # 兜底解析
                match = re.search(r'\[.*\]', topics_raw, re.DOTALL)
                new_topics = json.loads(match.group(0)) if match else []

            if not new_topics:
                continue

            # 过滤黑名单并清洗数据格式
            valid_new_topics = []
            for item in new_topics:
                if isinstance(item, dict) and "title" in item and "reason" in item:
                    title = str(item["title"]).strip()
                    reason = str(item["reason"]).strip()

                    if title and title not in history_topics:
                        valid_new_topics.append({"title": title, "reason": reason})

            for item in valid_new_topics:
                if total_generated >= TARGET_TOTAL_TITLES:
                    break

                title = item["title"]
                name_repeat = 0  # 默认非重复

                # 🌟 3. 核心：获取当前生成的标题的向量 (自动生成并写入本地 jsonl 缓存)
                topic_vec_list = get_cached_vector(title)

                # 🌟 4. 计算与历史库的相似度
                if topic_vec_list and history_embeddings:
                    topic_vec = np.array([topic_vec_list])
                    history_mat = np.array(history_embeddings)
                    similarities = cosine_similarity(topic_vec, history_mat)[0]
                    max_sim = np.max(similarities)

                    if max_sim > 0.82:
                        print(f"⚠️ 触发防重叠机制: [{title}] 与历史相似度最高达 {max_sim:.2f}，标记为重复(1)。")
                        name_repeat = 1

                # 🌟 5. 将当前新向量加入到本轮上下文的历史库中，防止本轮自己生成重复卡片
                if topic_vec_list:
                    history_embeddings.append(topic_vec_list)

                # 历史黑名单依然只记录 title
                history_topics.append(item["title"])
                item["name_repeat"] = name_repeat
                # 数据库累加池存入完整的对象 {"title": ..., "reason": ...}
                accumulated_for_db.append(item)
                total_generated += 1

                # 每满 5 个插入一次
                if len(accumulated_for_db) >= BATCH_INSERT_SIZE:
                    insert_file_card_names(file_id, user_id, accumulated_for_db)
                    accumulated_for_db.clear()

            # 短暂等待以防止 Qwen 频控限制
            await asyncio.sleep(2)

        except Exception as e:
            print(f"[卡片标题 LLM 生成异常]: {e}")

    # 如果最后还有剩余未能满足 5 个的元素，一次性清空插入
    if accumulated_for_db:
        insert_file_card_names(file_id, user_id, accumulated_for_db)
        accumulated_for_db.clear()

# ================= 批次后台处理任务 =================
async def background_process_pdf_batch(task_id: str, batch_id: str, file_path: str, user_id: str = '', file_name: str = '', file_id: int = 0):
    """
    批次模式后台处理：解析本地文件 -> 分块 -> 向量化入库（共享 batch_id）
    支持文件类型：PDF（含xmind导出）、TXT
    Word(docx/doc) 已在接口层转换为 PDF 后传入
    """
    result_file  = os.path.join(RESULT_DIR, f"{task_id}.json")
    detected_lang = "zh"
    chunked_data  = []

    try:
        ext = os.path.splitext(file_path)[-1].lower()
        print()
        print(f"文件类型是：{ext}")
        # ===== 分支一：TXT 文件 =====
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text_content = f.read()
            detected_lang = is_chinese_or_english(text_content)
            chunked_data  = chunk_text_data(
                [{"type": "text", "text": text_content}], split_to_length=200
            )
            db_result = process_and_insert_to_milvus(
                task_id, chunked_data, batch_id=batch_id, type='txt',
                user_id=user_id, file_id=file_id
            )
            # 🌟 ===== 分支三：Excel 文件 (新增) =====
        # elif ext in ['.xlsx', '.xls', '.csv']:
        #     print(f"[批次 {batch_id}] 任务 {task_id} 检测到表格类型({ext})，开始行列上下文绑定解析...")
        #
        #     try:
        #         df_dict = {}
        #
        #         # 1. 区分文件类型进行读取
        #         if ext == '.csv':
        #             # 读取 CSV 时，加入编码 fallback 机制（应对国内常见的 GBK 编码）
        #             try:
        #                 df = pd.read_csv(file_path, encoding='utf-8')
        #             except UnicodeDecodeError:
        #                 df = pd.read_csv(file_path, encoding='gbk')
        #             # 将单张表包装成字典结构，保持和 Excel 一致
        #             df_dict = {"CSV数据": df}
        #         else:
        #             # 读取 Excel 所有的 Sheet 表
        #             df_dict = pd.read_excel(file_path, sheet_name=None)
        #
        #         text_content_list = []
        #
        #         # 2. 统一的数据转换逻辑
        #         for sheet_name, df in df_dict.items():
        #             # 删除全空的行和列
        #             df = df.dropna(how='all').dropna(axis=1, how='all')
        #             if df.empty:
        #                 continue
        #
        #             headers = df.columns.tolist()
        #             # 遍历每一行，将结构化数据转换为带有上下文的自然语言段落
        #             for index, row in df.iterrows():
        #                 row_texts = []
        #                 for col in headers:
        #                     val = row[col]
        #                     # 过滤掉空值（NaN / NaT 等）
        #                     if pd.notna(val) and str(val).strip() != "":
        #                         # 将列名和单元格值绑定
        #                         row_texts.append(f"{col}: {val}")
        #
        #                 if row_texts:
        #                     # 组装成一句话，保留了所在的表名和字段含义
        #                     row_str = f"【表格数据 - 工作表:{sheet_name} 第{index + 1}行】 " + ", ".join(row_texts) + "。"
        #                     text_content_list.append(row_str)
        #
        #         # 将所有转换后的行用换行符连接
        #         full_excel_text = "\n".join(text_content_list)
        #
        #         if not full_excel_text.strip():
        #             raise Exception(f"{ext} 文件中未提取到有效数据")
        #
        #         detected_lang = is_chinese_or_english(full_excel_text)
        #
        #         # 传入现有的分块函数
        #         chunked_data = chunk_text_data(
        #             [{"type": "text", "text": full_excel_text}], split_to_length=400
        #         )
        #
        #         # 存入向量库 (可以将 type 统一标记为 'table' 或者保留 'excel')
        #         db_result = process_and_insert_to_milvus(
        #             task_id, chunked_data, batch_id=batch_id, type='table',
        #             user_id=user_id, file_id=file_id
        #         )
        #     except ImportError:
        #         raise Exception("缺少 pandas 库，请执行 pip install pandas")
        #     except Exception as e:
        #         raise Exception(f"表格解析失败: {str(e)}")

        # 🌟 ===== 分支二：图片文件 (新增) =====
        elif ext in ['.png', '.jpg', '.jpeg']:
            print(f"[批次 {batch_id}] 任务 {task_id} 检测到图片类型，开始 OCR 识别...")
            text_content = await extract_text_from_image(file_path)

            if not text_content or not text_content.strip():
                raise Exception("图片 OCR 识别未能提取到有效文本")

            detected_lang = is_chinese_or_english(text_content)
            # 组装成分块所需的格式
            chunked_data = chunk_text_data(
                [{"type": "text", "text": text_content}], split_to_length=200
            )
            print(f"解析图片中的内容：")
            print(chunked_data)
            print()
            # 入库，type 标记为 'image'
            db_result = process_and_insert_to_milvus(
                task_id, chunked_data, batch_id=batch_id, type='image',
                user_id=user_id, file_id=file_id
            )
        # ===== 分支二：PDF / xmind(PDF导出) 文件 =====
        else:
            pdf_type = detect_pdf_type(file_path)
            print(f"[批次 {batch_id}] 任务 {task_id} 检测到文档类型: {pdf_type}")

            if pdf_type == "xmind":
                detected_lang = "zh"

                # 【分支 A】：用户直接上传了原生 .xmind 文件
                if file_path.lower().endswith('.xmind'):
                    print(f"[批次 {batch_id}] 任务 {task_id} 使用原生 xmind 树状解析模式")

                    # 1. 直接读取原生树状结构（自带完美的父子层级，无需 X,Y 坐标）
                    # xmind_to_dict 会返回一个嵌套的字典，直接转为字符串给大模型
                    try:
                        xmind_dict = xmind_to_dict(file_path)
                        json_data = str(xmind_dict)
                    except Exception as e:
                        print(f"解析原生 xmind 文件失败: {e}")
                        json_data = "{}"

                    # 2. 原生专属 Prompt：直接告诉模型这是树状结构，无需猜测方向
                    prompt = f"""
                        这是一个原生思维导图（Xmind）导出的纯净树状数据结构。
                        数据本身已经通过嵌套层级完美体现了逻辑关联，不需要你进行任何坐标推断。
            
                        请你扮演一位极其严谨的知识提取专家，执行以下任务：
            
                        1. 全面提取与深度解析（绝不遗漏）：请仔细阅读这份层级数据，理解核心主题后，**提取其中的所有内容，绝不要只做概括性总结**。
                           你需要逐一深入剖析每一个分支，将该分支下属的所有子节点、微小细节、具体数据等内容**全部囊括进去**。务必做到“每个分支越详细越好”，100%保留原始信息。
            
                        2. 整合输出（严格纯文本模式）：将整个导图表达的内容融会贯通，转化为一篇结构完整、内容极其详实的叙述性长文。
                           - 【绝对禁止 Markdown】：不要使用任何 Markdown 语法！不要用 '#' 写标题，不要用 '**' 加粗，不要用 '*' 或 '-' 打列表，不要写代码块。
                           - 【纯文本段落】：只输出干干净净的汉字和基础标点符号。请仅通过**换行（自然段落）**以及过渡性的连接词（如“首先”、“此外”、“具体而言”）来体现分支间的递进和并列关系。
                           - 再次强调：不要省略任何细节，不要生成任何排版符号，只要详实的纯文本段落！
                        数据如下：
                        {json_data}
                    """

                # 【分支 B】：用户上传的是思维导图导出的 PDF 文件
                else:
                    print(f"[批次 {batch_id}] 任务 {task_id} 使用 PDF 坐标系推断模式")
                    # 依赖您现有的 PDF 坐标提取逻辑
                    json_data = extract_xmind_to_text(file_path)

                    # 使用您之前确认过的、基于坐标推断的 Prompt
                    prompt = f"""
                        这是一个从思维导图导出的节点数据，包含了文本以及它们在页面上的物理坐标（X为横坐标，Y为纵坐标）。

                        请你扮演一位极其严谨的知识提取专家，执行以下任务：

                        1. 隐式还原逻辑：请注意，思维导图通常是“中心发散”结构的。请先通过坐标分布找出位于中心的“核心主题节点”。然后，根据其余节点相对于中心节点的物理距离、方向以及聚集规律，动态推断各个节点之间的"父子、并列"等逻辑关联。

                        2. 全面提取与深度解析（绝不遗漏）：理解这个思维导图的主题后，请**提取其中的所有内容，绝不要只做概括性总结**。
                           你需要逐一深入剖析每一个分支，将该分支下属的所有子节点等内容**全部囊括进去**，100%保留原始信息。

                        3. 整合输出（严格纯文本模式）：转化为一篇内容极其详实的叙述性长文。
                       - 【绝对禁止 Markdown】：不要使用任何 Markdown 语法（无 '#' 标题，无 '**' 加粗，无列表符号）。
                       - 【纯文本段落】：只输出纯文本。仅通过**换行**和自然的文字叙述来划分段落。
                       - 再次强调：不要省略任何细节，拒绝任何特殊排版字符！
                        数据如下：
                        {json_data}
                    """

                # 调用大模型生成长文本
                xmind_str = await request_qwen_async("", prompt)

                # 打印检查（可用于调试）

                # 将详实的文本分块并存入 Milvus 向量数据库
                chunked_data = chunk_text_data(
                    [{"type": "text", "text": xmind_str}], split_to_length=200
                )
                print(chunked_data)
                db_result = process_and_insert_to_milvus(
                    task_id, chunked_data, batch_id=batch_id, type='xmind',
                    user_id=user_id, file_id=file_id
                )

            else:
                # 1. GPU 解析
                async with gpu_lock:
                    print(f"[批次 {batch_id}] 任务 {task_id} 开始 GPU 解析...")
                    loop = asyncio.get_running_loop()
                    data_lis, detected_lang = await loop.run_in_executor(
                        None,
                        pdf_parser_engine.pdf_parsing,
                        file_path,
                        None,
                        False
                    )

                # 2. 分块
                chunked_data = chunk_text_data(data_lis, split_to_length=200)
                print()
                print("分块内容 ：")
                print(chunked_data)
                print()
                # 3. 向量化并入库（共用 batch_id）
                print(f"[批次 {batch_id}] 任务 {task_id} 开始入库...")
                db_result = process_and_insert_to_milvus(task_id, chunked_data, batch_id=batch_id, type='pdf', user_id=user_id, file_id=file_id)

        # --- 判定 Milvus 入库结果 ---
        if isinstance(db_result, dict) and db_result.get("status") == "success":
            # 🌟 全部流程成功：更新数据库状态为 1
            update_file_parse_status(file_id, 1)

            # 🌟 新增：触发卡片标题提炼
            embeddings = db_result.get("embeddings", [])
            texts = db_result.get("texts", [])
            metadata = db_result.get("metadata", [])

            if embeddings:
                print(f"[批次 {batch_id}] 任务 {task_id} 向量入库完毕，开始智能提炼卡片标题...")
                # user_id 强转为 int 方便落库
                # 生成卡片
                await mine_card_titles_for_file(file_id, int(user_id), embeddings, texts, metadata)
                print(f"[批次 {batch_id}] 任务 {task_id} 卡片标题提炼全部完成！")
        else:
            # 入库环节显式返回失败：更新状态为 2
            update_file_parse_status(file_id, 2)


        # 保存任务完成状态
        result_data = {
            "task_id":     task_id,
            "batch_id":    batch_id,
            "status":      "completed",
            "language":    detected_lang,
            "chunk_count": len(chunked_data),
            "db_status":   db_result
        }
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"[批次 {batch_id}] 任务 {task_id} 完成！共 {len(chunked_data)} 个分块入库。")

    except Exception as e:
        print(f"[批次 {batch_id}] 任务 {task_id} 失败: {str(e)}")
        error_str = str(e)
        # 🌟 新增：拦截 PDF 加密异常，替换为友好的用户提示
        if "encrypted" in error_str.lower() or "password" in error_str.lower():
            friendly_error_msg = "文档加密，请解密后重新上传"
        else:
            friendly_error_msg = type(e).__name__ + ": " + error_str

        # 🌟 捕获任何阶段的异常（下载后解析失败、分块失败等）：更新状态为 2
        update_file_parse_status(file_id, 2)

        error_data = {
            "task_id":   task_id,
            "batch_id":  batch_id,
            "status":    "failed",
            "error_msg": friendly_error_msg
        }
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(error_data, f, ensure_ascii=False, indent=2)

    finally:
        # 处理完毕后清理临时文件
        if os.path.exists(file_path):
            os.remove(file_path)



def mock_db_execute_update(files: List[FileCallbackItem]):
    """使用 PyMySQL 执行实际的数据库写入和更新操作"""
    if not files:
        return

    insert_card_data   = []
    update_status_data = []

    for file in files:
        file_id = int(file.id)
        user_id = int(file.user_id)

        card_name = "安神抗癫方联合西药治疗癫痫中血清 SOD 水平回升幅度作为氧自由基清除达标及减药"
        name_repeat = 0
        status = 1
        name_info = "描述xxxxxx"
        status_time = 1776594104631
        insert_card_data.append((file_id, card_name, user_id, name_repeat, status, name_info, status_time))
        update_status_data.append((file_id,))

    conn   = None
    cursor = None

    try:
        conn   = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()

        insert_sql = """
            INSERT INTO evidence_file_card_name (file_id, card_name, user_id, name_repeat, status, name_info, status_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.executemany(insert_sql, insert_card_data)
        print(f"[DB LOG] 成功插入 {cursor.rowcount} 条记录到 evidence_file_card_name")

        # # 批量更新 evidence_file_info 表的状态
        # update_sql = """
        #     UPDATE evidence_file_info
        #     SET file_status = 1
        #     WHERE id = %s
        # """
        # cursor.executemany(update_sql, update_status_data)
        # print(f"[DB LOG] 成功更新 {cursor.rowcount} 条记录的 file_status")

        conn.commit()

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"[DB ERROR] 数据库操作失败，已回滚: {e}")
        raise e

    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ================= 🌟 新增：手动触发生成卡片的请求模型 =================
class GenerateTitlesRequest(BaseModel):
    file_id: int = Field(..., description="要生成卡片的文件ID")
    user_id: int = Field(..., description="所属用户ID")


# ================= 🌟 新增：从向量库查询并生成卡片的后台任务 =================
async def background_generate_titles(file_id: int, user_id: int):
    """
    根据 file_id 查向量库，并调用卡片生成逻辑
    """
    try:
        print(f"[手动触发] 开始为文件ID {file_id} 查库并生成卡片标题...")

        # 1. 查 Milvus 库 (提取对应的 text 和 vector)
        # 注意：这里假设你在 Milvus 中的 file_id 字段类型是整型 (Int64)
        results = milvus_client.query(
            collection_name=MILVUS_COLLECTION_MAIN,
            filter=f"file_id == {file_id}",
            output_fields=["text", "vector"]
        )

        if not results:
            print(f"[手动触发] 文件ID {file_id} 在向量库中未找到任何切片数据！(可能是文件未解析成功或已被删除)")
            return

        embeddings_list = []
        texts = []
        metadata = []

        # 2. 组装数据供大模型提炼使用
        for item in results:
            if "vector" in item and "text" in item:
                embeddings_list.append(item["vector"])
                texts.append(item["text"])
                metadata.append(f"来源文件ID: {file_id}")

        if not embeddings_list:
            print(f"[手动触发] 文件ID {file_id} 没有有效的向量数据！")
            return

        print(f"[手动触发] 查找到 {len(texts)} 个分块，开始智能提炼卡片...")

        # 3. 调用核心生成逻辑
        await mine_card_titles_for_file(file_id, user_id, embeddings_list, texts, metadata)

        print(f"[手动触发] 文件ID {file_id} 卡片标题提炼全部完成！")

    except Exception as e:
        print(f"[手动触发] 文件ID {file_id} 生成卡片标题异常: {e}")





# ================= 接口一：OSS URL 批量提交解析入库 =================

async def download_convert_and_process(file_url: str, download_path: str, ext: str, task_id: str, batch_id: str,
                                       user_id: str, file_name: str, file_id: int):
    """后台全流程：独立线程下载 -> (转PDF) -> 解析分块入库"""
    loop = asyncio.get_running_loop()
    final_path = download_path

    try:
        # ── 1. 在后台执行下载 ──
        def download_file_sync(url, path):
            response = requests.get(url, timeout=600)
            response.raise_for_status()
            with open(path, 'wb') as f:
                f.write(response.content)

        await loop.run_in_executor(None, download_file_sync, file_url, download_path)
        print(f"[后台下载成功] {file_name}  ->  {download_path}")

        # ── 2. 在后台执行 Office 转 PDF ──
        if ext in ('.docx', '.doc', '.ppt', '.pptx'):
            final_path = os.path.join(UPLOAD_DIR, f"{task_id}.pdf")
            await loop.run_in_executor(None, convert_office_to_pdf_sync, download_path, UPLOAD_DIR)
            # 转换成功后删除源文件
            if os.path.exists(download_path):
                os.remove(download_path)

    except Exception as e:
        print(f"❌ [任务 {task_id}] 下载或转换失败: {e}")
        # 失败时更新 MySQL 状态为 2
        update_file_parse_status(file_id, 2)

        # 记录失败状态到文件
        error_data = {
            "task_id": task_id,
            "batch_id": batch_id,
            "status": "failed",
            "error_msg": f"文件下载或转换异常: {str(e)}"
        }
        with open(os.path.join(RESULT_DIR, f"{task_id}.json"), "w", encoding="utf-8") as f:
            json.dump(error_data, f, ensure_ascii=False, indent=2)

        # 清理残余文件
        if os.path.exists(download_path):
            os.remove(download_path)
        return  # 直接终止，不再进入后续解析

    # ── 3. 下载转换均成功，进入核心解析环节 ──
    # 这里直接调用原有的处理入口
    await background_process_pdf_batch(task_id, batch_id, final_path, user_id, file_name, file_id)


@app.post("/api/v1/analysis_file")
async def create_upload_batch_from_url(
        background_tasks: BackgroundTasks,
        files: List[FileCallbackItem]
):
    """
    接收线上 OSS 文件列表，极速返回响应。
    下载、解析、分块、入库全部交由后台任务处理。
    """
    if not files:
        raise HTTPException(status_code=400, detail="请至少提供一个文件")

    SUPPORTED_EXTS = {'.pdf', '.xmind', '.txt', '.docx', '.doc', '.ppt', '.pptx', '.png', '.jpg', '.jpeg'}

    batch_id = str(uuid.uuid4())
    task_list = []

    for file in files:
        file_url = file.file_url
        file_name = file.file_name
        ext = os.path.splitext(file_name.lower())[-1]

        if ext not in SUPPORTED_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"文件 [{file_name}] 格式不支持"
            )

        task_id = str(uuid.uuid4())
        download_path = os.path.join(UPLOAD_DIR, f"{task_id}{ext}")

        # ── 步骤 1：仅初始化任务状态文件，极速完成 ──
        initial_status = {
            "task_id": task_id,
            "batch_id": batch_id,
            "file_id": file.id,
            "file_name": file_name,
            "user_id": file.user_id,
            "status": "downloading"  # 状态改为下载中
        }
        with open(os.path.join(RESULT_DIR, f"{task_id}.json"), "w", encoding="utf-8") as fp:
            json.dump(initial_status, fp, ensure_ascii=False)

        # ── 步骤 2：把【下载+转换+解析】一口气全塞进后台 ──
        background_tasks.add_task(
            download_convert_and_process,
            file_url,
            download_path,
            ext,
            task_id,
            batch_id,
            file.user_id,
            file_name,
            file.id
        )

        task_list.append({
            "task_id": task_id,
            "file_id": file.id,
            "filename": file_name
        })

    # 接口不等待下载，立刻返回响应给客户端
    return JSONResponse(content={
        "code": 200,
        "msg": f"批量提交成功，共 {len(task_list)} 个文件已加入后台下载与解析队列",
        "batch_id": batch_id,
        "tasks": task_list
    })


#
# @app.post("/api/v1/analysis_file")
# async def create_upload_batch_from_url(
#     background_tasks: BackgroundTasks,
#     files: List[FileCallbackItem]
# ):
#     """
#     接收线上 OSS 文件列表，自动下载后解析、分块并写入向量库。
#
#     支持格式：PDF、xmind（PDF导出格式）、TXT、Word（docx / doc）
#     Word 文件会先通过 LibreOffice 转换为 PDF 再处理。
#
#     请求体示例：
#     [
#       {
#         "file_url": "https://image.evimed.com/oss/evidence-card/xxx.pdf",
#         "user_id": "5724839605178470186",
#         "file_name": "(2006_BSG)肝硬化腹水的管理指南.pdf",
#         "id": 9
#       }
#     ]
#
#     立即返回 batch_id 与每个文件对应的 task_id，后续可通过 task_id 查询进度。
#     """
#     if not files:
#         raise HTTPException(status_code=400, detail="请至少提供一个文件")
#
#     # SUPPORTED_EXTS = {'.pdf', '.xmind', '.txt', '.docx', '.doc', '.ppt', '.pptx', '.xlsx', '.xls', '.csv'}
#     SUPPORTED_EXTS = {'.pdf', '.xmind', '.txt', '.docx', '.doc', '.ppt', '.pptx', '.png', '.jpg', '.jpeg'}
#
#     # 整批共用同一个 batch_id
#     batch_id  = str(uuid.uuid4())
#     task_list = []
#
#     for file in files:
#         file_url   = file.file_url
#         file_name  = file.file_name
#         ext        = os.path.splitext(file_name.lower())[-1]
#
#         if ext not in SUPPORTED_EXTS:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"文件 [{file_name}] 格式不支持，仅允许 PDF、xmind、TXT、Word(docx/doc)"
#             )
#
#         task_id       = str(uuid.uuid4())
#         download_path = os.path.join(UPLOAD_DIR, f"{task_id}{ext}")
#         final_path    = download_path
#
#         # ── 步骤 1：从 OSS 异步下载文件（🌟 修复阻塞问题） ──
#         try:
#             # 使用 asyncio.to_thread 或 run_in_executor 将同步下载放入独立线程，不阻塞主线程
#             loop = asyncio.get_running_loop()
#
#             def download_file_sync(url, path):
#                 response = requests.get(url, timeout=600)  # 给大文件长一点的超时时间
#                 response.raise_for_status()
#                 with open(path, 'wb') as f:
#                     f.write(response.content)
#
#             await loop.run_in_executor(None, download_file_sync, file_url, download_path)
#             print(f"[下载成功] {file_name}  ->  {download_path}")
#
#         except Exception as e:
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"下载文件 [{file_name}] 失败: {str(e)}"
#             )
#
#         # ── 步骤 2：Word 文件同步转换为 PDF ──
#         if ext in ('.docx', '.doc', '.ppt', '.pptx'):
#             final_path = os.path.join(UPLOAD_DIR, f"{task_id}.pdf")
#             try:
#                 loop = asyncio.get_running_loop()
#                 await loop.run_in_executor(
#                     None, convert_office_to_pdf_sync, download_path, UPLOAD_DIR
#                 )
#                 # 转换成功后，删除原始的 docx/ppt 临时文件
#                 if os.path.exists(download_path):
#                     os.remove(download_path)
#             except Exception as e:
#                 # 如果转换失败也要清理临时文件
#                 if os.path.exists(download_path):
#                     os.remove(download_path)
#                 raise HTTPException(
#                     status_code=500,
#                     detail=f"文件 [{file_name}] 转PDF失败: {str(e)}"
#                 )
#
#         # ── 步骤 3：初始化任务状态文件 ──
#         initial_status = {
#             "task_id":   task_id,
#             "batch_id":  batch_id,
#             "file_id":   file.id,
#             "file_name": file_name,
#             "user_id":   file.user_id,
#             "status":    "processing"
#         }
#         with open(os.path.join(RESULT_DIR, f"{task_id}.json"), "w", encoding="utf-8") as fp:
#             json.dump(initial_status, fp, ensure_ascii=False)
#
#         # ── 步骤 4：加入后台解析队列（解析 + 分块 + 入库，共用 batch_id）──
#         background_tasks.add_task(
#             background_process_pdf_batch,
#             task_id,
#             batch_id,
#             final_path,
#             file.user_id,
#             file_name,
#             file.id  # 传入文件ID
#         )
#         task_list.append({
#             "task_id":  task_id,
#             "file_id":  file.id,
#             "filename": file_name
#         })
#
#     return JSONResponse(content={
#         "code":     200,
#         "msg":      f"批量提交成功，共 {len(task_list)} 个文件已加入解析队列",
#         "batch_id": batch_id,
#         "tasks":    task_list
#     })




# ================= 🌟 新增接口三：根据 file_id 单独生成卡片标题 =================
@app.post("/api/v1/generate_titles")
async def generate_titles_endpoint(
        req: GenerateTitlesRequest,
        background_tasks: BackgroundTasks
):
    """
    提供给前端/业务端的独立接口：
    传入 file_id 和 user_id，系统会从 Milvus 中提取该文件的数据，并生成 10 个卡片标题入库。
    """
    if req.file_id <= 0:
        raise HTTPException(status_code=400, detail="非法的 file_id")

    # 加入后台异步任务，避免 HTTP 请求超时
    background_tasks.add_task(
        background_generate_titles,
        req.file_id,
        req.user_id
    )

    return JSONResponse(content={
        "code": 200,
        "msg": f"已成功将文件ID [{req.file_id}] 的卡片生成任务加入后台队列，稍后可在数据库查看生成结果"
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(PORT), workers=1)
