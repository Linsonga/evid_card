import os
import re
import json
import uuid
import time
import asyncio
import shutil
import subprocess
from datetime import datetime
from typing import List
from urllib.parse import quote
import pymysql
import requests
import pdfplumber
import numpy as np
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Query
from fastapi.responses import JSONResponse
from pymilvus import MilvusClient
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
import uvicorn
from multi_pdf_to_json_queue import PDFParsing, LAYOUT_PATH
from config import (
    DB_CONFIG, CARD_API_URL, EVID_DESC_URL,MILVUS_MAIN_URI, MILVUS_MAIN_TOKEN, MILVUS_COLLECTION_MAIN,
    MILVUS_REFINEDDATA,MONGO_URI, MONGO_DATABASE, MONGO_COLLECTION, LOGIN_URL,GENERATE_PIC_URL,
    UPLOAD_ZONE_URL,USERNAME,PASSWORD
)
from utils import (
    requestQwen, requestQwenMultiTurn, request_qwen_async,
    vector_4b, extract_json_array_from_text,
)
from matcher import QwenEmbeddingMatcher
from audit import AgenticPipeline
from logger import logger

from sklearn.metrics.pairwise import cosine_similarity
import time
from datetime import datetime
from typing import Optional
import numpy as np
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob  # 记得在文件顶部引入 glob
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from utils import vector_4b  # 确保你的文件顶部有这个导入
from database import get_db_connection

app = FastAPI(title="异步 PDF 解析 API", description="支持超大 PDF 的非阻塞解析服务")

# ================= 配置区 =================
BASE_DIR   = os.path.join(os.getcwd(), "api_data")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ======= 配置区 =======
TOKEN_FILE_PATH = "evimed_token.json"

# 全局变量
pdf_parser_engine = None
gpu_lock = asyncio.Lock()  # 保护显卡不被并发撑爆的核心！

milvus_client = MilvusClient(uri=MILVUS_MAIN_URI, token=MILVUS_MAIN_TOKEN)

# 🌟 新增：声明全局的匹配器和质检管道变量
global_matcher = None
global_pipeline = None

# ================= 启动事件 =================
@app.on_event("startup")
async def startup_event():
    global pdf_parser_engine
    global global_matcher, global_pipeline  # 🌟 引入全局变量

    # print("正在加载 YOLO 和 OCR 模型入显存...")
    # pdf_parser_engine = PDFParsing(layout_path=LAYOUT_PATH)
    print("模型加载完毕，API 准备就绪！")

    # 🌟 新增：在系统启动时，且仅在启动时，初始化一次！
    print("🚀 正在初始化全局专区匹配器与质检大脑，请稍候...")
    global_matcher = QwenEmbeddingMatcher()
    global_pipeline = AgenticPipeline()
    print("✅ 全局匹配器和质检管道初始化完成！内存加载完毕。")

    # 🌟 新增：清理僵尸任务逻辑 🌟
    print("正在清理上次异常退出的僵尸任务...")
    try:
        # 启动 MongoDB 监听任务 (非阻塞)
        # asyncio.create_task(mongo_polling_worker())

        # 扫描 task_states 目录下所有的 json 文件
        state_files = glob.glob(os.path.join(STATE_DIR, "*.json"))
        cleaned_count = 0
        for filepath in state_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    state = json.load(f)

                # 如果服务刚启动，发现有任务还是 running，说明是上次异常崩溃遗留的
                if state.get("status") == "running":
                    state["status"] = "interrupted"  # 改为中断状态
                    state["msg"] = "检测到服务曾意外终止，任务已挂起，可通过传入 task_id 续传。"

                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(state, f, ensure_ascii=False, indent=2)
                    cleaned_count += 1
            except Exception:
                continue
        if cleaned_count > 0:
            print(f"✅ 成功将 {cleaned_count} 个僵尸任务重置为挂起状态，现在可以正常断点续传了！")
    except Exception as e:
        print(f"清理僵尸任务失败: {e}")


# ================= 启动事件 =================
def questionToTitle(question):
    """步骤 1 & 7：生成卡片标题"""
    prompt = f"""
    请你根据以下问题:{question}。生成证据卡片的标题。
    以下是对证据专区和证据卡片的理解
    1.证据专区
    “证据专区”是整个医学知识库的分类导航和核心锚点。它不能是教科书式的大杂烩，而必须是精准切中临床痛点的“微靶点”。
       A.聚焦与精准（双实体原则）：
          拒绝泛泛而谈： 不能用“心血管专区”、“肿瘤专区”这种大词，这会让医生觉得毫无新意且失去阅读方向。
          多维度交叉： 专区命名必须包含至少两个医学实体（如：疾病+特定靶点药物、疾病+特定并发症、疾病+新兴技术）。例如：“GLP-1受体激动剂在心血管高危糖尿病患者中的应用专区”、“晚期非小细胞肺癌免疫治疗耐药管理专区”。这样的命名，医生一看就知道里面解决的是具体的临床细分问题。
       B.紧跟前沿与热点（RCT驱动与AI融入）：
          专区的设立要基于近三年的RCT研究趋势，这意味着专区代表了当下医学界最关心、最受争议或最具突破性的领域。
          同时，需要敏锐捕捉**“AI+医疗”**的趋势，建立如“AI影像在微小肺结节鉴别诊断中的应用专区”或“大模型在罕见病辅助诊断中的探索”等前沿专区。
       C.临床实用导向：
          专区设立的终极目的是“帮医生解决日常诊疗中的困惑”，如药物选择、剂量调整、不良反应处理、特殊人群用药等。
    2.证据卡片
    “证据卡片题目”是吸引医生点击阅读的核心“钩子”。它不是科研论文的Title，而是**“学术软文”或“临床决策参考”的标题**。
       A.极强的差异化（拒绝模板化）：
          摒弃重复前缀： 坚决不能出现整个专区所有题目都以“XXX技术在XXX中的应用”开头的情况。医生不是机器，阅读需要新鲜感和层次感。
          多角度切入： 在同一个专区内，卡片题目要从有效性比较、安全性分析、特殊人群（老人/孕妇/肝肾功能不全）、经济学效益、临床指南差异等不同维度展开。
       B.去科研化，强临床化（学术软文基调）：
          不带研究类型： 题目中坚决不写“随机对照试验(RCT)”、“回顾性分析”、“Meta分析”等字眼。医生看卡片是为了直接获取“结论和临床指导”，而不是来做文献审查的。
          拒绝空泛词汇： 杜绝“真实世界中的应用”、“临床观察”这类不知所云的词。题目必须言之有物，直接点出文章的讨论核心。
          制造悬念与对比（学术比较）： 题目应该像高质量的医学媒体文章。例如，不写“A药和B药的疗效分析”，而是写“面对高血压合并微量蛋白尿：A药与B药谁是优选？”或者“A药导致的心动过缓：临床干预时机与策略”。
       C.内容的独立性与连贯性：
          用户在专区内滑动卡片时，看第一篇觉得“解渴”，看第二篇觉得“有新启发”。每张卡片都是一个独立的临床微决策点，合在一起又能拼凑出该专区的完整知识图谱。
    请根据对证据专区和证据卡片的理解，将问题生成对应的证据卡片标题，每个问题生成1个卡片title即可，如果问题相似，仅生成一个标题。
    要求：1.生成证据卡片的标题不能超过30个字。2.绝对禁止出现任何中文或英文标点符号。
    请以 JSON 数组格式返回，例如：["标题1", "标题2"]
    """
    res = requestQwen("你是一名医学专家。", prompt)
    try:
        titles = json.loads(res)
        if isinstance(titles, list):
            return titles
        return [res]
    except json.JSONDecodeError as e:
        # 记录警告，方便你在控制台看到大模型是不是没有按格式输出
        print(f"⚠️ [questionToTitle] 大模型返回非标准 JSON，已降级为单元素列表。报错详情: {e}\n原文: {res[:50]}...")
        return [res]
    except Exception as e:
        print(f"❌ [questionToTitle] 发生未知异常: {e}")
        return [res]



def convert_docx_to_pdf_sync(docx_path: str, output_dir: str):
    """
    调用系统 LibreOffice 将 docx 转换为 pdf
    """
    # libreoffice 无头模式转换命令
    # 注意：Windows 下如果是默认安装，命令可能需要改成完整路径，例如 "C:\\Program Files\\LibreOffice\\program\\soffice.exe"
    command = [
        "soffice",
        "--headless",
        "--convert-to", "pdf",
        docx_path,
        "--outdir", output_dir
    ]

    try:
        # 执行转换，设置超时时间防止卡死
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    except subprocess.CalledProcessError as e:
        print(f"转换失败，错误信息: {e.stderr.decode('utf-8', errors='ignore')}")
        raise Exception("DOCX 转 PDF 失败，请确保服务器已安装 LibreOffice")
    except FileNotFoundError:
        raise Exception("找不到 soffice 命令，请确保服务器安装了 LibreOffice 并配置了环境变量")


# ================= 全局预编译正则表达式 =================
# 🌟 优化点：将正则表达式提至全局，在系统启动时只编译一次，避免每次循环都重复编译
ZH_PATTERN = re.compile(r'[\u4e00-\u9fa5]')
SYMBOL_PATTERN = re.compile(r'[#\|\$_\~\^\/\\<>\*\}]')

# ================= 写入向量库 =================
def is_garbage_text(text: str) -> bool:
    """
    判断是否为乱码/无意义文本（如 OCR 失败、纯公式、全是乱码符号）
    返回 True 表示是垃圾数据，应该丢弃；False 表示正常数据
    """
    # 1. 过滤太短的文本
    if len(text) < 50:
        return True

    text_length = len(text)

    # 2. 计算中文字符占比 (假设你的目标语料主要是中文)
    # 🌟 优化点：使用全局预编译的正则对象
    zh_chars = ZH_PATTERN.findall(text)
    zh_ratio = len(zh_chars) / text_length
    # 如果中文字符占比低于 20%，判定为乱码或纯无意义字母数字
    if zh_ratio < 0.2:
        return True

    # 3. 计算异常符号的密度
    # 🌟 优化点：使用全局预编译的正则对象
    weird_symbols = SYMBOL_PATTERN.findall(text)
    symbol_ratio = len(weird_symbols) / text_length
    # 如果特殊符号占比超过 10%，判定为公式乱码或OCR错误
    if symbol_ratio > 0.1:
        return True

    # 4. 计算空格密度
    space_ratio = text.count(' ') / text_length
    # 如果空格占比超过 25%，通常是 OCR 碎片化导致的
    if space_ratio > 0.25:
        return True

    return False

def process_and_insert_to_milvus(task_id: str, chunked_data: list, batch_id: str = None, type:str ='pdf'):
    """
    遍历分块数据，调用 Embedding 接口，并批量写入 Milvus。
    batch_id: 批次ID，同一批次的多个文件共用同一个 batch_id；不传则默认使用 task_id。
    """
    if batch_id is None:
        batch_id = task_id
    filtered_count = 0
    if not chunked_data:
        return {"status": "error", "msg": "没有有效的分块数据可供插入"}

    try:
        # 遍历数据生成向量并写入 Milvus
        for index, item in enumerate(chunked_data):
            text = item.get("text", "").strip()
            if not text:
                continue
            if type == 'pdf':
                if len(text) > 600:
                    continue
            if is_garbage_text(text):
                # print(f"[清洗拦截] 发现乱码分块 {index}，已丢弃。内容前缀: {text[:30]}")
                filtered_count += 1
                continue

            # 请求 Embedding 接口
            # 使用 params 传递，requests 会自动帮你做 URL 编码，避免特殊字符报错
            try:
                embeddings_list = vector_4b(text)
            except Exception as e:
                print(f"[Milvus] 块 {index} 请求 Embedding API 报错: {e}")
                continue

            # 3. 拼装 Milvus 所需的数据结构
            # 假设你的 Milvus ID 是 VARCHAR 类型。如果是 INT64，你需要把 task_id 转成纯数字 hash
            doc_id = f"{task_id}_{index}"

            to_insert = []


            res_dic = {
                "id": doc_id,
                "vector": embeddings_list,
                "batch_id": batch_id,
                "text": text,
                "date": str(time.time()),
            }
            to_insert.append(res_dic)

            milvus_client.insert(collection_name=MILVUS_COLLECTION_MAIN, data=to_insert)

    except Exception as e:
        return {"status": "error", "msg": f"Milvus 入库失败: {str(e)}"}


# ================= 判断PDF文件类型 =================
def detect_pdf_type(pdf_path: str) -> str:
    """
    升级版：分析 PDF 文件类型
    返回类型: "xmind" | "image" | "text"
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) == 0:
                return "text"

            # ================= 1. 强特征：元数据检测 =================
            metadata = pdf.metadata or {}
            # 将所有元数据的值拼成一个大字符串，统一转小写查找
            meta_string = " ".join([str(v) for v in metadata.values()]).lower()

            if 'xmind' in meta_string or 'mindmap' in meta_string:
                return "xmind"

            # ================= 2. 弱特征：页面尺寸与排版特征检测 =================
            page = pdf.pages[0]

            # (A) 提取出所有的单词/文字块及其坐标
            words = page.extract_words()
            if not words:
                # 没有任何矢量文字，绝对是图片或扫描件
                return "image"

            # 提取纯文本（按默认的行读取模式）
            text = page.extract_text() or ""
            text_length = len(text.strip())

            # (B) 如果文字极少，判定为图片/扫描件（可能有几个水印字，所以阈值设为30）
            if text_length < 30:
                return "image"

            # (C) 核心逻辑：判断是否为“散落分布”的思维导图
            # 正常文章，每一行的字应该拥有相近的 top (Y) 坐标或者左对齐的 x0 坐标。
            # 思维导图的坐标则非常分散。我们通过计算独特 Y 坐标（行）的数量来判断。

            # 统计有多少个不同的 Y 坐标（允许 3 像素的误差）
            unique_y_positions = []
            for w in words:
                y = w['top']
                # 看看这个 y 是否和已有的 y 接近
                if not any(abs(y - uy) < 3 for uy in unique_y_positions):
                    unique_y_positions.append(y)

            # 特征判断：
            # 如果这页纸有大量的文字（比如>50个文字块），但是它们几乎分布在几十个完全不同的 Y 坐标上
            # 并且缺少大段连续的长句子（导图都是短词），大概率就是思维导图

            # 计算平均每个文字块的字符长度（导图通常是短词，平均长度短）
            avg_word_len = sum(len(w['text']) for w in words) / len(words)

            # 如果 独立Y坐标的数量 接近 文字块的总数 (说明文字到处乱飘，不是规整的一行行排版)
            # 且页面宽度大于标准 A4 (很多导图导出时很宽)
            is_scattered = len(unique_y_positions) > (len(words) * 0.4)
            is_short_phrases = avg_word_len < 10

            # 你可以打印这些特征用于调试
            # print(f"坐标分散度: {len(unique_y_positions)}/{len(words)}, 平均词长: {avg_word_len}")

            if is_scattered and is_short_phrases:
                print(f"⚠️ 元数据无 Xmind，但通过排版特征识别为思维导图")
                return "xmind"

            # ================= 3. 兜底：普通文本 =================
            return "text"

    except Exception as e:
        print(f"检测 PDF 类型失败: {e}")
        return "text"

# ================= 分块事件 =================
def is_chinese_or_english(text, threshold=0.05):
    """判断文本主要是中文还是英文"""
    chinese_count = 0
    english_count = 0
    total_count = 0

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
def chunk_text_data(content_list, split_to_length: int = 500):
    """
    核心分块逻辑：输入原本的解析列表，输出分块后的列表
    """
    if not content_list:
        return []

    # 1. 提取需要的条目 (text, title)
    require_items = [x for x in content_list if x.get("type") in ("text", "title")]
    if not require_items:
        return []

    require_list = [x.get("text", "") for x in require_items]
    combined_text = "".join(require_list)

    # 2. 自动识别语言并确定分隔符
    lang_detect = is_chinese_or_english(combined_text)
    if lang_detect == "中文":
        separators = ("。", "！", "!")
    else:
        separators = (". ", ".", "!", "！", "。")

    new_block_list = []

    # 3. 拼接与切分
    for i, con in enumerate(require_items):
        t = (con.get("text") or "").strip()
        if lang_detect == "英文" and t:
            t += " "

            # 安全获取 bbox，如果有的项没有 bbox，给个空列表
        current_bbox = con.get("bbox", [])

        if not new_block_list:
            new_block_list.append({
                "text": t,
                "bbox": [current_bbox] if current_bbox else [],
                "end_oldblock_type": con["type"]
            })
            continue

        last = new_block_list[-1]
        last_text = last["text"]
        end_with_sep = last_text.rstrip().endswith(separators)
        is_last_required = (con is require_items[-1])

        if (
                len(last_text) < split_to_length
                or last["end_oldblock_type"] == "title"
                or (not end_with_sep)
                or is_last_required
        ):
            last["text"] += t
            if current_bbox:
                last["bbox"].append(current_bbox)
            last["end_oldblock_type"] = con["type"]
        else:
            new_block_list.append({
                "text": t,
                "bbox": [current_bbox] if current_bbox else [],
                "end_oldblock_type": con["type"]
            })

    # 4. 结尾小块并入前一块
    if len(new_block_list) >= 2 and len(new_block_list[-1]["text"]) < 300:
        new_block_list[-2]["text"] += new_block_list[-1]["text"]
        new_block_list[-2]["bbox"].extend(new_block_list[-1]["bbox"])
        new_block_list.pop()

    # 5. 组装成你需要的最终格式
    final_res_list = []
    for index, block in enumerate(new_block_list):
        final_res_list.append({
            "text": block["text"],
            "index": index,
            "length": len(block["text"]),
            "bbox": block["bbox"]
        })

    return final_res_list

# ================= xmind 转换 成 text =================
def extract_xmind_to_text(pdf_path):
    nodes = []
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        words = page.extract_words()
        for w in words:
            nodes.append({
                "text": w['text'],
                "x": round(w['x0'], 1), # 保留一位小数
                "y": round(w['top'], 1)
            })
    return json.dumps(nodes, ensure_ascii=False)


# ================= 自动执行完整流程的后台任务 =================
async def auto_process_workflow(batch_id: str, task_ids: list):
    """
    自动执行完整流程：等待所有任务完成 -> 调用 file_create_card
    """
    print(f"[自动流程] 批次 {batch_id} 开始监控，共 {len(task_ids)} 个任务")

    # 等待所有任务完成
    while True:
        all_completed = True
        for task_id in task_ids:
            result_file = os.path.join(RESULT_DIR, f"{task_id}.json")
            if os.path.exists(result_file):
                with open(result_file, "r", encoding="utf-8") as f:
                    task_data = json.load(f)
                    if task_data.get("status") == "processing":
                        all_completed = False
                        break
            else:
                all_completed = False
                break

        if all_completed:
            break
        await asyncio.sleep(5)

    print(f"[自动流程] 批次 {batch_id} 所有任务已完成，开始创建卡片")
    file_create_card(batch_id)
    print(f"[自动流程] 批次 {batch_id} 流程全部完成")

# ================= 批次后台处理任务 =================
async def background_process_pdf_batch(task_id: str, batch_id: str, pdf_path: str):
    """
    批次模式的后台处理：解析 PDF -> 分块 -> 自动向量化入库（使用共享 batch_id）
    """
    result_file = os.path.join(RESULT_DIR, f"{task_id}.json")

    try:
        # ================= 1. 判断文档类型 =================
        pdf_type = detect_pdf_type(pdf_path)
        print(f"Task {task_id} 检测到文档类型: {pdf_type}")

        # ================= 2. 根据类型路由不同的提取逻辑 =================
        full_text = ""
        if pdf_type == "xmind":
            detected_lang = "zh"
            json_data = extract_xmind_to_text(pdf_path)

            prompt = f"""
                这是一个从思维导图（Xmind）导出的节点数据，包含了文本以及它们在页面上的物理坐标。
                X 是横坐标（越小越靠左），Y 是纵坐标（越小越靠上）。
                请你扮演一位逻辑分析专家，执行以下任务：
                隐式还原逻辑：请先在底层根据 X,Y 坐标的分布规律（如从左到右、从上到下的发散关系），理清各个节点之间的“父子、并列”等逻辑关联。
                提取核心大意：理解这个思维导图到底在表达什么主题，包含了哪些核心分支和细节。
                请抛弃列表或树状格式，将整个导图表达的意思融会贯通，用一段通顺、连贯的文字总结出来。只要总结内容，逻辑清晰，语言自然流畅。
                数据如下：
                {json_data}
                """
            xmind_str = await request_qwen_async("", prompt)

            parts = re.split(r'\n\s*\n', xmind_str)
            chunked_data = [{"text": x.strip()} for x in parts if x.strip()]
            print(chunked_data)
            print(type(chunked_data))
            chunked_data = chunk_text_data([{"type": "text", "text": xmind_str}], split_to_length=400)
            print(chunked_data)
            exit()
            db_result = process_and_insert_to_milvus(task_id, chunked_data, batch_id=batch_id, type='xmind')

        else:
            # 1. GPU 解析
            async with gpu_lock:
                print(f"[批次 {batch_id}] 任务 {task_id} 开始 GPU 解析...")
                loop = asyncio.get_running_loop()
                data_lis, detected_lang = await loop.run_in_executor(
                    None,
                    pdf_parser_engine.pdf_parsing,
                    pdf_path,
                    None,
                    False
                )

            # 2. 分块
            chunked_data = chunk_text_data(data_lis, split_to_length=400)

            # 3. 向量化并入库（共用 batch_id）
            print(f"[批次 {batch_id}] 任务 {task_id} 开始入库...")
            db_result = process_and_insert_to_milvus(task_id, chunked_data, batch_id=batch_id)

        # 4. 保存结果状态
        result_data = {
            "task_id": task_id,
            "batch_id": batch_id,
            "status": "completed",
            "language": detected_lang,
            "chunk_count": len(chunked_data),
            "db_status": db_result
        }
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"[批次 {batch_id}] 任务 {task_id} 完成！")

    except Exception as e:
        print(f"[批次 {batch_id}] 任务 {task_id} 失败: {str(e)}")
        error_data = {
            "task_id": task_id,
            "batch_id": batch_id,
            "status": "failed",
            "error_msg": type(e).__name__ + ": " + str(e)
        }
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(error_data, f, ensure_ascii=False, indent=2)

    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


# ================= 接口 1：批量提交文件夹 =================
@app.post("/api/v1/upload_batch")
async def create_upload_batch( background_tasks: BackgroundTasks, files: List[UploadFile] = File(..., description="同一批次的多个 PDF/DOCX 文件")):
    """
    接收多个 PDF/DOCX 文件（相当于一个文件夹），生成一个共享 batch_id，
    每个文件独立解析、分块并入库，入库时 batch_id 相同，方便按批次检索。
    立刻返回 batch_id 和每个文件对应的 task_id，后续可用 /get_result/{task_id} 查询单文件进度。
    """
    if not files:
        raise HTTPException(status_code=400, detail="请至少上传一个文件")

    # 整批共用同一个 batch_id
    batch_id = str(uuid.uuid4())
    task_list = []

    for file in files:
        filename_lower = file.filename.lower()
        ext = os.path.splitext(filename_lower)[-1]

        if ext not in ['.pdf', '.docx']:
            raise HTTPException(
                status_code=400,
                detail=f"文件 {file.filename} 格式不支持，只允许 PDF 或 DOCX"
            )

        task_id = str(uuid.uuid4())
        upload_path = os.path.join(UPLOAD_DIR, f"{task_id}{ext}")
        final_pdf_path = os.path.join(UPLOAD_DIR, f"{task_id}.pdf")

        # 保存文件
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # DOCX 同步转 PDF
        if ext == '.docx':
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, convert_docx_to_pdf_sync, upload_path, UPLOAD_DIR)
                if os.path.exists(upload_path):
                    os.remove(upload_path)
            except Exception as e:
                if os.path.exists(upload_path):
                    os.remove(upload_path)
                raise HTTPException(status_code=500, detail=f"文件 {file.filename} 转换失败: {str(e)}")

        # 初始化任务状态
        initial_status = {"task_id": task_id, "batch_id": batch_id, "status": "processing"}
        with open(os.path.join(RESULT_DIR, f"{task_id}.json"), "w", encoding="utf-8") as f:
            json.dump(initial_status, f, ensure_ascii=False)

        # 加入后台队列（解析+分块+自动入库，共用 batch_id）
        background_tasks.add_task(background_process_pdf_batch, task_id, batch_id, final_pdf_path)

        task_list.append({"task_id": task_id, "filename": file.filename})

    # 启动自动流程：等待所有任务完成后自动调用 file_create_card
    # task_ids = [task["task_id"] for task in task_list]
    # background_tasks.add_task(auto_process_workflow, batch_id, task_ids)

    return JSONResponse(content={
        "code": 200,
        "msg": f"批量上传成功，共 {len(task_list)} 个文件已加入解析队列，完成后将自动创建卡片",
        "batch_id": batch_id,
        "tasks": task_list
    })


# ================= 接口 2：查询结果并支持分块 =================
@app.get("/api/v1/get_result/{task_id}")
async def get_task_result( task_id: str, chunk: bool = Query(False, description="是否对结果进行文本分块处理"), chunk_size: int = Query(500, description="分块的目标长度"), store_to_db: bool = Query(False, description="是否将分块数据存入 Milvus 向量库")):
    # processed_data = [{"text":"aaaa"},{"text":"bbb"}]
    # task_id =  '11111'
    # db_result = process_and_insert_to_milvus(task_id, processed_data)
    #
    # print(db_result)
    # exit()
    """
    前端拿着 task_id 来查询结果。
    参数说明：
    - chunk=true: 返回切块后的数据
    - store_to_db=true: 触发向量化并将数据自动存入 Milvus
    """
    result_file = os.path.join(RESULT_DIR, f"{task_id}.json")

    if not os.path.exists(result_file):
        raise HTTPException(status_code=404, detail="任务 ID 不存在或已过期")

    with open(result_file, "r", encoding="utf-8") as f:
        task_data = json.load(f)

    status = task_data.get("status")

    if status == "processing":
        return JSONResponse(content={"code": 202, "msg": "正在排队或解析中，请稍后再试...", "status": "processing"})

    elif status == "failed":
        return JSONResponse(
            content={"code": 500, "msg": "解析失败", "status": "failed", "error": task_data.get("error_msg")})

    elif status == "completed":
        raw_data = task_data.get("data", [])

        # 1. 执行分块
        if chunk or store_to_db:
            # 只要开启入库，就必须先分块
            processed_data = chunk_text_data(raw_data, split_to_length=chunk_size)
            msg = f"解析并分块完成 (分块长度: {chunk_size})"
        else:
            processed_data = raw_data
            msg = "解析完成 (原始版面数据)"

        db_msg = "未触发入库"
        # 2. 只有当明确传递 store_to_db=true 时，才执行向量化和插入
        if store_to_db:
            print(f"[{task_id}] 开始进行向量化和 Milvus 入库...")
            db_result = process_and_insert_to_milvus(task_id, processed_data)
            db_msg = db_result

        return JSONResponse(content={
            "code": 200,
            "msg": msg,
            "status": "completed",
            "db_status": db_msg,
            "data": processed_data
        })

    else:
        return JSONResponse(content={"code": 400, "msg": "未知状态"})


# 增加 zone_name 参数
def query_and_audit(topic_id, title, batch_id, pipeline, materials, zone_name):
    """【封装方法】：制卡成功后，查库组装数据并触发 RL 审核"""
    # conn = pymysql.connect(**DB_CONFIG)
    # cursor = conn.cursor(pymysql.cursors.DictCursor)
    # cursor.execute("SELECT id, title, info, core_conclusion FROM evidence_card WHERE topic_id = %s and title = %s", (topic_id, title))
    # results = cursor.fetchall()
    # cursor.close()
    # conn.close()
    # 🌟 修改点：从连接池获取连接
    conn = get_db_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    try:
        cursor.execute("SELECT id, title, info, core_conclusion FROM evidence_card WHERE topic_id = %s and title = %s",
                       (topic_id, title))
        results = cursor.fetchall()
    finally:
        # 🌟 修改点：使用 try-finally 确保连接一定会归还给连接池
        cursor.close()
        conn.close()

    dummy_input_data = []
    for row in results:
        dummy_input_data.append({
            "id": row["id"],
            "title": row["title"],
            "info": row["info"],
            "core_conclusion": row["core_conclusion"]
        })

    if dummy_input_data:
        # 🌟 改动：将 zone_name 传给 run_round
        pipeline.run_round(dummy_input_data, title, batch_id, materials, zone_name)
    else:
        logger.warning("- 数据库中未查询到记录。")
    return True


def callingCard(topic_id, card_title):
    url = f"{CARD_API_URL}?topicId={topic_id}&cardTitle={quote(card_title)}"
    print()
    print(f"正在发送制卡请求 -> TopicID: {topic_id}, 标题: {card_title}")
    try:
        resp = requests.get(url, timeout=(30, 18000))
        if resp.status_code == 200:
            print(f"制卡成功: {card_title}")
            return True
        print()
        print("卡片接口请求失败...")
        return False
    except Exception as e:
        logger.error(f"制卡请求异常: {str(e)}")
        return False

# 生成4个问题
def five_prompt(topic):
    system_prompt = "你是一名医学循证研究专家。"
    user_prompt = (f"请围绕专题：{topic}，生成 4 个不同的医学研究问题。"
                   f"\n输出 JSON 格式：{{\"questions\": [\"问题1\", \"问题2\", \"问题3\", \"问题4\"]}}")
    res = requestQwen(system_prompt, user_prompt)
    try:
        data = json.loads(res)
        return data.get("questions", [])
    except json.JSONDecodeError as e:
        print(f"⚠️ [five_prompt] JSON解析失败，大模型格式错误。报错详情: {e}\n原文: {res[:50]}...")
        return []
    except Exception as e:
        print(f"❌ [five_prompt] 发生未知异常: {e}")
        return []

# 生成专区题目
def creationTopicTitle(title):
    user_prompt = f"""
    请根据我提供的【证据卡片标题】，生成1个最优的【证据专区题目】。
    【生成要求】：   
    1. 必须符合“微靶点”原则，避免宽泛分类（如“心血管疾病”“内分泌疾病”禁止出现）
    2. 至少包含两个医学实体（如：症状/行为 + 疾病风险 / 机制 / 干预方式）
    3. 优先体现“机制、风险关联、临床管理或干预策略”其中一个维度
    4. 风格需偏“医学专题/学术方向”，而非问题句或口语句
    5. 不超过20个字，简洁有力
    6. 避免与常见大类重复（如“慢病管理”“药物研究”等泛化词）
    7. 具备扩展性（该专区下至少可延展5-10个不同角度的卡片）
    【输出要求】：
    仅输出1个最终专区名称，不要解释
    【输入卡片标题】：
    {title}
    """
    return requestQwen("你是一名医学内容策划专家，擅长将“单一临床问题”抽象为“高质量证据专区名称”。", user_prompt)




# 假设你在外部定义了 EVID_DESC_URL
# EVID_DESC_URL = "..."

# ==========================================
# 1. 登录与 Token 管理机制 (缓存 1 天)
# ==========================================
def get_valid_token():
    """获取 Token，优先从本地缓存读取，过期（1天）或不存在则重新登录"""
    current_time = time.time()

    # 检查本地是否有缓存且未过期 (86400秒 = 1天)
    if os.path.exists(TOKEN_FILE_PATH):
        try:
            with open(TOKEN_FILE_PATH, "r", encoding="utf-8") as f:
                token_data = json.load(f)
                save_time = token_data.get("timestamp", 0)
                if current_time - save_time < 86400:
                    # print("✅ 使用本地缓存的有效 Token")
                    return token_data.get("token")
        except Exception as e:
            print(f"⚠️ 读取本地Token文件异常，将重新登录: {e}")

    # 如果没有有效 Token，执行登录请求
    print("🔄 Token 不存在或已过期，正在请求登录接口...")
    login_data = {
        "username": USERNAME,
        "password": PASSWORD  # ⚠️ 请替换为实际的密码
    }

    try:
        response = requests.post(LOGIN_URL, json=login_data, timeout=10)
        if response.status_code == 200:
            response_data = response.json()
            token = response_data.get("token")

            if token:
                print(f"✅ 登录成功！获取到新 Token: {token}")
                # 保存到本地文件
                with open(TOKEN_FILE_PATH, "w", encoding="utf-8") as f:
                    json.dump({"token": token, "timestamp": current_time}, f)
                return token
            else:
                print("❌ 登录失败，接口未返回 token 字段:", response_data)
        else:
            print(f"❌ 登录请求失败，状态码：{response.status_code}")
    except Exception as e:
        print(f"❌ 发生网络请求错误：{e}")

    return None


# ==========================================
# 2. 生成图片并调用 API 创建专区
# ==========================================
def generate_and_transfer_image(title, describe):
    """请求生成图片，下载流，并带上 Token 提交专区创建表单"""
    # 1. 获取有效 Token
    token = get_valid_token()
    if not token:
        print("❌ 无法获取有效 Token，终止创建专区。")
        return None

    # 2. 生成图片链接
    print(f"正在根据标题 '{title}' 请求生成图片...")
    try:
        gen_response = requests.get(GENERATE_PIC_URL, params={'title': title}, timeout=15)
        if gen_response.status_code != 200:
            print(f"❌ 图片生成失败，状态码: {gen_response.status_code}")
            return None

        image_url = gen_response.text.strip()
        print(f"✅ 成功获取图片链接: {image_url}")
    except Exception as e:
        print(f"❌ 请求生成图片异常: {e}")
        return None

    # 3. 获取图片文件流
    print("正在以【文件流】模式下载该图片...")
    try:
        img_response = requests.get(image_url, stream=True, timeout=15)
        if img_response.status_code != 200:
            print(f"❌ 图片下载失败，状态码: {img_response.status_code}")
            return None
    except Exception as e:
        print(f"❌ 下载图片网络异常: {e}")
        return None

    # 4. 组装数据并上传创建专区
    headers = {
        "token": token
    }

    # 填入大模型生成的专区描述 describe
    data = {
        "title": title,
        "describe": describe,
        "isPub": "0",
        "cardCorrelation": "0",
        "publishAuth": "0",
        "unit": "灵犀医疗",
        "type": "0",
        "users": "16710810141"
    }

    files = {
        "image": ("cover_image.jpg", img_response.content, "image/jpeg")
    }

    print("正在提交图文数据到服务器...")
    try:
        upload_response = requests.post(
            UPLOAD_ZONE_URL,
            headers=headers,
            data=data,
            files=files,
            timeout=20
        )

        if upload_response.status_code == 200:
            res_data = upload_response.json()
            latest_id = res_data.get("data", {}).get("id")

            if latest_id:
                print(f"✅ 上传成功！最新专区 ID 为: {latest_id}")
                return latest_id
            else:
                print("⚠️ 上传成功，但未在返回数据中找到 ID 字段:", res_data)
                return None
        else:
            print(f"❌ 上传失败，状态码: {upload_response.status_code}")
            print("错误详情:", upload_response.text)
            return None

    except Exception as e:
        print(f"❌ 提交图文数据异常: {e}")
        return None


# ==========================================
# 3. 改造原有的 creationZone 函数
# ==========================================
def creationZone(title):
    """
    主入口：获取专区大模型描述 -> 走 API 创建专区
    (替代了原先直接写入 MySQL 的逻辑)
    """
    # 1. 动态获取大模型的“专区描述”
    try:
        response = requests.get(EVID_DESC_URL, params={"desc": title}, verify=False, timeout=15)
        data_text = response.json().get("data", "")
        describe = data_text.split("\n\n")[0] if data_text else "智能生成的专区描述"
    except Exception as e:
        print(f"⚠️ 获取大模型描述失败，将使用默认描述。原因: {e}")
        describe = "智能生成的专区描述"

    # 2. 调用 API 创建流程获取 new_topic_id
    new_topic_id = generate_and_transfer_image(title, describe)

    if new_topic_id:
        print(f"🎉 专区【{title}】创建全流程结束，获得最终 ID: {new_topic_id}")
    else:
        print(f"💔 专区【{title}】创建失败。")

    return new_topic_id

# def creationZone(title):
#     # logger.info(f"开始创建新专区: 【{title}】")
#     try:
#         response = requests.get(EVID_DESC_URL, params={"desc": title}, verify=False)
#         data_text = response.json().get("data", "")
#         describe = data_text.split("\n\n")[0] if data_text else "智能生成的专区描述"
#     except Exception as e:
#         describe = "智能生成的专区描述"
#
#     # 🌟 修改点：从连接池获取连接
#     conn = get_db_connection()
#     cursor = conn.cursor()
#     new_topic_id = None
#     try:
#         sql = """INSERT INTO evidence_topic (title, `describe`, classification, is_pub, publish_auth, card_correlation, `type`) VALUES (%s, %s, 0, 0, 0, 0, 0)"""
#         cursor.execute(sql, (title, describe))
#         conn.commit()
#         new_topic_id = cursor.lastrowid
#     except Exception as e:
#         conn.rollback()
#         logger.error(f"新专区插入数据库失败: {e}")
#     finally:
#         # 🌟 修改点：确保归还连接
#         cursor.close()
#         conn.close()
#
#     return new_topic_id


# ==== Matching 匹配专区  ====
def matching_zone(titles, batch_id, matcher, pipeline, materials, allowed_topic_ids=None):
    for title in titles:
        title_vec = vector_4b(title)
        if not title_vec: continue

        # 🌟 改动：同时接收 topic_id 和 topic_name
        topic_id, topic_name = matcher.cardDistribution(title_vec, title, threshold=0.35, allowed_topic_ids=allowed_topic_ids)

        if topic_id:
            if matcher.filterExistingTitleInTopic(title, title_vec, topic_id):
                continue
            if callingCard(topic_id, title):
                matcher.add_card_to_cache(topic_id, title, title_vec)
                # 🌟 改动：传入 topic_name
                info = query_and_audit(topic_id, title, batch_id, pipeline, materials, topic_name)
        else:
            # 新建专区
            topic_title = creationTopicTitle(title)  # topic_title 就是新专区的名字
            new_topic_id = creationZone(topic_title)
            if not new_topic_id: continue

            related_questions = five_prompt(title)
            for q in related_questions:
                sub_titles = questionToTitle(q)
                for sub_title in sub_titles:
                    sub_vec = vector_4b(sub_title)
                    if not sub_vec: continue

                    if matcher.filterExistingTitleInTopic(sub_title, sub_vec, new_topic_id): continue
                    if callingCard(new_topic_id, sub_title):
                        matcher.add_card_to_cache(new_topic_id, sub_title, sub_vec)
                        # 🌟 改动：传入刚生成的新专区名字 topic_title
                        info = query_and_audit(new_topic_id, sub_title, batch_id, pipeline, materials, topic_title)

    logger.info(f"========== 处理完毕 ==========")




# # ===== 问题创建卡片 =======
# @app.get("/api/v1/question_create_card")
# def question_create_card(question: str):
#     """
#     根据输入问题生成证据卡片。
#     - question: 医学研究问题
#     """
#     if not question or not question.strip():
#         raise HTTPException(status_code=400, detail="question 参数不能为空")
#
#     global_matcher = QwenEmbeddingMatcher()
#     global_pipeline = AgenticPipeline()
#
#     # 1. 生成卡片标题列表
#     topics = questionToTitle(question)
#     if not topics:
#         return JSONResponse(content={"code": 500, "msg": "AI 未能根据问题生成有效标题，请重试"})
#
#     # 2. 匹配专区并生成卡片
#     matching_zone(topics, None, global_matcher, global_pipeline, "")
#
#     return JSONResponse(content={
#         "code": 200,
#         "msg": f"问题卡片生成流程已完成，共处理 {len(topics)} 个标题",
#         "question": question,
#         "topics": topics,
#     })



# # ===== 文件创建卡片 =======
# @app.get("/api/v1/file_create_card")
# def file_create_card(batch_id: str):
#     # ==========================================
#     # 第一步：拉取数据，并增加“数据清洗”逻辑
#     # ==========================================
#     print("正在从 Milvus 向量库提取数据...")
#
#     print(batch_id)
#     results = milvus_client.query(
#         collection_name=MILVUS_COLLECTION_MAIN,
#         filter=f'batch_id == "{batch_id}"',
#         output_fields=["vector", "text"],
#         limit=2000  # 稍微拉大一点，保证过滤后有足够的数据
#     )
#
#     embeddings_list = []
#     texts = []
#     metadata = []
#
#     for item in results:
#         text_content = item["text"].strip()
#
#         # 🌟 优化1：使用增强版的正则特征过滤乱码
#         if is_garbage_text(text_content):
#             continue
#
#         # 🌟 优化1：过滤掉太短的废话片段（比如目录、页码、单一的图表标题）
#         if len(text_content) < 50:
#             continue
#
#         embeddings_list.append(item["vector"])
#         texts.append(text_content)
#         doc_id = item.get("id", "未知ID")
#         metadata.append(f"库内记录ID: {doc_id}")
#
#     embeddings = np.array(embeddings_list)
#     num_chunks, dim = embeddings.shape
#     print(f"✅ 数据清洗完毕，有效文本块: {num_chunks} 条！")
#
#     # ==========================================
#     # 第二步：L2 归一化与聚类
#     # ==========================================
#     embeddings = normalize(embeddings, norm='l2')
#
#     N = 10  # 提取 10 个核心主题
#     print(f"正在将向量聚合成 {N} 个核心簇...")
#
#     kmeans = KMeans(n_clusters=N, random_state=42, n_init="auto")
#     kmeans.fit(embeddings)
#     centroids = kmeans.cluster_centers_
#
#     # ==========================================
#     # 🌟 第三步：计算距离，每个主题提取 Top 3 片段
#     # ==========================================
#     # 计算所有真实向量到 10 个质心的距离矩阵
#     # distances 的 shape 是 (10, num_chunks)
#     distances = pairwise_distances(centroids, embeddings)
#
#     # 获取每个簇距离最近的 Top 3 的索引
#     top_k = 3
#     closest_indices_matrix = np.argsort(distances, axis=1)[:, :top_k]
#
#     # ==========================================
#     # 第四步：打印结果（打包输出）
#     # ==========================================
#     print("\n🎉 成功提取！以下是 Milvus 库中 10 个核心主题的【群组素材】：\n")
#
#     # 用一个列表把所有结果存起来，方便后面发给大模型
#     evidence_materials = []
#
#     for cluster_id, chunk_indices in enumerate(closest_indices_matrix):
#         # print(f"==========【精华主题候选群组 {cluster_id + 1}】==========")
#
#         cluster_texts = []
#         for rank, chunk_index in enumerate(chunk_indices):
#             core_text = texts[chunk_index]
#             source = metadata[chunk_index]
#             cluster_texts.append(f"片段{rank + 1} [{source}]: {core_text}")
#
#             # print(f"👉 支撑片段 {rank + 1} [{source}]:")
#             # print(f"文本: {core_text}\n")
#
#         evidence_materials.append({
#             "cluster_id": cluster_id + 1,
#             "content": "\n".join(cluster_texts)
#         })
#
#     materials = "\n\n".join(
#         f"==========【精华主题候选群组 {item['cluster_id']}】==========\n{item['content']}"
#         for item in evidence_materials
#     )
#
#     system_prompt = '你现在是“医学证据卡片选题总编 + 临床知识编辑 + 循证医学研究员 + 医生知识资产工程师”。'
#     user_prompt = f"""
#     你的任务不是写一篇普通医学科普，也不是机械整理资料，而是基于我提供的全部材料，为医生持续生产“可读、可用、可检索、可沉淀为个人知识资产”的证据卡片【设计选题池】。
#     你的最高目标有五个：
#     1. 每张卡片（每个选题）必须服务于一个明确的临床决策，而不是泛泛介绍疾病。
#     2. 每张卡片（每个选题）必须同时吸收：医生资料（若有）、思维导图/决策树（若有）、医案（若有）、书籍（若有）、指南、文献、联网搜索和学术搜索结果。
#     3. 每张卡片（每个选题）必须有明显循证属性，尤其是量化指标、结局指标、分层条件、适用人群、证据等级或证据强弱。
#     4. 选题之间必须避免千篇一律，不能只是把症状、证型、方药替换一下。
#     5. 产出的选题不仅要能给医生每天阅读，也要能沉淀为医生自己的知识资产，未来可外挂给大模型用于垂类问答与临床辅助决策。
#
#     一、必须使用的资料范围
#     你必须优先并显式整合以下来源来挖掘选题：
#     A. 我上传的医生资料（若有）包括但不限于：
#     - 决策树/思维导图
#     - 医案
#     - 书籍
#     - 证据卡片示例
#     - 研究资料/指南整理稿/症状评分表/胃镜病理评分表
#     我已将医生资料筛选出精华片段，如下：
#     {materials}
#
#     B. 联网搜索
#     你必须检索最新公开资料寻找选题灵感，尤其包括：
#     - 最新国内外指南
#     - 专家共识
#     - 系统综述
#     - Meta分析
#     - RCT
#     - 高质量观察性研究
#     - 真实世界研究
#     - 疾病管理规范
#     - 近3-5年高质量综述
#
#     C. 学术搜索
#     优先检索：
#     - PubMed
#     - Google Scholar
#     - Cochrane
#     - 国内核心数据库可公开获得部分
#     - 指南发布机构官网
#     - 学协会官网
#
#     二、你对上传资料的使用原则（挖掘选题的依据）
#     1. 决策树 / 思维导图的作用：把它视为“医生显性化诊疗逻辑”。提取风险因素分层、症状识别路径、辅助检查路径等，这决定“临床推理顺序”。
#     2. 医案与书籍的作用：把它们视为“医生隐性经验的文本化载体”。提取核心病机、理法方药对应关系、误治/漏治等，这决定“为什么这样治”。
#     3. 指南与文献的作用：把它们视为“校准器”和“证据增强器”。提取推荐意见、证据等级、与医生经验一致和不一致之处。
#     4. 既有证据卡片样例的作用：借用其成品感、医生阅读友好度、栏目意识。
#
#     三、选题机制：先选题，再写卡片
#     在正式写证据卡片前，你必须先做“选题推演”。
#     每次选题时，先在我上传资料、联网结果和学术搜索结果中提取选题池，然后按照以下九类角度生成候选题目：
#
#     1. 决策冲突型:围绕医生最容易犹豫的点来选题。例如：什么时候以清热化湿为先，什么时候先健脾和胃；某症状背后究竟更偏气滞、湿热、阴虚还是瘀阻；同类表现如何区分不同治法路径。
#     2. 鉴别诊断型:围绕“看起来像，但处理不同”的问题来选题。重点写：相似主诉、关键分叉点、舌脉/胃镜/病理/病程/诱因的区别、误判的临床后果。
#     3. 病机拆解型:围绕一个复杂病机或关键病机来选题。重点写：核心病机是什么、当前主要病机是什么、为什么不能只盯一个证型、病机转化后治法如何变化。
#     4. 症状切入口型:围绕临床高频、但背后病机不单一的症状来选题。例如：烧心、胃脘胀满、胃脘刺痛、嗳气、咽堵、纳呆、背沉背痛。但必须写出“同一症状，不同处理路径”。
#     5. 检查结果驱动型:围绕胃镜、病理、幽门螺杆菌、胆汁反流、萎缩、肠化、异型增生等结果来选题。重点写：某检查发现如何改变辨证权重、某病理结果如何改变随访和干预优先级、某结果出现后治法是否应升级或调整。
#     6. 用药决策型:围绕“什么情况下加什么、减什么、避什么”来选题。重点写：加减触发条件、药物配伍逻辑、与基础方的关系、剂量侧重、安全边界。不能只罗列药物。
#     7. 预后与随访型:围绕“什么时候需要更密切随访、什么时候可观察、什么指标提示风险上升”来选题。重点写：病程风险、复查触发条件、干预有效性的判定指标。
#     8. 生活方式干预型:围绕饮食、情志、作息、分餐、戒烟、低盐、劳逸等内容来选题。必须把它做成“能影响决策质量的内容”，而不是泛泛养生建议。
#     9. 经验-证据碰撞型:围绕医生经验与现代循证之间一致的地方、不一致的地方、互补的地方来选题。这类题目最容易形成差异化内容和讨论价值。
#
#     四、选题去同质化规则
#     你必须把“避免千篇一律”当作硬约束。每个专区内连续生成多张卡片时，禁止出现以下情况：
#     - 连续多张都用同一题目句式
#     - 连续多张都以“某某证治疗方案”命名
#     - 连续多张都只是在症状、证型、方药上做替换
#     - 连续多张都采用同一种切入角度
#     - 连续多张的核心段落结构几乎一致
#     - 连续多张都只讲治疗，不讲鉴别与边界
#     - 连续多张都没有明确的证据指标
#
#     你必须主动拉开差异，至少从以下维度错开：题目句式、切入角度、问题层级、病机层级、检查层级、决策层级、证据层级、适用人群层级、写作重心。
#     题目允许的风格应该是多样的，例如：决策判断式、鉴别分析式、证据比较式、病机拆解式、治疗时机式、误区纠偏式、随访管理式、药物加减式、检查结果解读式。但无论题目风格如何变化，都必须最终落到可执行的临床决策上。
#
#     五、最终输出要求（仅输出选题清单）
#     每次先给出一组候选题目（建议5-8个），并为每个题目标注：
#     - 候选题目
#     - 选题角度
#     - 临床决策价值
#     - 差异化理由
#     - 主要资料来源
#     - 是否适合做成高阅读量卡片
#     - 是否适合做成高决策价值卡片
#     请把自己当成一个长期运营“医生学术内容系统”的总编辑，而不是一次性文案写手。开始你的选题推演！
#     """
#
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     output_dir = f"evidence_cards_{timestamp}"
#     os.makedirs(output_dir, exist_ok=True)
#
#     # ── 第一轮：只生成选题清单 ──────────────────────────────────────────
#     topic_request = (
#             user_prompt
#             + "\n\n【本轮任务】输出5个证据卡片候选选题清单，绝对禁止出现任何中文或英文标点符号。"
#             + "\n\n【格式要求】每个选题的标题行必须严格按照以下格式输出，方便程序提取：\n"
#             + '请严格输出一个标准的JSON数组格式，单占一行，**必须使用双引号**，格式如下：["标题1", "标题2", "标题3", "标题4", "标题5"]。'
#     )
#
#     messages = [{"role": "user", "content": topic_request}]
#
#     print()
#     print("正在生成5个候选选题...")
#
#     topics_raw = requestQwenMultiTurn(system_prompt, messages, False, False)
#     topics = extract_json_array_from_text(topics_raw)
#
#
#     if not topics:
#         return JSONResponse(content={"code": 500, "msg": "AI 未能生成有效的选题列表，请重试"})
#
#     print()
#     print(f"候选选题如下：{topics}")
#
#
#     global_matcher = QwenEmbeddingMatcher()
#     global_pipeline = AgenticPipeline()
#
#     matching_zone(topics, batch_id, global_matcher, global_pipeline, materials)
#
#     return JSONResponse(content={
#         "code": 200,
#         "msg": f"文件卡片生成流程已完成，共处理 {len(topics)} 个选题",
#         "batch_id": batch_id,
#         "topics": topics,
#     })



from filelock import FileLock
# ==========================================
# 模块 1：基于文件的状态管理 (State Management)
# ==========================================
STATE_DIR = "task_states"
os.makedirs(STATE_DIR, exist_ok=True)

def get_state_filepath(task_id: str) -> str:
    return os.path.join(STATE_DIR, f"{task_id}.json")


def init_task_state(task_id: str):
    state = {
        "task_id": task_id,
        "status": "running",  # running, finished, failed
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": None,
        "history_topics": [],  # 核心记忆库：防重复
        "stats": {
            "total_clusters": 0,
            "current_cluster": 0,
            "empty_rounds": 0
        },
        "cursor": {
            "total_processed": 0  # 游标：用于全局挖掘的断点续传快进
        },
        "msg": "后台挖掘任务已启动..."
    }
    _save_state(task_id, state)
    return state


def read_task_state(task_id: str) -> dict:
    filepath = get_state_filepath(task_id)
    lock_path = f"{filepath}.lock"
    if not os.path.exists(filepath):
        return None
    # 读取时也加锁，防止读到写了一半的脏数据
    with FileLock(lock_path, timeout=10):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
# def read_task_state(task_id: str) -> dict:
#     filepath = get_state_filepath(task_id)
#     if not os.path.exists(filepath):
#         return None
#     with open(filepath, "r", encoding="utf-8") as f:
#         return json.load(f)


def _save_state(task_id: str, state_dict: dict):
    filepath = get_state_filepath(task_id)
    lock_path = f"{filepath}.lock"
    # 获取文件锁，防止多线程同时写入导致 JSON 损坏
    with FileLock(lock_path, timeout=10):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, ensure_ascii=False, indent=2)
# def _save_state(task_id: str, state_dict: dict):
#     filepath = get_state_filepath(task_id)
#     with open(filepath, "w", encoding="utf-8") as f:
#         json.dump(state_dict, f, ensure_ascii=False, indent=2)


def update_task_state(task_id: str, updates: dict):
    filepath = get_state_filepath(task_id)
    lock_path = f"{filepath}.lock"
    if not os.path.exists(filepath):
        return

    with FileLock(lock_path, timeout=10):
        # 1. 读出最新状态
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)

        # 2. 更新内存数据
        for k, v in updates.items():
            if isinstance(v, dict) and k in state and isinstance(state[k], dict):
                state[k].update(v)
            elif k == "history_topics" and isinstance(v, list):
                # 针对 history_topics 做特殊处理：多线程下采用合并增量，而非直接覆盖
                existing_topics = set(state.get("history_topics", []))
                existing_topics.update(v)
                state["history_topics"] = list(existing_topics)
            else:
                state[k] = v

        # 3. 写回文件
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

# def update_task_state(task_id: str, updates: dict):
#     state = read_task_state(task_id)
#     if not state:
#         return
#     for k, v in updates.items():
#         if isinstance(v, dict) and k in state and isinstance(state[k], dict):
#             state[k].update(v)
#         else:
#             state[k] = v
#     _save_state(task_id, state)


# ==========================================
# 模块 2：FastAPI 接口层
# ==========================================
@app.get("/api/v1/file_create_card_async")
def file_create_card_async(
        background_tasks: BackgroundTasks,
        batch_id: Optional[str] = None,
        task_id: Optional[str] = Query(None, description="用于断点续传的旧任务ID"),
        force: bool = Query(False, description="是否强制接管(忽略running状态强制执行)"),  # 🌟 新增参数
        allowed_topic_ids: Optional[List[int]] = Query(None, description="指定只匹配的专区ID列表，不传则全库匹配"),
        max_workers: Optional[int] = Query(None, description="自定义最大并发线程数") # 🌟 新增前端传参
):
    # ==========================================
    # 🌟 新增：动态判断并发数逻辑
    # ==========================================
    if max_workers is not None:
        final_workers = max_workers
    elif allowed_topic_ids is not None:
        final_workers = 3
    elif batch_id is not None:
        final_workers = 1
    else:
        final_workers = 1  # 兜底默认值

    """前端触发接口：兼容单文件与全库挖掘"""
    # ==========================================
    # 分支 1：如果是断点续传（传了 task_id）
    # ==========================================
    if task_id:
        existing_state = read_task_state(task_id)
        if not existing_state:
            return JSONResponse(content={"code": 404, "msg": "未找到指定的历史任务文件，无法续传"})

        # 修改这里：如果状态是 running，但用户传了 force=true，就强制接管
        if existing_state.get("status") == "running" and not force:
            return JSONResponse(
                content={"code": 400, "msg": "该挖掘任务正在后台运行中。如果确定是死任务，请传递 &force=true 强制续传"})

        update_task_state(task_id, {
            "status": "running",
            "msg": "后台断点续传任务已恢复..."
        })

        background_tasks.add_task(continuous_mining_worker, task_id, None, allowed_topic_ids, final_workers)
        return JSONResponse(content={
            "code": 200,
            "msg": "后台断点续传任务已成功启动",
            "task_id": task_id
        })

    if batch_id:
        task_id = f"batch_{batch_id}"
    else:
        # 没传 batch_id，开启全库漫游任务
        task_id = f"global_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"

    existing_state = read_task_state(task_id)
    if existing_state and existing_state.get("status") == "running":
        return JSONResponse(content={"code": 400, "msg": "该挖掘任务正在后台运行中，请勿重复提交"})

    init_task_state(task_id)

    # 丢入后台执行
    background_tasks.add_task(continuous_mining_worker, task_id, batch_id, allowed_topic_ids, final_workers)

    return JSONResponse(content={
        "code": 200,
        "msg": "后台自动挖掘任务已启动，请通过 task_id 轮询进度",
        "task_id": task_id,
        "batch_id": batch_id
    })


@app.get("/api/v1/task_status")
def get_task_status(task_id: str):
    """前端轮询接口"""
    state = read_task_state(task_id)
    if not state:
        return JSONResponse(content={"code": 404, "msg": "未找到该任务进度"})
    return JSONResponse(content={"code": 200, "data": state})


# ==========================================
# 模块 3：语义级防重复机制
# ==========================================
def filter_duplicate_topics(new_topics: list, history_topics: list, global_matcher) -> list:
    """
    语义级防重复机制：将新生成的题目与历史题目计算余弦相似度
    注意：这里保留 global_matcher 参数是为了不改变你外部的调用传参，但实际我们使用 vector_4b 来获取向量
    """
    if not history_topics:
        return new_topics

    valid_topics = []
    try:
        # 1. 批量获取历史题目的向量（因为 vector_4b 通常是单条获取，所以我们用循环）
        history_embeddings_list = []
        valid_history_topics = []  # 用于同步记录真正拿到向量的历史题目

        for ht in history_topics:
            vec = vector_4b(ht)
            # 防止 API 偶尔失败返回空
            if vec:
                history_embeddings_list.append(vec)
                valid_history_topics.append(ht)

        # 如果历史记录获取向量全都失败了，直接放行
        if not history_embeddings_list:
            return new_topics

        history_embeddings = np.array(history_embeddings_list)

        # 2. 遍历新题目进行比对
        for topic in new_topics:
            vec = vector_4b(topic)
            if not vec:
                # 如果这个新题目获取向量失败了，为安全起见降级为字面去重
                if topic not in valid_history_topics:
                    valid_topics.append(topic)
                continue

            # 转成 sklearn 需要的 2D 数组格式 (1, 维度)
            topic_vec = np.array([vec])

            # 计算当前新题目与所有历史题目的相似度
            similarities = cosine_similarity(topic_vec, history_embeddings)[0]

            # 设定阈值 0.82 (可以根据你的模型实际表现微调)
            if np.max(similarities) > 0.82:
                print(f"⚠️ 触发防重叠机制: [{topic}] 与历史相似度最高达 {np.max(similarities):.2f}，已剔除。")
                continue

            # 验证通过，加入有效列表
            valid_topics.append(topic)

            # 🌟 关键：将当前刚通过的题目向量“动态”加入历史库中！
            # 这样如果大模型在同一个批次里生成了两个意思一模一样的新题目，第二个也会被拦截
            history_embeddings = np.vstack((history_embeddings, topic_vec))
            valid_history_topics.append(topic)

    except Exception as e:
        print(f"向量去重过程发生异常，自动降级为字面严格去重: {e}")
        # 降级处理：只要字面不完全一样就放行
        valid_topics = [t for t in new_topics if t not in history_topics]

    return valid_topics
# def filter_duplicate_topics(new_topics: list, history_topics: list, global_matcher) -> list:
#     if not history_topics:
#         return new_topics
#     valid_topics = []
#     try:
#         # 这里请确保 可以在当前作用域访问
#         history_embeddings = global_matcher.get_embeddings(history_topics)
#         for topic in new_topics:
#             topic_vec = global_matcher.get_embeddings([topic])[0]
#             similarities = cosine_similarity([topic_vec], history_embeddings)[0]
#             if np.max(similarities) > 0.82:
#                 print(f"⚠️ 触发防重叠机制: [{topic}] 与历史相似度过高，已剔除。")
#                 continue
#             valid_topics.append(topic)
#             history_embeddings = np.vstack((history_embeddings, topic_vec))
#             history_topics.append(topic)
#     except Exception as e:
#         print(f"向量去重失败，降级为字面去重: {e}")
#         valid_topics = [t for t in new_topics if t not in history_topics]
#     return valid_topics


# ==========================================
# 模块 4：核心底层共用逻辑 -> 对一批数据进行“榨汁”
# ==========================================
def _mine_data_batch(task_id: str, target_batch_id: str, embeddings: np.ndarray, texts: list, metadata: list,
                     total_processed: int, allowed_topic_ids: Optional[List[int]] = None):
    """不论是局部 5000 条，还是全局的某 10000 条，都调用这个函数进行聚类与大模型榨取"""
    num_chunks = len(embeddings)
    if num_chunks == 0:
        return

    N_CLUSTERS = max(1, num_chunks // 5)
    update_task_state(task_id, {
        "msg": f"[进度: 库内已阅 {total_processed} 条] 本批次划分为 {N_CLUSTERS} 个矿区准备挖掘。",
        "stats": {"total_clusters": N_CLUSTERS}
    })

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init="auto")
    kmeans.fit(embeddings)

    clusters_pool = {i: [] for i in range(N_CLUSTERS)}
    for idx, label in enumerate(kmeans.labels_):
        clusters_pool[label].append({
            "text": texts[idx],
            "source": metadata[idx],
        })

    # global_matcher = QwenEmbeddingMatcher()
    # global_pipeline = AgenticPipeline()
    global global_matcher, global_pipeline

    for cluster_id, items in clusters_pool.items():
        state = read_task_state(task_id)
        if state["status"] != "running":
            return  # 被外部强行终止

        history_topics = state.get("history_topics", [])
        empty_rounds = state["stats"]["empty_rounds"]

        if empty_rounds >= 3:
            update_task_state(task_id, {"msg": "当前批次数据大模型灵感枯竭，跳跃至下一批新数据！"})
            update_task_state(task_id, {"stats": {"empty_rounds": 0}})
            break  # 放弃当前批次，直接跳出函数

        update_task_state(task_id, {
            "stats": {"current_cluster": cluster_id + 1},
            "msg": f"[进度: 库内已阅 {total_processed} 条] 正在挖掘第 {cluster_id + 1}/{N_CLUSTERS} 矿区..."
        })

        current_materials_str = ""
        for rank, item in enumerate(items[:20]):
            current_materials_str += f"片段{rank + 1} [{item['source']}]: {item['text']}\n"

        history_str = "\n".join(history_topics) if history_topics else "无"

        system_prompt = '你现在是“医学证据卡片选题总编 + 临床知识编辑 + 循证医学研究员 + 医生知识资产工程师”。'
        # 保留您原有的业务要求，但在最前面加上极其严厉的“黑名单”规则
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
        6. 请严格输出一个标准的JSON数组格式，单占一行，必须使用双引号，格式如下：["标题1", "标题2", "标题3"]。
        如果实在没有新角度，输出 []。
        """

        # 请求大模型
        messages = [{"role": "user", "content": dynamic_user_prompt}]
        topics_raw = requestQwenMultiTurn(system_prompt, messages, False, False)
        new_topics = extract_json_array_from_text(topics_raw)

        if not new_topics:
            update_task_state(task_id, {"stats": {"empty_rounds": empty_rounds + 1}})
            time.sleep(3)
            continue

        valid_new_topics = filter_duplicate_topics(new_topics, history_topics.copy(), global_matcher)

        if not valid_new_topics:
            update_task_state(task_id, {"stats": {"empty_rounds": empty_rounds + 1}})
            continue

        # 下发生成卡片任务
        pass_batch_id = target_batch_id if target_batch_id else "GLOBAL_ALL"
        matching_zone(valid_new_topics, pass_batch_id, global_matcher, global_pipeline, current_materials_str, allowed_topic_ids)

        history_topics.extend(valid_new_topics)
        update_task_state(task_id, {
            "history_topics": history_topics,
            "stats": {"empty_rounds": 0}
        })
        time.sleep(5)

    # ==========================================


# 模块 5：总调度台 (Worker Main)
# ==========================================
def continuous_mining_worker(task_id: str, target_batch_id: Optional[str], allowed_topic_ids: Optional[List[int]] = None, max_workers: int = 1):
    # 【多线程配置】
    MAX_WORKERS = max_workers

    try:

        # ========================================================
        # 🌟 新增：前置校验 allowed_topic_ids (防止无效扫描)
        # ========================================================
        if allowed_topic_ids is not None:
            # 1. 如果前端传了一个空列表 []
            if len(allowed_topic_ids) == 0:
                update_task_state(task_id, {
                    "status": "failed",
                    "msg": "❌ 启动失败：传入的专区限制列表为空，任务已终止。"
                })
                print(f"🛑 提前终止：任务 [{task_id}] 传入的 allowed_topic_ids 为空。")
                return

            # 2. 校验传入的 ID 是否在我们的全局内存库中
            valid_ids = [tid for tid in allowed_topic_ids if tid in global_matcher.topic_ids]

            # 如果没有一个是有效的，直接掐断线程！
            if not valid_ids:
                update_task_state(task_id, {
                    "status": "failed",  # 或者标为 finished
                    "msg": f"❌ 启动失败：您指定的专区 ID {allowed_topic_ids} 在数据库中不存在或已失效！"
                })
                print(f"🛑 提前终止：任务 [{task_id}] 指定的专区 IDs {allowed_topic_ids} 全部无效，停止扫描以释放资源。")
                return

            # 3. 纠正数据：过滤掉其中错的，只保留对的，防止后续报错
            allowed_topic_ids = valid_ids
            print(f"✅ 专区 ID 校验通过，最终生效的匹配专区为: {allowed_topic_ids}")

        # ========================================================

        state = read_task_state(task_id)
        total_processed = state.get("cursor", {}).get("total_processed", 0)

        # ========================================================
        # 1. 动态组装迭代器参数（巧妙整合局部和全局逻辑）
        # ========================================================
        if target_batch_id:
            print(f"🚀 开始为 Batch [{target_batch_id}] 执行局部并发挖掘...")
            update_task_state(task_id, {"msg": f"正在建立 Milvus 局部数据迭代器通道..."})

            query_kwargs = {
                "collection_name": MILVUS_COLLECTION_MAIN,
                "filter": f'batch_id == "{target_batch_id}"', # 限定只查这个批次
                "output_fields": ["vector", "text", "batch_id"],
                "batch_size": 200
            }
        else:
            print(f"🌍 开始为 Task [{task_id}] 执行全库百万数据并发挖掘...")
            update_task_state(task_id, {"msg": f"正在建立 Milvus 全库百万级数据迭代器通道..."})

            query_kwargs = {
                "collection_name": MILVUS_REFINEDDATA,
                "output_fields": ["vector", "text"],
                "batch_size": 200
            }

        # 统一生成迭代器
        iterator = milvus_client.query_iterator(**query_kwargs)

        # ========================================================
        # 2. 断点续传快进机制
        # ========================================================
        if total_processed > 0:
            BATCH_SIZE = 200  # 必须与上面的 query_kwargs 里的 batch_size 保持完全一致
            batches_to_skip = total_processed // BATCH_SIZE
            if batches_to_skip > 0:
                update_task_state(task_id, {
                    "msg": f"检测到历史进度 {total_processed} 条，快进跳过前 {batches_to_skip} 个批次..."})
                for _ in range(batches_to_skip):
                    iterator.next()  # 空转迭代器，只跳过不处理

        # ========================================================
        # 3. 统一进入多线程生产者/消费者模式
        # ========================================================
        # 🚀 启动线程池
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = set()

            while True:
                # 【🌟 新增监听】检测是否被用户手动打断
                current_state = read_task_state(task_id)
                if current_state and current_state.get("status") != "running":
                    print(f"🛑 检测到任务 [{task_id}] 已被中断，主线程停止分发！")
                    break

                # 【背压控制】防止消费者慢、生产者快导致内存爆炸
                if len(futures) >= MAX_WORKERS * 2:
                    completed, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    # 检查完成的任务是否有抛出异常
                    for future in completed:
                        try:
                            future.result()
                        except Exception as e:
                            print(f"⚠️ 某批次处理发生异常: {e}")

                # 1. 抽水
                results = iterator.next()
                if not results or len(results) == 0:
                    iterator.close()
                    break  # 彻底穷尽（如果是单批次，没查到数据也会直接跳出）

                update_task_state(task_id, {"msg": f"成功抽取新一批 {len(results)} 条数据，清洗分发中..."})

                # 2. 清洗
                embeddings_list, texts, metadata = [], [], []
                for item in results:
                    text_content = item.get("text", "").strip()
                    if is_garbage_text(text_content) or len(text_content) < 50:
                        continue
                    embeddings_list.append(item["vector"])
                    texts.append(text_content)
                    metadata.append(f"来源文件:{item.get('batch_id', '全局')}")

                current_batch_count = len(results)
                total_processed += current_batch_count
                update_task_state(task_id, {"cursor": {"total_processed": total_processed}})

                if not embeddings_list:
                    continue

                # 3. 提交给线程池异步执行
                print(f"[调度] 提交新批次 (有效大小: {len(embeddings_list)}) 进入线程池，当前积压任务数: {len(futures)}")
                future = executor.submit(
                    _mine_data_batch,
                    task_id,
                    target_batch_id,  # 传入真实的 target_batch_id
                    np.array(embeddings_list),
                    texts,
                    metadata,
                    total_processed,
                    allowed_topic_ids
                )
                futures.add(future)

            # 迭代器抽干后，在此阻塞，等待线程池中剩余的所有任务完成
            print("[调度] 库内数据已全部抽出，等待剩余线程收尾...")
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"⚠️ 结尾批次处理异常: {e}")

        # --------------------------------------------------------
        # 任务完美结束 (判断一下是不是被手动停止的)
        # --------------------------------------------------------
        final_state = read_task_state(task_id)
        if final_state and final_state.get("status") == "running":
            final_count = len(final_state.get('history_topics', []))
            update_task_state(task_id, {
                "status": "finished",
                "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "msg": f"🎉 挖掘圆满完成！共产出 {final_count} 个不重复的高价值选题！"
            })
            print(f"✅ 任务 [{task_id}] 后台挖掘正常结束！")

    except Exception as e:
        print(f"❌ 后台挖掘异常: {e}")
        update_task_state(task_id, {
            "status": "failed",
            "msg": f"发生错误终止: {str(e)}",
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })


@app.get("/api/v1/stop_task")
def stop_task(
        batch_id: Optional[str] = Query(None, description="要停止的 batch_id"),
        task_id: Optional[str] = Query(None, description="要停止的 task_id")
):
    """手动停止正在运行的挖掘任务"""
    # 统一转换为 task_id
    if batch_id:
        target_task_id = f"batch_{batch_id}"
    elif task_id:
        target_task_id = task_id
    else:
        return JSONResponse(content={"code": 400, "msg": "必须提供 batch_id 或 task_id"})

    existing_state = read_task_state(target_task_id)
    if not existing_state:
        return JSONResponse(content={"code": 404, "msg": "未找到该任务的运行状态记录"})

    if existing_state.get("status") != "running":
        return JSONResponse(content={"code": 200, "msg": "该任务本身已不在运行状态"})

    # 更新状态为 interrupted，打断后台循环
    update_task_state(target_task_id, {
        "status": "interrupted",
        "msg": "任务已被用户手动终止！",
        "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    return JSONResponse(content={"code": 200, "msg": f"停止指令已发送，任务 {target_task_id} 即将终止。"})







    # prompt  =  """
    # 根据 胃炎 生成10个不同证据专区标题。
    # 以下是对证据专区的理解
    # 1.证据专区
    # “证据专区”是整个医学知识库的分类导航和核心锚点。它不能是教科书式的大杂烩，而必须是精准切中临床痛点的“微靶点”。
    # A.聚焦与精准（双实体原则）：
    #   拒绝泛泛而谈： 不能用“心血管专区”、“肿瘤专区”这种大词，这会让医生觉得毫无新意且失去阅读方向。
    #   多维度交叉： 专区命名必须包含至少两个医学实体（如：疾病+特定靶点药物、疾病+特定并发症、疾病+新兴技术）。例如：“GLP-1受体激动剂在心血管高危糖尿病患者中的应用专区”、“晚期非小细胞肺癌免疫治疗耐药管理专区”。这样的命名，医生一看就知道里面解决的是具体的临床细分问题。
    # B.紧跟前沿与热点（RCT驱动与AI融入）：
    #   专区的设立要基于近三年的RCT研究趋势，这意味着专区代表了当下医学界最关心、最受争议或最具突破性的领域。
    #   同时，需要敏锐捕捉**“AI+医疗”**的趋势，建立如“AI影像在微小肺结节鉴别诊断中的应用专区”或“大模型在罕见病辅助诊断中的探索”等前沿专区。
    # C.临床实用导向：
    #   专区设立的终极目的是“帮医生解决日常诊疗中的困惑”，如药物选择、剂量调整、不良反应处理、特殊人群用药等。
    #   """


    # """前端轮询接口"""
    # state = read_task_state(task_id)
    # if not state:
    #     return JSONResponse(content={"code": 404, "msg": "未找到该任务进度"})
    # return JSONResponse(content={"code": 200, "data": state})


# ================= 全局 MongoDB 客户端初始化 =================
# Motor 是异步安全的，建议在全局初始化，或者在 startup_event 中初始化
# mongo_client = AsyncIOMotorClient(MONGO_URI)
# db = mongo_client[MONGO_DATABASE]
# collection = db[MONGO_COLLECTION]

# STATE_DIR = "task_states"
# CURSOR_FILE = os.path.join(STATE_DIR, "mongo_sync_cursor.json")

#
# def get_last_ts() -> int:
#     """获取上次处理的时间戳 (毫秒级)"""
#     if os.path.exists(CURSOR_FILE):
#         with open(CURSOR_FILE, "r") as f:
#             data = json.load(f)
#             return data.get("last_ts", 0)
#     return 0
#
#
# def save_last_ts(ts: int):
#     """保存最新的时间戳"""
#     with open(CURSOR_FILE, "w") as f:
#         json.dump({"last_ts": ts}, f)
#
#
# # ================= 修改后的核心处理逻辑 (带有返回值) =================
# async def process_mongo_questions_batch():
#     """手动抓取增量数据并送入挖掘管线"""
#     try:
#         last_ts = get_last_ts()
#
#         # 增量拉取：大于上次时间戳，按时间正序，限制单次 200 条
#         cursor = collection.find({"ts": {"$gt": last_ts}}).sort("ts", 1).limit(200)
#         questions = await cursor.to_list(length=200)
#
#         if not questions:
#             return {"status": "success", "count": 0, "msg": "当前没有新的提问数据需要处理"}
#
#         max_ts_in_batch = max([q["ts"] for q in questions])
#         print(f"[API触发] 抓取到 {len(questions)} 条新用户提问，准备进行大模型提纯...")
#
#         # 1. 提取所有问题内容
#         # 注意：这里请将 'question_content' 替换为你 MongoDB 里面实际存提问文本的字段名
#         raw_questions_text = "\n".join([f"- {q.get('question_content', '')}" for q in questions])
#
#         # ===== 调用大模型提纯的逻辑 (复用你现有的代码) =====
#         # res_raw = requestQwen(system_prompt, user_prompt)
#         # new_topics = extract_json_array_from_text(res_raw)
#
#         # 这里为了演示，假设提取出了结果
#         new_topics = []  # 你需要用真实提纯结果替换它
#
#         # 就算没有提取出有效的卡片，也要更新 TS，否则下次还会查到这批废数据陷入死循环！
#         if not new_topics:
#             save_last_ts(max_ts_in_batch)
#             return {"status": "success", "count": len(questions), "msg": "处理了数据但大模型未提取出有效卡片标题",
#                     "new_ts": max_ts_in_batch}
#
#         # ===== 对接现有的 matching_zone =====
#         # global_matcher = QwenEmbeddingMatcher()
#         # global_pipeline = AgenticPipeline()
#         # batch_id = f"mongo_{max_ts_in_batch}"
#         # matching_zone(new_topics, batch_id, global_matcher, global_pipeline, raw_questions_text)
#
#         # 2. 跑通全流程后，更新游标
#         save_last_ts(max_ts_in_batch)
#
#         return {
#             "status": "success",
#             "processed_count": len(questions),
#             "generated_topics_count": len(new_topics),
#             "new_topics": new_topics,
#             "new_ts": max_ts_in_batch
#         }
#
#     except Exception as e:
#         print(f"❌ 处理 MongoDB 增量数据时发生异常: {str(e)}")
#         raise e
#
# # ================= 新增：FastAPI 触发接口 =================
# @app.post("/api/v1/trigger_mongo_mining")
# async def trigger_mongo_mining(
#     background_tasks: BackgroundTasks,
#     run_in_background: bool = Query(False, description="是否放入后台静默执行(避免接口超时)")
# ):
#     """
#     手动触发：从 MongoDB 中拉取最新用户提问并生成证据卡片
#     """
#     if run_in_background:
#         # 扔进后台队列，接口立刻返回 200
#         background_tasks.add_task(process_mongo_questions_batch)
#         return JSONResponse(content={
#             "code": 200,
#             "msg": "已成功提交到后台执行增量挖掘任务",
#             "status": "processing"
#         })
#     else:
#         # 阻塞等待大模型执行完毕，返回完整明细
#         try:
#             result = await process_mongo_questions_batch()
#             return JSONResponse(content={
#                 "code": 200,
#                 "msg": "挖掘任务执行完成",
#                 "data": result
#             })
#         except Exception as e:
#             return JSONResponse(content={
#                 "code": 500,
#                 "msg": f"挖掘执行失败: {str(e)}"
#             })


# ================= 在 async_api_server.py 顶部增加引用 =================
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
from utils import get_distributed_filenames, record_distribution

# 确保你在 config.py 中有这两个变量，或者根据你实际的变量名修改
# from config import WECHAT_APP_ID, WECHAT_APP_SECRET

CARD_MD_DIR = os.path.join(os.getcwd(), "card_md")


# ================= 添加每天定时触发的 API 接口 =================
@app.post("/api/v1/daily_wechat_push")
async def daily_wechat_push(background_tasks: BackgroundTasks):
    """
    每天定时分发接口：扫描 card_md -> 过滤已发 -> 取前 8 篇 -> 后台调用 MCP 推送
    """
    if not os.path.exists(CARD_MD_DIR):
        return JSONResponse(content={"code": 404, "msg": "card_md 目录不存在"})

    # 1. 获取曾经成功发送过的内容
    sent_list = get_distributed_filenames('wechat')

    # 2. 扫描目录获取所有 markdown，并过滤掉已经发送的
    all_files = [f for f in os.listdir(CARD_MD_DIR) if f.endswith('.md')]
    pending_files = [f for f in all_files if f not in sent_list]

    # 3. 严格截取前 8 篇（微信每日订阅号限制或测试限制）
    target_files = pending_files[:8]

    if not target_files:
        return JSONResponse(content={"code": 200, "msg": "今日无新文章可推送，所有卡片均已分发。"})

    # 4. 把耗时的 MCP 启动和发布过程丢到后台执行
    background_tasks.add_task(process_wechat_distribution_via_mcp, target_files)

    return JSONResponse(content={
        "code": 200,
        "msg": f"已启动后台分发任务，准备推送 {len(target_files)} 篇文章",
        "files": target_files
    })


# ================= 基于你提供的示例改写的后台执行函数 =================
async def process_wechat_distribution_via_mcp(filenames: list):
    """
    后台任务：建立一次 MCP 会话，批量发布传入的文件列表
    """
    server_script_path = os.path.abspath("./wechat-publisher-mcp/src/server.js")

    server_params = StdioServerParameters(
        command="node",
        args=[server_script_path],
        env=None
    )

    print(f"\n[后台分发] 正在启动并连接到 MCP Server，共有 {len(filenames)} 篇文章待发...")

    try:
        # 建立全局唯一的 stdio 连接，避免每发一篇文章重启一次 Node 进程
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print("✅ [后台分发] MCP 会话初始化成功！开始执行发布任务...")

                for idx, fname in enumerate(filenames):
                    file_path = os.path.join(CARD_MD_DIR, fname)
                    title = fname.replace('.md', '')

                    try:
                        # 读取本地 MD 文件内容
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # 构建 MCP 工具调用参数
                        arguments = {
                            "title": title,
                            "content": content,
                            "author": "灵犀量子 AI",
                            "appId": WECHAT_APP_ID,  # 请替换为你在 config 里的真实变量
                            "appSecret": WECHAT_APP_SECRET,
                            "previewMode": False,
                            "coverImagePath": ""  # 没有封面留空，MCP内部会自动生成
                        }

                        print(f"\n🚀 正在发布第 {idx + 1}/{len(filenames)} 篇: {title}")

                        # 调用 MCP 工具
                        result = await session.call_tool(
                            "wechat_publish_article",
                            arguments=arguments
                        )

                        # 解析 MCP Server 返回的文本信息
                        res_text = ""
                        for content_item in result.content:
                            if content_item.type == "text":
                                res_text += content_item.text

                        # 判断是否发布成功 (如果你的 server.js 失败会返回 "❌ 发布失败")
                        if "❌" not in res_text and result.isError is not True:
                            # 记录到 jsonl，状态为 success
                            record_distribution('wechat', fname, 'success', res_text)
                            print(f"✅ 【{fname}】 发布成功: {res_text.strip()}")
                        else:
                            # 记录到 jsonl，状态为 failed
                            record_distribution('wechat', fname, 'failed', res_text)
                            print(f"❌ 【{fname}】 业务级失败: {res_text.strip()}")

                    except Exception as e:
                        # 异常捕捉：防止单篇文章报错中断了后面7篇的发送
                        record_distribution('wechat', fname, 'failed', f"Python脚本异常: {str(e)}")
                        print(f"❌ 【{fname}】 发送发生内部异常: {e}")

                print("\n🎉 [后台分发] 批量发布任务执行结束，MCP 连接已安全关闭。")

    except Exception as e:
        print(f"❌ [后台分发] MCP Server 连接或会话建立失败: {e}")



if __name__ == '__main__':
    # 强制 workers=1 保护显存
    # uvicorn.run("async_api_server:app", host="0.0.0.0", port=5811, workers=1)
    uvicorn.run("async_api_server:app", host="0.0.0.0", port=6006, workers=1)







