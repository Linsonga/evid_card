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

import uvicorn
import requests
import pdfplumber
import pymysql
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pymilvus import MilvusClient

from config import DB_CONFIG, MILVUS_MAIN_URI, MILVUS_MAIN_TOKEN
from utils import vector_4b, request_qwen_async

app = FastAPI(title="文件处理回调接口")

MILVUS_COLLECTION_MAIN = 'evidence_card_4B'

# ================= 配置区 =================
BASE_DIR   = os.path.join(os.getcwd(), "api_data")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

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


def detect_pdf_type(pdf_path: str) -> str:
    """
    分析 PDF 文件类型
    返回: 'xmind' | 'image' | 'text'
    """
    try:

        print()
        print(f"文件路劲：{pdf_path}")

        with pdfplumber.open(pdf_path) as pdf:
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
        print(f"检测 PDF 类型失败: {e}")
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


def chunk_text_data(content_list, split_to_length: int = 500):
    """核心分块逻辑：输入解析列表，输出分块后的列表"""
    if not content_list:
        return []

    require_items = [x for x in content_list if x.get("type") in ("text", "title")]
    if not require_items:
        return []

    require_list  = [x.get("text", "") for x in require_items]
    combined_text = "".join(require_list)

    lang_detect = is_chinese_or_english(combined_text)
    if lang_detect == "中文":
        separators = ("。", "！", "!")
    else:
        separators = (". ", ".", "!", "！", "。")

    new_block_list = []

    for i, con in enumerate(require_items):
        t = (con.get("text") or "").strip()
        if lang_detect == "英文" and t:
            t += " "

        current_bbox = con.get("bbox", [])

        if not new_block_list:
            new_block_list.append({
                "text": t,
                "bbox": [current_bbox] if current_bbox else [],
                "end_oldblock_type": con["type"]
            })
            continue

        last           = new_block_list[-1]
        last_text      = last["text"]
        end_with_sep   = last_text.rstrip().endswith(separators)
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

    # 结尾小块并入前一块
    if len(new_block_list) >= 2 and len(new_block_list[-1]["text"]) < 300:
        new_block_list[-2]["text"] += new_block_list[-1]["text"]
        new_block_list[-2]["bbox"].extend(new_block_list[-1]["bbox"])
        new_block_list.pop()

    final_res_list = []
    for index, block in enumerate(new_block_list):
        final_res_list.append({
            "text": block["text"],
            "index": index,
            "length": len(block["text"]),
            "bbox": block["bbox"]
        })

    return final_res_list


def convert_docx_to_pdf_sync(docx_path: str, output_dir: str):
    """调用系统 LibreOffice 将 docx/doc 转换为 PDF"""
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


def process_and_insert_to_milvus(task_id: str, chunked_data: list, batch_id: str = None, type: str = 'pdf', user_id: str = '', file_name: str = ''):
    """
    遍历分块数据，调用 Embedding 接口，批量写入 Milvus。
    batch_id:  同一批次的多个文件共用同一个 batch_id；不传则默认使用 task_id。
    user_id:   文件所属用户 ID。
    file_name: 原始文件名称。
    """
    if batch_id is None:
        batch_id = task_id
    if not chunked_data:
        return {"status": "error", "msg": "没有有效的分块数据可供插入"}

    filtered_count = 0
    try:
        for index, item in enumerate(chunked_data):
            text = item.get("text", "").strip()
            if not text:
                continue
            if type == 'pdf':
                if len(text) > 600:
                    continue
            if is_garbage_text(text):
                filtered_count += 1
                continue

            try:
                embeddings_list = vector_4b(text)
            except Exception as e:
                print(f"[Milvus] 块 {index} 请求 Embedding API 报错: {e}")
                continue

            doc_id    = f"{task_id}_{index}"
            to_insert = [{
                "id":        doc_id,
                "vector":    embeddings_list,
                "batch_id":  batch_id,
                "text":      text,
                "date":      str(time.time()),
                "user_id":   user_id,
                "file_name": file_name,
            }]
            milvus_client.insert(collection_name=MILVUS_COLLECTION_MAIN, data=to_insert)

    except Exception as e:
        return {"status": "error", "msg": f"Milvus 入库失败: {str(e)}"}

    return {"status": "success", "filtered_count": filtered_count}


# ================= 批次后台处理任务 =================
async def background_process_pdf_batch(task_id: str, batch_id: str, file_path: str, user_id: str = '', file_name: str = ''):
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
                [{"type": "text", "text": text_content}], split_to_length=400
            )
            db_result = process_and_insert_to_milvus(
                task_id, chunked_data, batch_id=batch_id, type='txt',
                user_id=user_id, file_name=file_name
            )

        # ===== 分支二：PDF / xmind(PDF导出) 文件 =====
        else:
            pdf_type = detect_pdf_type(file_path)
            print(f"[批次 {batch_id}] 任务 {task_id} 检测到文档类型: {pdf_type}")

            if pdf_type == "xmind":
                # xmind 思维导图：提取坐标节点 -> 大模型还原文本结构 -> 分块入库
                detected_lang = "zh"
                json_data     = extract_xmind_to_text(file_path)
                prompt = f"""
                    这是一个从思维导图（Xmind）导出的节点数据，包含了文本以及它们在页面上的物理坐标。
                    X 是横坐标（越小越靠左），Y 是纵坐标（越小越靠上）。
                    请你扮演一位逻辑分析专家，执行以下任务：
                    隐式还原逻辑：请先在底层根据 X,Y 坐标的分布规律（如从左到右、从上到下的发散关系），理清各个节点之间的"父子、并列"等逻辑关联。
                    提取核心大意：理解这个思维导图到底在表达什么主题，包含了哪些核心分支和细节。
                    请抛弃列表或树状格式，将整个导图表达的意思融会贯通，用一段通顺、连贯的文字总结出来。只要总结内容，逻辑清晰，语言自然流畅。
                    数据如下：
                    {json_data}
                """
                xmind_str    = await request_qwen_async("", prompt)
                chunked_data = chunk_text_data(
                    [{"type": "text", "text": xmind_str}], split_to_length=400
                )
                db_result = process_and_insert_to_milvus(
                    task_id, chunked_data, batch_id=batch_id, type='xmind',
                    user_id=user_id, file_name=file_name
                )


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
                db_result = process_and_insert_to_milvus(task_id, chunked_data, batch_id=batch_id, type='pdf', user_id=user_id, file_name=file_name)

                # print(f"普通文本")
                # # 普通文本 PDF：使用 pdfplumber 逐页提取文本 -> 分块入库
                # data_lis = extract_text_from_pdf_with_plumber(file_path)
                # if data_lis:
                #     combined      = "".join(item.get("text", "") for item in data_lis)
                #     detected_lang = is_chinese_or_english(combined)
                # chunked_data = chunk_text_data(data_lis, split_to_length=400)
                #
                # db_result    = process_and_insert_to_milvus(
                #     task_id, chunked_data, batch_id=batch_id, type='pdf',
                #     user_id=user_id, file_name=file_name
                # )

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
        error_data = {
            "task_id":   task_id,
            "batch_id":  batch_id,
            "status":    "failed",
            "error_msg": type(e).__name__ + ": " + str(e)
        }
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(error_data, f, ensure_ascii=False, indent=2)

    finally:
        # 处理完毕后清理临时文件
        if os.path.exists(file_path):
            os.remove(file_path)


# ================= 接口一：OSS URL 批量提交解析入库 =================
@app.post("/api/v1/upload_batch")
async def create_upload_batch_from_url(
    background_tasks: BackgroundTasks,
    files: List[FileCallbackItem]
):
    """
    接收线上 OSS 文件列表，自动下载后解析、分块并写入向量库。

    支持格式：PDF、xmind（PDF导出格式）、TXT、Word（docx / doc）
    Word 文件会先通过 LibreOffice 转换为 PDF 再处理。

    请求体示例：
    [
      {
        "file_url": "https://image.evimed.com/oss/evidence-card/xxx.pdf",
        "user_id": "5724839605178470186",
        "file_name": "(2006_BSG)肝硬化腹水的管理指南.pdf",
        "id": 9
      }
    ]

    立即返回 batch_id 与每个文件对应的 task_id，后续可通过 task_id 查询进度。
    """
    if not files:
        raise HTTPException(status_code=400, detail="请至少提供一个文件")

    SUPPORTED_EXTS = {'.pdf', '.xmind', '.txt', '.docx', '.doc'}

    # 整批共用同一个 batch_id
    batch_id  = str(uuid.uuid4())
    task_list = []

    for file in files:
        file_url   = file.file_url
        file_name  = file.file_name
        ext        = os.path.splitext(file_name.lower())[-1]

        if ext not in SUPPORTED_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"文件 [{file_name}] 格式不支持，仅允许 PDF、xmind、TXT、Word(docx/doc)"
            )

        task_id       = str(uuid.uuid4())
        download_path = os.path.join(UPLOAD_DIR, f"{task_id}{ext}")
        final_path    = download_path

        # ── 步骤 1：从 OSS 下载文件 ──
        try:
            resp = requests.get(file_url, timeout=60)
            resp.raise_for_status()
            with open(download_path, 'wb') as fp:
                fp.write(resp.content)
            print(f"[下载成功] {file_name}  ->  {download_path}")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"下载文件 [{file_name}] 失败: {str(e)}"
            )

        # ── 步骤 2：Word 文件同步转换为 PDF ──
        if ext in ('.docx', '.doc'):
            final_path = os.path.join(UPLOAD_DIR, f"{task_id}.pdf")
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None, convert_docx_to_pdf_sync, download_path, UPLOAD_DIR
                )
                if os.path.exists(download_path):
                    os.remove(download_path)
            except Exception as e:
                if os.path.exists(download_path):
                    os.remove(download_path)
                raise HTTPException(
                    status_code=500,
                    detail=f"文件 [{file_name}] Word转PDF失败: {str(e)}"
                )

        # ── 步骤 3：初始化任务状态文件 ──
        initial_status = {
            "task_id":   task_id,
            "batch_id":  batch_id,
            "file_id":   file.id,
            "file_name": file_name,
            "user_id":   file.user_id,
            "status":    "processing"
        }
        with open(os.path.join(RESULT_DIR, f"{task_id}.json"), "w", encoding="utf-8") as fp:
            json.dump(initial_status, fp, ensure_ascii=False)

        # ── 步骤 4：加入后台解析队列（解析 + 分块 + 入库，共用 batch_id）──
        background_tasks.add_task(
            background_process_pdf_batch, task_id, batch_id, final_path,
            file.user_id, file_name
        )
        task_list.append({
            "task_id":  task_id,
            "file_id":  file.id,
            "filename": file_name
        })

    return JSONResponse(content={
        "code":     200,
        "msg":      f"批量提交成功，共 {len(task_list)} 个文件已加入解析队列",
        "batch_id": batch_id,
        "tasks":    task_list
    })


# ================= 接口二：文件处理完成回调（原有接口，保持不变） =================
@app.post("/api/v1/analysis_file", status_code=200)
async def update_file_status(files: List[FileCallbackItem]):
    """
    接收文件处理完成的回调，更新数据库状态
    """
    if not files:
        raise HTTPException(status_code=400, detail="请求数据不能为空")

    valid_ids = []
    for file in files:
        try:
            valid_ids.append(file.id)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"文件 ID [{file.id}] 格式错误，无法转换为数字")

    if not valid_ids:
        return {"code": 200, "message": "没有需要更新的数据"}

    try:
        mock_db_execute_update(files)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据库更新失败: {str(e)}")

    return {"code": 200, "message": "success"}


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

        insert_card_data.append((file_id, card_name, user_id))
        update_status_data.append((file_id,))

    conn   = None
    cursor = None
    try:
        conn   = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()

        insert_sql = """
            INSERT INTO evidence_file_card_name (file_id, card_name, user_id)
            VALUES (%s, %s, %s)
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5911, workers=1)
