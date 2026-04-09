"""
批量上传 PDF 文件夹并轮询解析进度的客户端脚本。

用法：
    python batch_client.py --folder /path/to/pdf_folder
    python batch_client.py --folder /path/to/pdf_folder --host http://localhost:5811
    python batch_client.py --folder /path/to/pdf_folder --interval 5 --timeout 600
"""

import os
import sys
import time
import argparse
import requests
from openai import OpenAI
import pdfplumber

# ================= 配置区（可按需修改） =================
DEFAULT_HOST = "http://192.168.20.129:5811"
DEFAULT_POLL_INTERVAL = 10    # 每次轮询间隔（秒）
DEFAULT_TIMEOUT = None       # 最长等待时间（秒），None 表示不设超时，永久等待直到全部完成


def requestQwencontentInterruption(system_prompt, user_prompt):
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-bb48765d7ed540d49639b1bd5e4dd82b",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen3-235b-a22b-instruct-2507",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content

def upload_batch(host: str, folder: str) -> dict:
    """
    扫描 folder 下所有 PDF/DOCX 文件，批量上传到 /api/v1/upload_batch。
    返回服务端响应的 JSON 字典。
    """
    supported_ext = ('.pdf', '.docx')
    file_paths = []
    for root, dirs, files in os.walk(folder):
        dirs.sort()  # 保证遍历顺序稳定
        for f in sorted(files):
            if f.lower().endswith(supported_ext):
                file_paths.append(os.path.join(root, f))

    if not file_paths:
        print(f"[错误] 文件夹 '{folder}' 及其子目录中未找到任何 PDF/DOCX 文件")
        sys.exit(1)

    print(f"[上传] 递归扫描完成，共找到 {len(file_paths)} 个文件，正在上传...")
    for p in file_paths:
        print(f"  {os.path.relpath(p, folder)}")

    # 构造 multipart/form-data，key 固定为 "files"
    file_handles = []
    try:
        for path in file_paths:
            fh = open(path, "rb")
            file_handles.append(("files", (os.path.basename(path), fh, "application/octet-stream")))

        url = f"{host}/api/v1/upload_batch"
        resp = requests.post(url, files=file_handles, timeout=120)
        resp.raise_for_status()
        return resp.json()
    finally:
        for _, (_, fh, _) in file_handles:
            fh.close()


def poll_task(host: str, task_id: str) -> dict:
    """
    查询单个任务的解析结果（不触发入库，因为批次模式已在后台自动入库）。
    返回服务端响应 JSON。
    """
    url = f"{host}/api/v1/get_result/{task_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def wait_for_all_tasks(host: str, tasks: list, interval: int, timeout):
    """
    轮询所有任务，直到全部完成或超时。
    tasks: [{"task_id": "...", "filename": "..."}, ...]
    timeout: 超时秒数，None 表示不限时永久等待。
    """
    pending = {t["task_id"]: t["filename"] for t in tasks}
    results = {}
    start = time.time()

    timeout_desc = f"{timeout}s" if timeout is not None else "不限时"
    print(f"\n[轮询] 开始轮询 {len(pending)} 个任务，间隔 {interval}s，超时 {timeout_desc}\n")

    while pending:
        if timeout is not None and time.time() - start > timeout:
            print(f"\n[超时] 以下任务超过 {timeout}s 仍未完成，停止等待：")
            for tid, fname in pending.items():
                print(f"  - {fname}  (task_id: {tid})")
            break

        time.sleep(interval)

        finished_ids = []
        for task_id, filename in list(pending.items()):
            try:
                data = poll_task(host, task_id)
            except Exception as e:
                print(f"  [查询失败] {filename}: {e}")
                continue

            status = data.get("status")
            if status == "processing":
                print(f"  [处理中] {filename}")
            elif status == "completed":
                chunk_count = data.get("chunk_count", "N/A")
                print(f"  [完成]   {filename}  (分块数: {chunk_count})")
                results[task_id] = data
                finished_ids.append(task_id)
            elif status == "failed":
                err = data.get("error_msg", "未知错误")
                print(f"  [失败]   {filename}  错误: {err}")
                results[task_id] = data
                finished_ids.append(task_id)
            else:
                print(f"  [未知状态] {filename}: {data}")

        for tid in finished_ids:
            pending.pop(tid, None)

        if pending:
            print(f"  --- 还剩 {len(pending)} 个任务未完成，{interval}s 后继续轮询...\n")

    return results


def print_summary(batch_id: str, tasks: list, results: dict):
    total = len(tasks)
    completed = sum(1 for v in results.values() if v.get("status") == "completed")
    failed = sum(1 for v in results.values() if v.get("status") == "failed")
    pending = total - len(results)

    print("\n" + "=" * 50)
    print(f"  批次 ID   : {batch_id}")
    print(f"  总文件数  : {total}")
    print(f"  已完成    : {completed}")
    print(f"  失败      : {failed}")
    print(f"  未完成    : {pending}")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="批量上传 PDF 文件夹并轮询解析进度")

    # 在配置区加一行
    DEFAULT_FOLDER = "/root/Data/GLS/evidence_card/基层医疗"

    parser.add_argument("--folder", default=DEFAULT_FOLDER, help="包含 PDF/DOCX 文件的文件夹路径")
    parser.add_argument("--host",     default=DEFAULT_HOST,          help=f"API 服务地址（默认: {DEFAULT_HOST}）")
    parser.add_argument("--interval", default=DEFAULT_POLL_INTERVAL, type=int, help=f"轮询间隔秒数（默认: {DEFAULT_POLL_INTERVAL}）")
    parser.add_argument("--timeout", default=DEFAULT_TIMEOUT, type=lambda x: int(x) if x is not None else None,
                        help="最长等待秒数，不传或传 0 表示不限时永久等待（默认: 不限时）")
    args = parser.parse_args()

    # 传 0 等同于不限时
    timeout = None if (args.timeout is None or args.timeout <= 0) else args.timeout

    if not os.path.isdir(args.folder):
        print(f"[错误] '{args.folder}' 不是有效目录")
        sys.exit(1)

    # 1. 批量上传
    resp = upload_batch(args.host, args.folder)
    if resp.get("code") != 200:
        print(f"[错误] 上传失败: {resp}")
        sys.exit(1)

    batch_id = resp["batch_id"]
    tasks = resp["tasks"]
    print(f"\n[成功] 批次 ID: {batch_id}")
    for t in tasks:
        print(f"  - {t['filename']}  ->  task_id: {t['task_id']}")

    # 2. 轮询所有任务完成
    results = wait_for_all_tasks(args.host, tasks, args.interval, timeout)

    # 3. 打印汇总
    print_summary(batch_id, tasks, results)





def extract_xmind_hierarchy(pdf_path):
    # 存储所有提取的文本块及其坐标：{'text': '节点文字', 'x0': 左边界, 'top': 上边界, 'level': 级别}
    nodes = []

    with pdfplumber.open(pdf_path) as pdf:
        # 假设导图在一页内
        page = pdf.pages[0]

        # 提取字典格式的文字块 (extract_words 会自动把靠近的字符拼成词)
        # 注意：如果一个节点里有长句子，可能需要调整 x_tolerance 和 y_tolerance
        words = page.extract_words(x_tolerance=3, y_tolerance=3)

        for w in words:
            nodes.append({
                'text': w['text'],
                'x': w['x0'],  # 左侧X坐标
                'y': w['top'],  # 顶部Y坐标
            })

    # 1. 按照 X 坐标对节点进行聚类，划分“层级 (Level)”
    # 考虑到同一列的节点 X 坐标可能有一点微小误差，设置一个容差值 (例如 20 像素)
    x_tolerance = 20
    nodes.sort(key=lambda n: n['x'])  # 先按 X 排序

    levels = []  # 用于存储不同层级的节点列表
    current_level_nodes = []
    current_x = nodes[0]['x']

    for node in nodes:
        if node['x'] - current_x <= x_tolerance:
            current_level_nodes.append(node)
        else:
            levels.append(current_level_nodes)
            current_level_nodes = [node]
            current_x = node['x']
    if current_level_nodes:
        levels.append(current_level_nodes)

    # 2. 为每个层级打上 Level 标签 (0是根节点，1是第一层子节点...)
    for i, level_nodes in enumerate(levels):
        for node in level_nodes:
            node['level'] = i

    # 3. 建立父子关系 (除了根节点，每个节点都要去上一层找爸爸)
    for i in range(1, len(levels)):
        current_level = levels[i]
        parent_level = levels[i - 1]

        for node in current_level:
            # 寻找上一层中，Y坐标与当前节点最接近的节点作为父节点
            closest_parent = min(parent_level, key=lambda p: abs(p['y'] - node['y']))
            node['parent_text'] = closest_parent['text']

    # 4. 打印 Markdown 格式的层级树
    print("--- 提取出的 Markdown 层级 ---")

    # 扁平列表重构为树形打印的辅助函数 (简单起见，这里直接通过缩进输出)
    all_nodes_sorted = sorted(nodes, key=lambda n: (n['level'], n['y']))
    for node in all_nodes_sorted:
        indent = "  " * node['level']
        print(f"{indent}- {node['text']}")

    return nodes

import pdfplumber
import json

def get_text_with_coordinates(pdf_path):
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



if __name__ == "__main__":
    main()

    # extract_xmind_hierarchy("/root/Data/GLS/evidence_card/证据卡片/中医药防治胃癌前病变3.7.pdf")
    # json_data = get_text_with_coordinates("/root/Data/GLS/evidence_card/证据卡片/中医药防治胃癌前病变3.7.pdf")
    # print(json_data)
    # print()
    #
    # prompt = f"""
    # 这是一个从思维导图（Xmind）导出的节点数据，包含了文本以及它们在页面上的物理坐标。
    # X 是横坐标（越小越靠左），Y 是纵坐标（越小越靠上）。
    # 请你扮演一位逻辑分析专家，执行以下任务：
    # 隐式还原逻辑：请先在底层根据 X,Y 坐标的分布规律（如从左到右、从上到下的发散关系），理清各个节点之间的“父子、并列”等逻辑关联。
    # 提取核心大意：理解这个思维导图到底在表达什么主题，包含了哪些核心分支和细节。
    # 请抛弃列表或树状格式，将整个导图表达的意思融会贯通，用一段通顺、连贯的文字总结出来。只要总结内容，逻辑清晰，语言自然流畅。
    #
    # 数据如下：
    # {json_data}
    # """
    # res = requestQwencontentInterruption("", prompt)
    # print(res)