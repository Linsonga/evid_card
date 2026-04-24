# -*- coding: utf-8 -*-
# @Time    : 14.04.2026
# @File    : publish_test.py.py
# @Software: PyCharm

"""
# 知乎发布文章
publish_test.py - Description of the file/module
"""

# -*- coding: utf-8 -*-
# @Time    : 14.04.2026
# @File    : release_zhihu.py
# @Software: PyCharm

"""
# 知乎发布文章（自动读取、过滤已发布、大模型智能匹配应季文章）
"""

# -*- coding: utf-8 -*-
# @Time    : 14.04.2026
# @File    : release_zhihu.py

import os
# 解决 Conda 库与系统 Chrome 动态库冲突的问题
if 'LD_LIBRARY_PATH' in os.environ:
    os.environ['LD_LIBRARY_PATH'] = ''

import json
import re
from datetime import datetime
from zhihu_mcp_server.server import create_atticle

# 从 utils.py 导入工具
from utils import (
    get_distributed_filenames,
    record_distribution,
    requestQwenMultiTurn,
    extract_json_array_from_text
)



def format_wechat_references(md_text: str) -> str:
    if not md_text:
        return ""

    # 1. 清理特殊空格
    md_text = md_text.replace('\xa0', ' ')

    # 2. 去掉 --- 分隔符（整行）
    md_text = re.sub(r'^\s*[-—－]{2,}\s*$', '', md_text, flags=re.MULTILINE)

    # 3. 统一参考文献标题（支持 ### / ** / 普通）
    # 1处理 ### 参考文献
    md_text = re.sub(
        r'^\s*#{1,6}\s*参考文献\s*$',
        '<br><br>参考文献<br>',
        md_text,
        flags=re.MULTILINE
    )

    # 2 处理 **参考文献**
    md_text = re.sub(
        r'^\s*\*\*参考文献\*\*\s*$',
        '<br><br>参考文献<br>',
        md_text,
        flags=re.MULTILINE
    )

    # 3 兜底（普通“参考文献”）
    md_text = re.sub(
        r'^\s*参考文献\s*$',
        '<br>参考文献<br>',
        md_text,
        flags=re.MULTILINE
    )

    # 4. 拆分正文和参考文献
    parts = md_text.split('<br>参考文献<br>')

    if len(parts) >= 2:
        ref_body = parts[-1].strip()

        # 5. 给每个 [n] 前加换行（除了第一个）
        formatted_ref_body = re.sub(
            r'(?<!^)\s*(\[|［|【)\s*(\d+)\s*(\]|］|】)',
            r'<br>[\2] ',
            ref_body
        )

        # 6. 清理开头多余 <br>
        formatted_ref_body = re.sub(r'^<br>', '', formatted_ref_body)

        parts[-1] = formatted_ref_body
        md_text = '<br>参考文献<br>'.join(parts)

    return md_text

def remove_invalid_lines(md_text: str) -> str:
    """
    极简过滤：只要该行包含“片段”或“来源文件”，就将整行删除。
    """
    # ^.*匹配行首任意内容，(?:片段|来源文件)匹配这两个词其一，.*匹配到行尾，最后带上换行符一起删掉
    pattern = r'^.*(?:片段|来源文件).*(?:\r?\n|$)'
    return re.sub(pattern, '', md_text, flags=re.MULTILINE)

def remove_fragments(text: str) -> str:
    """
    删除类似 [片段 1] / [片段 1-3] / [片段 2,3] 的标记
    """
    pattern = r'\[片段\s*\d+(?:[-,]\d+)*\]'
    return re.sub(pattern, '', text)

def get_seasonal_articles_realtime(candidate_files):
    """
    使用联网搜索获取实时热点并匹配文件
    candidate_files: 现在是包含子路径的文件列表，如 ["folder1/a.md", "b.md"]
    """
    if not candidate_files:
        return []

    today_str = datetime.now().strftime("%Y年%m月%d日")
    system_prompt = "你是一个拥有医学专业背景的社交媒体运营专家，擅长结合实时气象和流行病学数据进行内容分发。"

    user_content = f"""
    今天是 {today_str}。

    任务步骤：
    1. 请先联网搜索并分析当前时间节点下，中国大部分地区最受关注的医学预防、季节性疾病或健康生活话题（例如：柳絮过敏、某种传染病高发期、换季心血管提醒等）。
    2. 根据搜索到的实时热点，从下方的文件路径列表中，挑选出最适合在今天发布的文章。

    待筛选的文件列表（包含相对路径）：
    {json.dumps(candidate_files, ensure_ascii=False)}

    输出要求：
    请务必仅返回一个 JSON 数组，包含筛选出的文件路径字符串。
    如果没有符合热点的文件，请返回空数组 []。
    不要输出任何搜索过程或其他解释文字。
    """

    messages = [{"role": "user", "content": user_content}]

    print(f"🚀 正在联网搜索 {today_str} 的实时健康热点并递归匹配文章...")

    # 调用多轮对话接口，开启联网搜索
    response_text = requestQwenMultiTurn(
        system_prompt=system_prompt,
        messages=messages,
        enable_thinking=True,
        enable_search=True
    )

    selected_files = extract_json_array_from_text(response_text)
    return selected_files


def process_list_block(block_lines):
    result = []

    for line in block_lines:
        indent = len(line) - len(line.lstrip(' '))
        level = indent // 4

        content = re.sub(r'^\s*\*\s*', '', line).strip()

        # 识别 **标题：**
        match = re.match(r'\*\*(.*?)\*\*[:：]?\s*(.*)', content)

        if match:
            title = match.group(1).rstrip('：:')  # ✅ 防止 ：：
            rest = match.group(2)

            html = f'''
            <p style="
                margin-left:{level}em;
                line-height:1.75em;
                font-size:15px;
                color:#333;
            ">
                <strong style="color:#d32f2f;">{title}：</strong>{rest}
            </p>
            '''.strip()

        else:
            html = f'''
            <p style="margin-left:{level}em; line-height:1.75em;">
                {content}
            </p>
            '''.strip()

        result.append(html)

    return result


def format_wechat_nested_list(md_text: str) -> str:
    """
    只处理嵌套列表块，不破坏正文
    """

    lines = md_text.split("\n")
    new_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # 判断是否是列表块开始
        if re.match(r'^\s*\*\s+', line):
            block = []

            # 收集连续的 * 列表
            while i < len(lines) and re.match(r'^\s*\*\s+', lines[i]):
                block.append(lines[i])
                i += 1

            # 处理这个 block
            new_lines.extend(process_list_block(block))
        else:
            new_lines.append(line)
            i += 1

    return "\n".join(new_lines)



# 修改参考文献序号
def normalize_references(md_text: str) -> str:
    # 1. 拆分正文和参考文献
    parts = re.split(r'(参考文献)', md_text, maxsplit=1)
    if len(parts) < 3:
        return md_text  # 没有参考文献标题，直接返回

    body, ref_title, refs = parts

    # 2. 提取参考文献编号
    ref_items = re.findall(r'\[(\d+)\]', refs)

    # ==========================================
    # ✅ 新增逻辑：如果文献被删空了，连标题一起隐藏
    # ==========================================
    if not ref_items:
        # 此时说明 refs 里没有任何 [1], [2] 等文献条目
        # 我们不仅不返回 ref_title 和 refs，还要把 body 末尾的残留符号清理干净
        # 清理可能前置的 <br>、#、*、横线或多余的换行和空格
        clean_body = re.sub(r'(?:<br>|#|\*|\s|-|—|－)+$', '', body)
        return clean_body

    # 3. 构建映射 old -> new
    unique_nums = []
    for n in ref_items:
        if n not in unique_nums:
            unique_nums.append(n)

    mapping = {old: str(i + 1) for i, old in enumerate(unique_nums)}

    # 4. 删除正文中不存在的引用（如[1]）
    def clean_invalid(match):
        num = match.group(1)
        return f"[{mapping[num]}]" if num in mapping else ""

    body = re.sub(r'\[(\d+)\]', clean_invalid, body)

    # 5. 替换参考文献编号
    def replace_ref(match):
        num = match.group(1)
        return f"[{mapping[num]}]"

    refs = re.sub(r'\[(\d+)\]', replace_ref, refs)

    return body + ref_title + refs



def main():
    # --- 配置项 ---
    md_dir = "card_md"
    test_mode = True  # 设置为 True 则只成功发布一条后停止

    if not os.path.exists(md_dir):
        print(f"目录 {md_dir} 不存在。")
        return

    # 1. 递归查询所有 .md 文件
    all_md_files = []
    for root, dirs, files in os.walk(md_dir):
        for file in files:
            if file.endswith(".md"):
                # 获取相对于 md_dir 的路径，例如 "folder1/test.md"
                relative_path = os.path.relpath(os.path.join(root, file), md_dir)
                all_md_files.append(relative_path)

    if not all_md_files:
        print("未找到任何 .md 文件。")
        return

    # 2. 过滤已成功的记录 (使用相对路径作为标识)
    published_files = get_distributed_filenames(platform='zhihu')
    unpublished_files = [f for f in all_md_files if f not in published_files]

    if not unpublished_files:
        print("所有子文件夹中的文章均已发布。")
        return

    print(f"共发现 {len(all_md_files)} 个文件，待发布 {len(unpublished_files)} 个。")

    # # 3. 联网搜索实时热点并筛选
    # seasonal_files = get_seasonal_articles_realtime(unpublished_files)
    #
    # if not seasonal_files:
    #     print("未匹配到实时热点文章。")
    #     if test_mode:
    #         print("测试模式：强制选取第一个待发布文件进行测试。")
    #         seasonal_files = [unpublished_files[0]]
    #     else:
    #         return
    seasonal_files = [unpublished_files[0]]
    # 4. 批量/单条发布
    success_count = 0
    for rel_path in seasonal_files:
        # 构建完整物理路径
        full_path = os.path.join(md_dir, rel_path)

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            # 调用公共方法
            # content = clean_markdown_for_wechat(content)
            # 删除类似 [片段 1] / [片段 1-3] / [片段 2,3] 的标记
            content = remove_fragments(content)

            # ✅ 新增：简单粗暴地删除任何包含“片段”或“来源文件”的行
            content = remove_invalid_lines(content)

            # 参考文献
            content = format_wechat_references(content)

            # 层级效果
            # content = format_wechat_nested_list(content)

            content = re.sub(r'^#\s+.*(?:\r?\n|$)', '', content, count=1).strip()

            # 修改参考文献序号
            content = normalize_references(content)

            # 使用文件名（不含路径和后缀）作为标题
            file_name = os.path.basename(rel_path)
            title = os.path.splitext(file_name)[0].replace('_', '')[:100]

            print(f"\n正在发布路径为 [{rel_path}] 的文章...")

            # 调用知乎发布接口
            result = create_atticle(title=title, content=content, images=[], topic="")
            result_text = result[0].text if result else "无返回结果"

            print(f"发布结果: {result_text}")

            # 记录分发结果 (存入 rel_path 确保唯一性)
            record_distribution('zhihu', rel_path, 'success', result_text)
            success_count += 1

            break

            if test_mode and success_count >= 1:
                print("✅ 今日发文任务已完成（1篇），程序正常退出。")
                break

        except Exception as e:
            print(f"处理文件 {rel_path} 失败: {e}")
            record_distribution('zhihu', rel_path, 'failed', str(e))
        exit()


if __name__ == "__main__":
    main()


# from zhihu_mcp_server.server import create_atticle
#
#
# def main():
#     # 准备发布的内容
#     title = "测试在Linux服务器通过自动脚本发布知乎文章"  # 文章标题，不多于100个字符
#     content = "这是一篇测试文章的内容。测试使用 Cookies 在 Linux 无头浏览器环境下自动发布文章是否成功。"  # 文章内容，不少于9个字
#     images = []  # 如果有封面图片，填入本地绝对路径，例如 ["/root/images/cover.jpg"]
#     topic = "测试"  # 最好输入已存在的话题，否则可能无法成功发帖
#
#     print(f"准备发布文章: {title}")
#
#     # 调用发文接口 (注意代码中的原方法名为 create_atticle)
#     result = create_atticle(title=title, content=content, images=images, topic=topic)
#
#     print("执行结果:", result[0].text)
#
#
# if __name__ == "__main__":
#     main()