# -*- coding: utf-8 -*-
# @Time    : 14.04.2026
# @File    : release_wechat.py
# @Software: PyCharm

"""
# 微信发布文章
# 微信发布文章（自动读取、过滤已发布、大模型智能匹配应季文章）
publish_test.py - Description of the file/module
"""


import os
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


def main():
    # --- 配置项 ---
    md_dir = "card_md"
    test_mode = True  # 设置为 True 则只成功发布一条后停止
    # --------------

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

    # # # 3. 联网搜索实时热点并筛选
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

            # ========== 新增：处理知乎列表双重序号问题 ==========
            # 使用正则，将行首的 "1. ", "2. " 替换为 "1、", "2、"
            # flags=re.MULTILINE 表示让 ^ 匹配每一行的开头
            # ========== 处理知乎排版 Bug ==========
            # ========== 1. 处理正文内容 (知乎排版 Bug) ==========
            # 处理知乎列表双重序号问题 (将 "1. " 替换为 "1、")
            content = re.sub(r'^(\s*\d+)\.\s+', r'\1、', content, flags=re.MULTILINE)

            # 处理加粗符号导致的多余星号问题 (去除文本中的 **)
            content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)

            # 将无序列表的星号 "* " 替换为中文小圆点 "· "，排版更美观
            content = re.sub(r'^\s*\*\s+', '· ', content, flags=re.MULTILINE)
            # ====================================================

            # 使用文件名（不含路径和后缀）作为标题
            file_name = os.path.basename(rel_path)
            # title = os.path.splitext(file_name)[0][:100]
            title = os.path.splitext(file_name)[0].strip('_')[:100]

            print(f"\n正在发布路径为 [{rel_path}] 的文章...")

            # 调用知乎发布接口
            result = create_atticle(title=title, content=content, images=[], topic="医疗科研")
            result_text = result[0].text if result else "无返回结果"

            print(f"发布结果: {result_text}")

            # 记录分发结果 (存入 rel_path 确保唯一性)
            record_distribution('zhihu', rel_path, 'success', result_text)
            success_count += 1

            break
            # if test_mode and success_count >= 1:
            #     print("\n💡 测试模式：已成功发布一条，程序退出。")
            #     break

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