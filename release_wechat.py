# -*- coding: utf-8 -*-
# @Time    : 14.04.2026
# @File    : release_wechat.py
# @Software: PyCharm

"""
# 微信发布文章
# 微信发布文章（自动读取、过滤已发布、大模型智能匹配应季文章）
publish_test.py - Description of the file/module
"""

# -*- coding: utf-8 -*-
# @Time    : 14.04.2026
# @File    : release_wechat.py

"""
# 微信发布文章（自动读取、过滤已发布、大模型智能匹配应季文章、MCP服务器发布）
"""

import os
import json
import re
import asyncio
from datetime import datetime

# MCP 相关依赖
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# 从 utils.py 导入工具
from utils import (
    get_distributed_filenames,
    record_distribution,
    requestQwenMultiTurn,
    extract_json_array_from_text
)

# 导入微信配置 (确保 config.py 中存在这几个常量)
from config import WECHAT_APP_ID, WECHAT_APP_SECRET


def get_seasonal_articles_realtime(candidate_files):
    """
    使用联网搜索获取实时热点并匹配文件
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


async def run_wechat_mcp_example(articles):
    """
    批量发布微信公众号文章（最多8篇打包为一次多图文发布）
    :param articles: 文章列表，每个元素包含 {title, content, image_paths, rel_path}
    :return: 包含发布结果状态的列表
    """
    # 1. 配置 MCP Server 启动参数
    server_script_path = os.path.abspath("./wechat-publisher-mcp/src/server.js")

    server_params = StdioServerParameters(
        command="node",
        args=[server_script_path],
        env=None
    )

    print(f"正在启动并连接到 MCP Server...")
    publish_results = []

    # 2. 建立 stdio 连接
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 3. 初始化 MCP 会话
            await session.initialize()
            print("✅ MCP 会话初始化成功！")

            # --- 核心修改：打包组装所有文章 ---
            mcp_articles = []
            rel_paths = [] # 记录关联的本地文件路径，方便后续写库

            # 遍历文章列表（微信最多一次发8篇，组成多图文）
            for article in articles[:8]:
                title = article.get("title", "")
                content = article.get("content", "")
                image_paths = article.get("image_paths", [])
                rel_path = article.get("rel_path", "")

                article_data = {
                    "title": title,
                    "content": content,
                    "author": "小灵AI科普"
                }

                # 图片处理
                if image_paths:
                    if isinstance(image_paths, list) and len(image_paths) > 0:
                        article_data["coverImagePath"] = image_paths[0]
                    elif isinstance(image_paths, str):
                        article_data["coverImagePath"] = image_paths
                else:
                    # 如果不需要封面可以不传这个字段，或者传空
                    pass

                mcp_articles.append(article_data)
                rel_paths.append(rel_path)

            if not mcp_articles:
                return []

            # 构建最终发给 MCP Server 的参数
            arguments = {
                "articles": mcp_articles,  # 将多篇文章作为一个数组传给服务端
                "appId": WECHAT_APP_ID,
                "appSecret": WECHAT_APP_SECRET,
                "previewMode": False,
            }

            # 4. 调用发布文章工具 (只调用一次！)
            print(f"\n正在尝试调用 'wechat_publish_article' 打包发布 {len(mcp_articles)} 篇文章...")
            try:
                result = await session.call_tool(
                    "wechat_publish_article",
                    arguments=arguments
                )

                # 收集返回结果
                result_text = ""
                for content_item in result.content:
                    if content_item.type == "text":
                        print(f"服务器返回: {content_item.text}")
                        result_text += content_item.text + " "

                # 因为是整体多图文打包发布，所以如果成功，列表中所有的文章都算成功
                for rel_path in rel_paths:
                    publish_results.append({
                        "rel_path": rel_path,
                        "status": "success",
                        "msg": result_text
                    })

            except Exception as e:
                print(f"\n❌ 多图文打包发布调用失败: {e}")
                # 如果失败，列表中所有的文章都算失败
                for rel_path in rel_paths:
                    publish_results.append({
                        "rel_path": rel_path,
                        "status": "failed",
                        "msg": str(e)
                    })

    return publish_results



def main():
    # --- 配置项 ---
    md_dir = "card_md"
    test_mode = True  # 设置为 True 则只选取较少数量测试
    # --------------

    if not os.path.exists(md_dir):
        print(f"目录 {md_dir} 不存在。")
        return

    # 1. 递归查询所有 .md 文件
    all_md_files = []
    for root, dirs, files in os.walk(md_dir):
        for file in files:
            if file.endswith(".md"):
                relative_path = os.path.relpath(os.path.join(root, file), md_dir)
                all_md_files.append(relative_path)

    if not all_md_files:
        print("未找到任何 .md 文件。")
        return

    # 2. 过滤已成功发布的记录 (注意：这里 platform 改为了 'wechat')
    published_files = get_distributed_filenames(platform='wechat')
    unpublished_files = [f for f in all_md_files if f not in published_files]

    if not unpublished_files:
        print("所有子文件夹中的文章均已发布到微信。")
        return

    print(f"共发现 {len(all_md_files)} 个文件，待发布 {len(unpublished_files)} 个。")

    # 3. 联网搜索实时热点并筛选
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
    # 4. 构建待发布的 articles 列表数据
    articles_to_publish = []

    # 微信一次最多只能传 8 篇文章进行多图文发布
    for rel_path in seasonal_files[:8]:
        full_path = os.path.join(md_dir, rel_path)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            # 微信公众号排版清理（如需）
            # 处理加粗符号导致的多余星号问题 (去除文本中的 **)
            content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)

            file_name = os.path.basename(rel_path)
            title = os.path.splitext(file_name)[0].strip('_')[:100]

            articles_to_publish.append({
                "title": title,
                "content": content,
                "image_paths": [],  # 如有图片可在此添加
                "rel_path": rel_path  # 隐式传递，用于后续数据库记录
            })
        except Exception as e:
            print(f"读取文件 {rel_path} 失败: {e}")

    if not articles_to_publish:
        return

    # 5. 调用 asyncio 异步执行微信 MCP 发文模块
    results = asyncio.run(run_wechat_mcp_example(articles_to_publish))

    # 6. 回调处理，记录发布成功/失败的数据到本地数据库记录中
    success_count = 0
    for res in results:
        record_distribution(
            platform='wechat',
            filename=res['rel_path'],
            status=res['status'],
            message=res['msg']
        )
        if res['status'] == 'success':
            success_count += 1

    print(f"本轮微信发文结束，总计尝试 {len(results)} 篇，成功 {success_count} 篇。")


if __name__ == "__main__":
    main()