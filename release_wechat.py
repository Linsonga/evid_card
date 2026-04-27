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
# from mdnice import to_wechat

# 从 utils.py 导入工具
from utils import (get_distributed_filenames, record_distribution, requestQwenMultiTurn,
                   extract_json_array_from_text, generate_ai_cover_dashscope_api
)

# 导入微信配置 (确保 config.py 中存在这几个常量)
from config import WECHAT_APP_ID, WECHAT_APP_SECRET


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


def get_seasonal_articles_realtime(candidate_files, limit=8):
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
    2. 根据搜索到的实时热点，从下方的文件路径列表中，挑选出最适合在今天发布的文章。请挑选出 {limit} 篇文章（如果符合热点的文章很多，请尽量选满 {limit} 篇）。
    3. 降级备选策略：如果文件列表中没有任何文章符合当前的实时热点话题，请启动备选方案，从列表中挑选出 {limit} 篇关于“常年热门/高发疾病”（如心脑血管疾病、三高、常见肿瘤、常见儿科疾病等）或“大众普遍关注的常规健康科普”文章。
    
    待筛选的文件列表（包含相对路径）：
    {json.dumps(candidate_files, ensure_ascii=False)}

    输出要求：
    请务必仅返回一个 JSON 数组，包含筛选出的文件路径字符串。
    如果既没有符合实时热点的文件，也没有符合常年热门疾病的文件，才返回空数组 []。
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


async def run_wechat_draft(articles):
    """
    将文章保存为微信公众号草稿（不发布），用于预览排版效果
    :param articles: 文章列表，每个元素包含 {title, content, image_paths, rel_path}
    :return: 包含保存结果状态的列表
    """
    server_script_path = os.path.abspath("./wechat-publisher-mcp/src/server.js")
    server_params = StdioServerParameters(
        command="node",
        args=[server_script_path],
        env=None
    )

    print(f"正在启动并连接到 MCP Server（草稿模式）...")
    draft_results = []

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ MCP 会话初始化成功！")

            mcp_articles = []
            rel_paths = []

            for article in articles[:8]:
                title = article.get("title", "")
                content = article.get("content", "")
                image_paths = article.get("image_paths", [])
                rel_path = article.get("rel_path", "")

                article_data = {
                    "title": title,
                    "content": content,
                    "author": "小灵"
                }

                if image_paths:
                    if isinstance(image_paths, list) and len(image_paths) > 0:
                        article_data["coverImagePath"] = image_paths[0]
                    elif isinstance(image_paths, str):
                        article_data["coverImagePath"] = image_paths

                mcp_articles.append(article_data)
                rel_paths.append(rel_path)

            if not mcp_articles:
                return []

            arguments = {
                "articles": mcp_articles,
                "appId": WECHAT_APP_ID,
                "appSecret": WECHAT_APP_SECRET,
            }

            print(f"\n正在将 {len(mcp_articles)} 篇文章保存到草稿箱...")
            try:
                result = await session.call_tool(
                    "wechat_save_draft",
                    arguments=arguments
                )

                result_text = ""
                for content_item in result.content:
                    if content_item.type == "text":
                        print(f"服务器返回: {content_item.text}")
                        result_text += content_item.text + " "

                for rel_path in rel_paths:
                    draft_results.append({
                        "rel_path": rel_path,
                        "status": "success",
                        "msg": result_text
                    })

            except Exception as e:
                print(f"\n❌ 保存草稿失败: {e}")
                for rel_path in rel_paths:
                    draft_results.append({
                        "rel_path": rel_path,
                        "status": "failed",
                        "msg": str(e)
                    })

    return draft_results


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
                    "author": "小灵"
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
            }

            # 4. 调用发布文章工具 (只调用一次！)
            print(f"\n正在尝试调用 'wechat_mass_send' 打包发布 {len(mcp_articles)} 篇文章...")
            try:
                result = await session.call_tool(
                    # "wechat_publish_article",
                    "wechat_mass_send",
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


def wechat_indent_formatter(match):
    # 捕获组 1：获取列表符号前面的空格
    leading_spaces = match.group(1)

    # 情况 1：如果没有空格，说明是一级列表（如：研究类型、1. 动态监测）
    # 直接去掉符号，并加上换行符隔离段落
    if not leading_spaces:
        return '\n'

    # 情况 2：如果有空格，说明是子列表（如：血清 SOD、脑电图）
    # Markdown 默认用 4 个普通空格表示一层缩进
    # 我们将其转换为 2 个“中文全角空格”（注意：不要改成普通空格），微信绝对无法吃掉全角空格
    # len(leading_spaces) // 2 意味着：4个普通空格 -> 2个全角空格
    # full_width_spaces = '  ·' * (len(leading_spaces) // 1)
    full_width_spaces = '·'

    # 返回换行符 + 强制全角空格缩进
    return '\n' + full_width_spaces


def clean_markdown_for_wechat(md_text: str) -> str:
    """
    终极微信排版清洗函数（防吞换行版）：
    强行拍平所有 Markdown 列表和加粗标识，转换为最普通的纯文本符号排版。
    """
    if not md_text:
        return ""


    # # 假设原始文本在 text 变量中
    # # 将替换目标从 '' (空) 改为 '\n' (换行符)
    # md_text = re.sub(r'^\s*\d+\.\s+', '\n', md_text, flags=re.MULTILINE)
    #
    # # 如果您也想处理无序列表（* 或 -），同样替换为 '\n'
    # md_text = re.sub(r'^\s*[\*\-]\s+', '\n', md_text, flags=re.MULTILINE)

    md_text = re.sub(r'^(\s*)(?:\d+\.|[\*\-])\s+', wechat_indent_formatter, md_text, flags=re.MULTILINE)


    # # 0. 清理网页复制带来的不可见字符
    # md_text = md_text.replace('\xa0', ' ')
    #
    # # ==========================================
    # # 1. 修复【一级数字列表】
    # # 将 \s 替换为 [ \t]，绝对不吞换行符
    # # ==========================================
    # pattern_num = r'^([ \t]*\d+\.[ \t]*)\*\*(.*?)\*\*(?:：|:)?[ \t]*'
    # md_text = re.sub(pattern_num, r'\1【\2】：', md_text, flags=re.MULTILINE)
    #
    # # ==========================================
    # # 2. 修复【嵌套的无序列表+加粗标题】（解决必要条件和辅助条件挤在一行）
    # # ==========================================
    # pattern_bullet_bold = r'^[ \t]{1,}[\*\-][ \t]+\*\*(.*?)\*\*(?:：|:)?[ \t]*'
    # md_text = re.sub(pattern_bullet_bold, r'  ▶ 【\1】：', md_text, flags=re.MULTILINE)
    #
    # # ==========================================
    # # 3. 兜底修复【普通嵌套无序列表】
    # # ==========================================
    # pattern_bullet_normal = r'^[ \t]{1,}[\*\-][ \t]+'
    # md_text = re.sub(pattern_bullet_normal, r'  ▶ ', md_text, flags=re.MULTILINE)

    return md_text




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


#
# def format_wechat_references(md_text: str) -> str:
#     if not md_text:
#         return ""
#
#     # 1. 清理特殊空格
#     md_text = md_text.replace('\xa0', ' ')
#
#     # 2. 去掉 --- 分隔符
#     md_text = re.sub(
#         r'[-—－]{2,}\s*(\*{0,2}参考文献\*{0,2})',
#         r'\1',
#         md_text
#     )
#
#     # 3. 在“参考文献”后面强制加 <br>
#     md_text = re.sub(
#         r'(\*{0,2}参考文献\*{0,2})',
#         r'\1<br>',
#         md_text
#     )
#
#     # 4. 拆分正文和参考文献部分
#     parts = re.split(r'(\*{0,2}参考文献\*{0,2}<br>)', md_text)
#
#     if len(parts) >= 3:
#         ref_body = parts[-1].strip()
#
#         # 5. 给每个 [n] 前加换行（除了第一个）
#         formatted_ref_body = re.sub(
#             r'(?<!^)\s*(\[|［|【)\s*(\d+)\s*(\]|］|】)',
#             r'<br>[\2] ',
#             ref_body
#         )
#
#         # 6. 清理开头可能多余的 <br>
#         formatted_ref_body = re.sub(r'^<br>', '', formatted_ref_body)
#
#         parts[-1] = formatted_ref_body
#         md_text = "".join(parts)
#
#     return md_text
# import re

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


def main():
    # --- 配置项 ---
    md_dir = "/root/autodl-tmp/evidence_card_online/card_md"
    test_mode = True  # 设置为 True 则只选取较少数量测试
    batch_size = 8
    draft_mode = False # 设置为 True 则只保存草稿，不正式发布（用于预览排版效果）
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
    seasonal_files = get_seasonal_articles_realtime(unpublished_files, limit=batch_size)

    if not seasonal_files:
        print("未匹配到实时热点文章。")
        if test_mode:
            print("测试模式：强制选取第一个待发布文件进行测试。")
            seasonal_files = unpublished_files[:batch_size]
        else:
            return

    # seasonal_files = unpublished_files[:batch_size]
    # 4. 构建待发布的 articles 列表数据
    articles_to_publish = []

    # 微信一次最多只能传 8 篇文章进行多图文发布
    for rel_path in seasonal_files[:batch_size]:
        full_path = os.path.join(md_dir, rel_path)
        try:

            file_name = os.path.basename(rel_path)
            title = os.path.splitext(file_name)[0].strip('_')

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
            content = format_wechat_nested_list(content)

            content = re.sub(r'^#\s+.*(?:\r?\n|$)', '', content, count=1).strip()

            # 修改参考文献序号
            content = normalize_references(content)

            print(f"正在为生成 AI 封面...")
            cover_abs_path = generate_ai_cover_dashscope_api(title=title)
            image_paths_list = []
            if cover_abs_path and os.path.exists(cover_abs_path):
                image_paths_list.append(cover_abs_path)
            else:
                print(f"警告: 封面图生成失败或路径无效: {cover_abs_path}")
            cover_path = image_paths_list[0] if image_paths_list else ""

            articles_to_publish.append({
                "title": title,
                "content": content,
                "image_paths": cover_path,  # 如有图片可在此添加
                "rel_path": rel_path  # 隐式传递，用于后续数据库记录
            })
        except Exception as e:
            print(f"读取文件 {rel_path} 失败: {e}")

    if not articles_to_publish:
        return

    # 5. 调用 asyncio 异步执行微信 MCP 模块
    if draft_mode:
        # 仅保存为草稿，用于在微信公众号后台预览排版效果
        print(f"\n📝 草稿模式：将 {len(articles_to_publish)} 篇文章保存到草稿箱（不发布）...")
        results = asyncio.run(run_wechat_draft(articles_to_publish))
        success_count = sum(1 for r in results if r['status'] == 'success')
        print(f"草稿保存结束，总计尝试 {len(results)} 篇，成功 {success_count} 篇。")
        print("💡 请登录微信公众号后台 → 草稿箱 查看排版效果。")
        return  # 草稿模式不写入数据库，不影响已发布记录

    # 正式发布模式
    results = asyncio.run(run_wechat_mcp_example(articles_to_publish))

    # 6. 回调处理，记录发布成功/失败的数据到本地数据库记录中
    success_count = 0
    for res in results:
        record_distribution(
            platform='wechat',
            filename=res['rel_path'],
            status=res['status'],
            note=res['msg']
        )
        if res['status'] == 'success':
            success_count += 1

    print(f"本轮微信发文结束，总计尝试 {len(results)} 篇，成功 {success_count} 篇。")


if __name__ == "__main__":
    main()







