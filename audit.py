import os
import re
import json
import asyncio
from datetime import datetime
import pymysql
from pymilvus import MilvusClient
from utils import call_qwen, extract_json_from_text, vector_4b, requestQwenMultiTurn
from logger import logger
from config import (
    DB_CONFIG, CARD_API_URL, EVID_DESC_URL,
    MILVUS_MAIN_URI, MILVUS_MAIN_TOKEN, MILVUS_COLLECTION_MAIN, WECHAT_APP_ID, WECHAT_APP_SECRET
)
from filelock import FileLock
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from database import get_db_connection


# 初始化 Milvus 客户端
try:
    milvus_client = MilvusClient(uri=MILVUS_MAIN_URI, token=MILVUS_MAIN_TOKEN)
except Exception as e:
    logger.warning(f"Milvus 连接失败，如果遇到幻觉检测可能会报错: {e}")
    milvus_client = None


def score_data(title, info):
    """Step 1: 评分"""
    system_prompt = """你是一名资深的医学文献质量评估专家。你的任务是对给定的医学文献“核心结论（core_conclusion）”进行质量打分和点评。
    你需要根据文献标题（title）来判断结论的“相关性”，并根据严格的格式和内容标准来评估结论的“规范性”和“信息密度”。
    满分为100分。请务必保持客观、严谨，并以JSON格式输出评估结果。"""

    user_prompt = f"""
        请根据以下评价维度和标准，对提供的医学文献标题及其核心结论进行打分。
        
        ### 1. 输入数据
        **文献标题 (Title):** 
        {title}
        **核心结论 (Core Conclusion):** 
        {info}
        
        ---
        ### 2. 评分维度与标准 (总分 100 分)
        
        请从以下四个维度进行扣分或得分评估：
        
        **A. 核心主旨契合度 (35分) [核心指标]**
        *   **满分标准**：结论内容与标题高度呼应，直接且准确地回答了标题所隐含的临床问题（如疗效对比、发病机制、不良反应等）。
        *   **扣分项**：
            *   答非所问或严重跑题（扣 20-35 分）。
            *   仅回答了标题的部分内容，遗漏了标题中的关键变量（扣 10-20 分）。
        
        **B. 逻辑一致性与自洽 (35分) [核心指标]**
        *   **满分标准**：结论内部逻辑严密，因果关系清晰，论述连贯无跳跃，不存在前后矛盾的表述。
        *   **扣分项**：
            *   存在明显的前后矛盾或逻辑谬误（扣 20-35 分）。
            *   语句拼凑感强，因果推导牵强或缺乏连贯性（扣 10-20 分）。
        
        **C. 医学严谨性与客观性 (20分)**
        *   **满分标准**：结论表述客观克制，符合医学科研的严谨语境，准确界定了结论的适用范围，无过度推断。
        *   **扣分项**：
            *   使用了不符合医学严谨性的绝对化词汇（如“彻底治愈”、“完全无效”、“100%”）或存在夸大/主观臆断倾向（扣 10-20 分）。
            *   未能区分“相关性”与“因果性”，或推论明显超出了标题限定的研究范畴（扣 5-10 分）。
        
        **D. 专业表达与精炼度 (10分)**
        *   **满分标准**：医学术语使用准确，语言精炼，无冗余废话。
        *   **扣分项**：
            *   口语化严重、使用非医学专业表述，或存在明显的语病（扣 5-10 分）。
            *   篇幅冗长且包含大量无实质指导意义的废话（扣 5 分）。
        
        ---
        ### 3. 输出要求
        请严格按照以下 JSON 格式输出，不要包含任何 Markdown 代码块标记（如 ```json 等），仅输出纯文本 JSON 字符串：
        {{
            "score": <0-100之间的整数>,
            "reasoning": "简要说明打分理由，指出优点和具体的扣分点（不超过100字）。",
            "dimensions": {{
                "alignment": <A维度得分>,
                "logic": <B维度得分>,
                "rigor": <C维度得分>,
                "professionalism": <D维度得分>
            }},
            "suggestion": "如果有改进空间，用一句话给出修改建议；如果是满分，则留空。"
        }}
        """

    result_text = call_qwen(system_prompt, user_prompt)
    return extract_json_from_text(result_text)


def check_timeliness(title, core_conclusion):
    """审核: 时效性与准确性 (极度宽容版)"""
    prompt = f"""
    # Input Data
    - 文献标题 (Title): {title}
    - 核心结论 (Core Conclusion): {core_conclusion}

    # Action (必须执行联网搜索)
    你现在的角色是一个“底线防御员”。你的任务不是苛求完美，而是仅仅拦截那些【纯粹捏造】或【存在致命硬伤】的信息。请按以下极度宽容的标准执行验证：

    1. **存在性模糊验证**：
       - 搜索该文献。允许存在中英文翻译差异、副标题缺失、或是略微的字词偏差。只要能找到研究对象和主题高度重合的文献，即视为文献真实存在。
    2. **局部事实性验证 (极度宽容)**：
       - 阅读原文献的摘要。输入的 `Core Conclusion` 不需要完美概括全文。
       - 只要该结论是原文献中的**任意一个真实发现**（例如次要结果、某个特定对照组的数据、或者是合理的延伸推论），就视为准确。
       - 完全忽略结论中可能附带的夸大修饰词（如“最新”、“颠覆性”、“首次发现”等），只评估核心事实本身。
    3. **时效性与状态检查 (仅查致命伤)**：
       - 仅在一种情况下判定时效性不合格：该文献已被官方明确标记为**撤稿（Retracted）**。
       - 只要没有被撤稿，哪怕是几十年前的文献，一律视为“具有历史事实有效性”。

    # Judgement Criteria (判断标准)
        - **has_issue = true (存在致命硬伤，必须拦截)**:
        1. 纯粹虚构/严重幻觉：网络上完全找不到任何与该标题及结论相关的主题研究，纯属 AI 捏造。
        2. 事实完全反转：结论与原文献的真实结果南辕北辙（例如：原文说A药物无效，结论说A药物疗效显著）。
        3. 已撤稿：文献由于造假或严重错误已被官方撤回。

        - **has_issue = false (底线之上，予以放行)**:
        1. 文献存在，且结论能从原文献的某个角落找到依据（哪怕经过了高度改写或只提取了次要信息）。
        2. 存在翻译瑕疵或夸大修饰词，但核心事实没有扭曲。

    # Output Format
    请仅返回一个纯 JSON 格式的数据，不要包含Markdown标记（如 ```json ... ```），格式如下：
    {{
        "has_issue": boolean, 
        "reason": "极简说明理由（若为true，明确指出是虚构、反转还是撤稿；若为false，简述验证依据）"
    }}
    (注意: 请竭尽全力寻找让其通过(false)的理由，只有在确凿无疑存在致命硬伤时才输出true)
    """
    res = call_qwen(
        "你是一个宽容的数据底线审查员。你的任务是利用网络搜索，拦截那些纯粹虚构、结论完全反转或已被撤稿的无效文献信息，对非致命瑕疵予以放行。",
        prompt, enable_search=True)
    parsed = extract_json_from_text(res)

    if "is_latest" in parsed:
        parsed["has_issue"] = not parsed["is_latest"]
    return parsed


def check_content_interruption(title, info):
    """审核: 标题与内容断裂 (宽容版)"""
    system_prompt = """你是一个注重临床实用性的医学内容审核专家。你的核心审核原则是“宽容匹配”：只要标题与内容有合理的逻辑关联，且不产生严重的临床误导，即视为无断裂。

    请严格按照以下【宽松标准】进行判断：

    【判定为存在断裂（has_issue=true）的情况】（仅限严重偏离，需谨慎判定）
    1. 完全矛盾：结论与标题表达的核心方向完全相反（如标题宣称某药“有效/安全”，结论却明确指出“无效/风险极高”）。
    2. 主题脱节：标题与结论讨论的疾病、药物或核心机制完全无关。

    【判定为不存在断裂（has_issue=false）的情况】（绝大多数情况应归于此类）
    1. 概念泛化包容：标题使用了宽泛的疾病/人群概念（如“糖尿病”、“老年人”），而结论针对的是具体的亚型（如“2型糖尿病合并肾病”、“75岁以上高血压老人”）。这是正常的标题吸引策略，不算断裂。
    2. 局部支撑：结论只回答或支撑了标题中的部分核心变量或次要变量，只要大方向一致即可。
    3. 隐性关联：标题提出一个临床疑问或方向，结论给出了前提条件、用药细节或机制延伸，只要专业人员能看出两者在同一个大语境下即可。
    4. 程度差异：标题语气较为绝对或吸睛，但结论语气较为保守或有条件限制，只要不构成完全反转，均算通过。

    请坚持“疑罪从无”的原则：如果不确定是否断裂，一律判定为不存在断裂（false）。

    必须严格返回JSON格式：{"has_issue": boolean, "reason": "说明"} (has_issue为true代表存在严重断裂或严重不一致)"""

    user_prompt = f"【文章标题】：{title}\n【核心结论】：{info}"

    res = call_qwen(system_prompt, user_prompt)
    # logger.info(res)
    return extract_json_from_text(res)
# def check_content_interruption(title, info):
#     """审核: 标题与内容断裂"""
#     system_prompt = """你是一个资深的医学学术编辑。请严格按照以下标准判断：
#     【判定为存在断裂（has_issue=true）的情况】
#     1. 标题研究对象范围明显大于结论支持范围（如标题为广泛人群，但结论仅限特定亚组）
#     2. 标题强调的核心变量未在结论中体现
#     3. 标题与结论研究重点不一致
#
#     【判定为不存在断裂（has_issue=false）的情况】
#     1. 结论能够支持标题核心命题（即使在特定人群中）
#     2. 方向一致，仅存在合理范围缩小
#
#     请只按上述标准判断，不允许自由发挥。
#     必须严格返回JSON格式：{"has_issue": boolean, "reason": "说明"} (has_issue为true代表存在断裂或不一致)"""
#     user_prompt = f"【文章标题】：{title}\n【核心结论】：{info}"
#
#     res = call_qwen(system_prompt, user_prompt)
#     # logger.info(res)
#     return extract_json_from_text(res)

def split_and_extract(text):

    # 先按行拆
    lines = [line.strip().lstrip("- ").strip() for line in text.split("\n") if line.strip()]

    # 再按句号拆
    sentences = []
    for line in lines:
        parts = re.split(r"[。！？]", line)
        sentences.extend([p.strip() for p in parts if p.strip()])
    return sentences


def check_hallucination(title, info, info_guide):
    # 分句抽取
    info_guide = split_and_extract(info_guide)
    # logger.info(str(info_guide))

    final_text_list = []

    try:
        for sentence in info_guide:
            if not sentence.strip():
                continue

            # 1. 单句向量化
            query_vector = vector_4b(sentence)

            # 2. 单条检索（limit=2，多拿一条防偏）
            results = milvus_client.search(
                collection_name=MILVUS_COLLECTION_MAIN,
                data=[query_vector],
                limit=3,
                output_fields=["abstract"],
                # 注意：这里最好加上参数返回 score/distance 以便过滤
            )

            # 3. 取结果并加入【相似度过滤】
            for hits in results:
                for hit in hits:
                    # 【重要修改1】：过滤掉相似度过低的检索结果（假设是余弦相似度，阈值需根据你实际的4b模型调整，比如 0.5）
                    # 避免向量库里没有该知识时，强行返回一篇毫不相干的文章导致大模型误判
                    score = hit.distance  # 或者是 hit.score，取决于你的Milvus距离度量设置
                    print("向量库，查询出来得分情况")
                    print(score)
                    # 这里的阈值(例如0.5)需要你通过测试打印score来确定一个合理的值
                    # 假设 score > 0.5 代表有相关性
                    if score < 0.5:
                        continue

                    text = hit.entity.get("abstract", "")
                    if text and text not in final_text_list:  # 去重
                        final_text_list.append(text)

    except Exception as e:
        # RAG崩溃时不应该直接认为没有issue，或者视你的业务容忍度而定
        logger.error(f"RAG检索失败: {str(e)}")
        final_text_list = []  # 降级为无参考事实模式

    # 4. 拼接
    final_text = "\n".join(final_text_list)
    # logger.info("-------------")
    # logger.info(f"检索到的参考事实: {final_text if final_text else '无相关参考事实'}")

    # 【重要修改2】：重构 Prompt，引入"未提及"时的内部知识兜底机制
    user_prompt = f"""你是一个严谨的高级医学质检专家。
    请根据【文章标题】和提供的【参考事实】，评估【生成的结论】是否存在事实错误或数据捏造。

    【文章标题】：{title}
    【参考事实】：{final_text if final_text else "无相关参考事实。"}
    【生成的结论】：{info}

    请按以下逻辑进行深度核查：
    1. 事实比对：
       - [支持]：如果【生成的结论】与【参考事实】一致，视为无捏造。
       - [矛盾]：如果【参考事实】中明确写了 HR=0.8，而【结论】写了 HR=0.5，这是绝对的“捏造/幻觉”。
       - [未提及]：如果【参考事实】为"无相关参考事实"，或未包含结论中的具体数据（如试验代号、罕见病理数据），**请调用你作为医疗大模型的专业医学知识进行判断**。如果该数据在真实医学世界中是极其公认且正确的，则不视为捏造；如果你判断该数据在医学常识上明显错误或违背客观事实，则视为捏造。

    请严格返回JSON格式（不要输出其他废话）：
    {{
        "has_issue": boolean, // true表示有严重事实错误或捏造，false表示事实正确或合理
        "reason": "如果has_issue为true，请指出具体是与参考事实矛盾，还是违背了医学客观事实；如果为false，请简述理由（如符合参考事实，或参考事实未提及但符合医学常识）"
    }}"""

    try:
        res = call_qwen("只返回JSON。", user_prompt)
        return extract_json_from_text(res)
    except Exception as e:
        logger.error(f"调用LLM判断幻觉失败: {str(e)}")
        return {"has_issue": False, "reason": "大模型校验异常"}

def check_dynamic_issue(title, core_conclusion, issue_name):
    """审核: 通用动态问题"""
    system_prompt = """你是一个医学文本质检助手。判断文本是否犯了指定的错误。
    返回JSON：{"has_issue": boolean, "reason": "说明"} (has_issue为true代表犯了该错误)"""
    user_prompt = f"文章标题: {title}\n核心结论: {core_conclusion}\n\n需要检查的错误类型:【{issue_name}】\n请判断上述文本是否包含该错误？"

    res = call_qwen(system_prompt, user_prompt)

    # logger.info(res)

    return extract_json_from_text(res)


def build_materials_and_references(materials_part_data):
    """
    从 materials_part 数据中提取并格式化：
    1. 给大模型阅读的带编号正文材料 (formatted_materials)
    2. 放在文末的标准参考文献列表 (reference_list)
    """
    formatted_materials = []
    formatted_references = []
    ref_index = 1

    # 遍历外部数组
    for item in materials_part_data:
        # 获取内嵌的 info JSON 字符串
        info_str = item.get("info", "{}")

        try:
            info_dict = json.loads(info_str)
        except json.JSONDecodeError:
            continue  # 如果解析 JSON 失败，跳过该条目

        # 提取 reference 列表（包含了论文、指南等）
        references = info_dict.get("reference", [])

        for ref in references:
            # === 1. 提取作者/制定者 ===
            author_list = ref.get("author", [])
            if author_list:
                author = ";".join(author_list)
            else:
                # 指南类文献通常没有 author 字段，而是有 zdz (制定者)
                author = ref.get("zdz", "佚名")

            # === 2. 提取标题 ===
            # 有的叫 title，有的叫 literatureTitle，这里做个兼容
            title = ref.get("title") or ref.get("literatureTitle", "未知标题")

            # === 3. 提取期刊出处 ===
            journal = ref.get("journal", "")
            if not journal:
                # 指南类没有 journal，通常在 cc 字段，格式如 "中医杂志.2021..."
                cc = ref.get("cc", "")
                journal = cc.split(".")[0] if "." in cc else cc
            if not journal:
                journal = "未知出处"

            # === 4. 提取年份 ===
            year = ref.get("year", "未知年份")

            # === 5. 提取摘要/核心内容 ===
            # 论文通常是 summary，指南通常是 nrjs (内容介绍)
            summary = ref.get("summary", "")
            if not summary:
                summary = ref.get("nrjs", "缺少具体摘要内容，请结合标题及临床经验推断。")

            # === 6. 组装：给大模型阅读的【资料 X】 ===
            material_text = f"【资料 {ref_index}】\n文献标题：{title}\n文献出处：{journal} ({year})\n核心内容：{summary}\n"
            formatted_materials.append(material_text)

            # === 7. 组装：文末的 [X] 参考文献 ===
            # 格式：[序号] 作者.标题[J]. 期刊,年份
            ref_str = f"[{ref_index}] {author}.{title}[J]. {journal},{year}"
            formatted_references.append(ref_str)

            # 递增序号
            ref_index += 1

    # 将数组拼接为带有换行符的字符串
    materials_text = "\n".join(formatted_materials)
    references_text = "\n".join(formatted_references)

    return materials_text, references_text

async def run_wechat_mcp_example(articles):
    """
    批量发布微信公众号文章（最多8篇）
    :param articles: 文章列表，每个元素包含 {title, content, image_paths}
    """
    # 1. 配置 MCP Server 启动参数
    server_script_path = os.path.abspath("/root/Data/GLS/wechat-publisher-mcp/src/server.js")

    server_params = StdioServerParameters(
        command="node",
        args=[server_script_path],
        env=None
    )

    print(f"正在启动并连接到 MCP Server...")

    # 2. 建立 stdio 连接
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 3. 初始化 MCP 会话
            await session.initialize()
            print("✅ MCP 会话初始化成功！")

            # 4. 调用发布文章工具
            print("\n正在尝试调用 'wechat_publish_article'...")

            try:
                # 遍历文章列表（最多8篇）
                for idx, article in enumerate(articles[:8]):
                    title = article.get("title", "")
                    content = article.get("content", "")
                    image_paths = article.get("image_paths", [])

                    # 构建参数
                    arguments = {
                        "title": title,
                        "content": content,
                        "author": "灵犀量子 AI",
                        "appId": WECHAT_APP_ID,
                        "appSecret": WECHAT_APP_SECRET,
                        "previewMode": False,
                    }

                    # 添加图片
                    if image_paths:
                        if isinstance(image_paths, list):
                            arguments["imagePaths"] = image_paths
                        else:
                            arguments["coverImagePath"] = image_paths
                    else:
                        arguments["coverImagePath"] = ""

                    # 发布
                    print(f"\n正在发布第 {idx + 1} 篇文章: {title}")
                    result = await session.call_tool(
                        "wechat_publish_article",
                        arguments=arguments
                    )

                    # 打印结果
                    for content_item in result.content:
                        if content_item.type == "text":
                            print(f"服务器返回: {content_item.text}")

                # 群发
                # result = await session.call_tool(
                #     "wechat_mass_send",
                #     arguments={
                #         "title": "调用MCP测试文章",
                #         "content": "# 测试\n\n这是一条测试消息，无事发生。",
                #         "author": "灵犀量子 AI",
                #         "appId": WECHAT_APP_ID,
                #         "appSecret": WECHAT_APP_SECRET,
                #         "isToAll": True,          # True=群发给所有粉丝
                #         "sendIgnoreReprint": 0    # 0=允许转载
                #     }
                # )


                # 打印返回结果
                for content in result.content:
                    if content.type == "text":
                        print(f"\n服务器返回: {content.text}")
            except Exception as e:
                print(f"\n❌ 调用失败: {e}")

# (此处省略上文中的具体审核函数代码以节省篇幅，请将第二套代码中的 5 个审核函数原封不动放在这里)

class AgenticPipeline:
    def __init__(self):
        self.registry_file = "issue_registry_tree.json"
        self.load_registry()

    def load_registry(self):
        """初始化 MemBrain 记忆树结构"""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                self.registry = json.load(f)
        else:
            # 初始化标准的自适应实体树结构
            self.registry = {
                "fixed_issues": ["时效性问题", "标题与内容断裂"],
                "memory_tree": {
                    "质量打分低": {
                        "通用": ["结论必须包含具体的量化指标（如HR、有效率等）", "禁止使用营销化废话，要求学术客观"]
                    },
                    "标题与内容断裂": {
                        "通用": ["结论研究对象范围不能小于标题限定范围"]
                    },
                    "事实与临床逻辑错误": {
                        "通用": []
                    }
                }
            }

    def save_registry(self):
        lock_path = f"{self.registry_file}.lock"
        with FileLock(lock_path, timeout=10):
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(self.registry, f, ensure_ascii=False, indent=4)

    def retrieve_local_memories(self, zone_name):
        """
        【渐进式检索】：只拿取“通用”规则以及“当前特定专区”的规则，避免上下文稀释。
        """
        local_rules = []
        tree = self.registry.get("memory_tree", {})

        for error_type, zone_dict in tree.items():
            if "通用" in zone_dict:
                local_rules.extend(zone_dict["通用"])
            if zone_name and zone_name in zone_dict:
                local_rules.extend(zone_dict[zone_name])
        # 去重并返回
        return list(set(local_rules))

    def check_local_rules(self, title, info, local_rules):
        """
        利用大模型一次性检查当前文本是否违背了本专区的任何一条避坑指南
        """
        rules_str = "\n".join([f"{i + 1}. {r}" for i, r in enumerate(local_rules)])
        sys_prompt = "你是顶级医学文本质检官。请判断目标结论是否违反了给定的【专属避坑规则】中的任意一条。"
        user_prompt = f"""
        文章标题: {title}
        核心结论: {info}

        专属避坑规则：
        {rules_str}

        请严格判断上述结论是否违反了其中的规则？
        返回JSON格式：{{"has_issue": boolean, "reason": "如果违反，说明具体违反哪条规则及表现；如果没有，请填无"}}
        """
        res = call_qwen(sys_prompt, user_prompt)
        return extract_json_from_text(res)

    def evolve_memory_tree(self, failed_data, zone_name):
        """
        基于被淘汰的错题，自动向树状知识库中补充针对本专区的精准规则
        """
        if not failed_data: return

        samples = "\n".join([f"标题:{d['title']}\n缺陷原因:{d.get('score_reason', '')}" for d in failed_data[:5]])
        sys_prompt = "你是临床循证规则提炼专家。请根据以下审核不合格的医学文案，总结出 1-3 条专属于该医学领域的强制性避坑规则。"

        user_prompt = f"""
        当前医学专区：【{zone_name}】
        以下是该专区内被扣分或拦截的错题样本：
        {samples}

        请提取出针对该专区的核心防错规则。
        分类只能是："质量打分低"、"标题与内容断裂"、"事实与临床逻辑错误"。

        要求输出JSON数组格式：
        {{"new_rules": [ {{"error_type": "质量打分低", "rule": "具体规则..."}} ]}}
        """
        res = call_qwen(sys_prompt, user_prompt)
        parsed = extract_json_from_text(res)
        new_rules = parsed.get("new_rules", [])

        tree = self.registry["memory_tree"]
        for item in new_rules:
            etype = item.get("error_type", "事实与临床逻辑错误")
            rule = item.get("rule", "").strip()
            if not rule: continue

            if etype not in tree:
                tree[etype] = {"通用": []}
            if zone_name not in tree[etype]:
                tree[etype][zone_name] = []

            # 避免重复插入相同规则
            existing_rules = tree[etype][zone_name]
            if not any(rule in ex or ex in rule for ex in existing_rules):
                tree[etype][zone_name].append(rule)

        self.save_registry()
        logger.info(f"专区【{zone_name}】记忆树已自我进化，新增 {len(new_rules)} 条规则。")

    def rewrite_info_with_feedback(self, title, original_info, feedback_reasons, local_rules):
        """
        【Reflexion 核心】：根据意见和规则重写文案
        """
        sys_prompt = "你是一个顶级的医学内容修复专家。你的任务是根据质检反馈，彻底修复重写医学结论。"
        avoid_rules = "\n".join([f"- {r}" for r in local_rules]) if local_rules else "无"

        user_prompt = f"""
        【任务说明】：以下医学初稿未通过质检。请根据反馈意见彻底重写【核心结论】。

        【文章标题】：{title}
        【原始结论】：{original_info}

        【必须解决的反馈意见】：
        {feedback_reasons}

        【本专区硬性避坑指南（绝对不能犯）】：
        {avoid_rules}

        【重写要求】：
        1. 直接修复反馈中提到的缺陷（如增加量化数据、拉回跑题逻辑）。
        2. 仅输出重写后的纯文本内容，不要包含 Markdown 代码块或废话解析。
        """
        return call_qwen(sys_prompt, user_prompt)


    def summarize_top3_issues(self, failed_data):
        """Step 2: 对低分数据总结前3个新问题"""
        if not failed_data: return []

        # 将低分数据的评价抽取出来给大模型
        samples = "\n".join(
            [f"标题:{d['title']}\n结论:{d['core_conclusion']}\n扣分原因:{d.get('score_reason', '')}" for d in
             failed_data[:10]])

        sys_prompt = "你是医学质检分析师，精通循证医学、肿瘤学（特别是临床试验数据）以及大模型生成文本的缺陷分析。根据以下被扣分的样本，总结出当前最常出现的3种具体的文案或逻辑错误。"
        user_prompt = f"""
        请忽略“时效性问题”、“标题与内容断裂”、“细节幻觉与数据捏造”这三个固定问题。

        # Constraints (严格遵守)
        1. **排除时效性**：绝对不要将“时效性（内容是否为最新数据/指南）”作为问题提出或统计。
        2. **基于事实**：所有提出的问题必须从我提供的数据中得出，不能凭空捏造。
        3. **医学严谨性**：以临床医学的专业标准来审视数据（如：HR值、OS/PFS、试验代号、用药方案、适用人群等关键信息）。

        提取出3个全新的、最频发的问题标签（短语形式，如“缺少关键生存期数据”、“对照组信息缺失”）。
        请严格返回JSON格式：{{"top3_issues": ["问题1", "问题2", "问题3"]}}

        样本数据：
        {samples}
        """
        res = call_qwen(sys_prompt, user_prompt)
        parsed = extract_json_from_text(res)
        return parsed.get("top3_issues", [])

    def evolve_rules(self, top3_new_issues):
        """Step 3: 动态规则库更新 (融合去重并添加新问题，累计问题频次)"""
        current_dynamic = self.registry["dynamic_issues"]
        active_dynamic = current_dynamic.copy()

        # 确保 dynamic_issue_counts 字段存在
        if "dynamic_issue_counts" not in self.registry:
            self.registry["dynamic_issue_counts"] = {}

        # 融合: 检查新提出的问题是否和现有问题重复
        existing_all = self.registry["fixed_issues"] + active_dynamic

        for new_issue in top3_new_issues:
            sys_prompt = "你是一个语义查重专家。判断新问题是否与已有问题列表中的某个问题语义重复。"
            user_prompt = f"""
            已有问题列表：{existing_all}
            新问题：{new_issue}
            如果重复，返回 {{"is_duplicate": true, "matched_issue": "从已有问题列表中原文复制匹配的问题"}}；
            如果不重复（是新问题），返回 {{"is_duplicate": false, "matched_issue": ""}}
            仅返回JSON。
            """
            res = call_qwen(sys_prompt, user_prompt)
            parsed = extract_json_from_text(res)

            if parsed.get("is_duplicate", True):
                # 相同问题：在动态问题中找到匹配项并累加计数
                matched = parsed.get("matched_issue", "").strip()
                target_issue = None
                for issue in active_dynamic:
                    if issue.strip() == matched or matched in issue or issue in matched:
                        target_issue = issue
                        break
                if target_issue:
                    self.registry["dynamic_issue_counts"][target_issue] = \
                        self.registry["dynamic_issue_counts"].get(target_issue, 0) + 1
            else:
                # 新问题：添加到动态规则库，初始计数为1
                active_dynamic.append(new_issue)
                self.registry["dynamic_issue_counts"][new_issue] = \
                    self.registry["dynamic_issue_counts"].get(new_issue, 0) + 1

        self.registry["dynamic_issues"] = active_dynamic
        self.save_registry()

    def generate_analysis(self, passed_count, failed_count, issue_counts):
        total = passed_count + failed_count
        pass_rate = (passed_count / total * 100) if total > 0 else 0
        sys_prompt = "你是一个系统审计分析师，根据本轮统计数据生成一份简短的Markdown分析报告，指导后续模型优化方向。"
        user_prompt = f"本轮审核数据总计{total}条，合格{passed_count}条，合格率{pass_rate:.1f}%。\n各问题触发次数：{json.dumps(issue_counts, ensure_ascii=False)}\n请输出简要的结论和Prompt优化建议。"
        report = call_qwen(sys_prompt, user_prompt)

        with open("system_analysis.md", "a", encoding="utf-8") as f:
            # 修复原代码中 os.times()[4] 导致跨平台报错的问题
            f.write(f"\n## {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} 审计报告\n")
            f.write(f"- **总计处理**: {total}条 (合格率 {pass_rate:.1f}%)\n")
            f.write(report)
            f.write("\n---\n")

    def cleaningInfo(self, data) -> str:
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                # 精准捕获 JSON 解析错误。说明这是纯文本，直接返回即可
                # logger.debug(f"文本非 JSON 格式，按纯文本返回: {e}")
                return data
            except Exception as e:
                # 捕获其他意料之外的错误并打印，防止错误被吞没
                logger.error(f"cleaningInfo 发生未知异常: {e}")
                return data
        parts = []
        if isinstance(data, dict):
            if data.get("coreConclusion"):
                for item in data["coreConclusion"]:
                    parts.append(f"- {item.get('content', '')}")
        return "\n".join(parts) if parts else str(data)

    # def cleaningInfo(self, data: dict) -> str:
    #     parts = []
    #
    #     if isinstance(data, str):  # 关键判断
    #         data = json.loads(data)
    #
    #     # 引导问题
    #     if data.get("guideQuestion"):
    #         q = data["guideQuestion"]
    #         parts.append("【引导性问题】")
    #         parts.append(q.get("question", ""))
    #         parts.append("")
    #
    #     # 核心结论
    #     if data.get("coreConclusion"):
    #         parts.append("【核心结论】")
    #         for item in data["coreConclusion"]:
    #             parts.append(f"- {item.get('content', '')}")
    #         parts.append("")
    #
    #     # 指南推荐
    #     if data.get("guide"):
    #         parts.append("【指南推荐】")
    #         for item in data["guide"]:
    #             parts.append(f"- {item.get('content', '')}")
    #         parts.append("")
    #
    #     # 临床要点
    #     if data.get("clinicalPoint"):
    #         parts.append("【临床要点】")
    #         for group in data["clinicalPoint"]:
    #             subtitle = group.get("subtitle", "")
    #             if subtitle:
    #                 parts.append(f"### {subtitle}")
    #             for item in group.get("item", []):
    #                 parts.append(f"- {item.get('content', '')}")
    #         parts.append("")
    #
    #     # 实践要点
    #     if data.get("practicalPoint"):
    #         parts.append("【实践要点】")
    #         for group in data["practicalPoint"]:
    #             subtitle = group.get("subtitle", "")
    #             if subtitle:
    #                 parts.append(f"### {subtitle}")
    #             for item in group.get("item", []):
    #                 parts.append(f"- {item.get('content', '')}")
    #         parts.append("")
    #
    #     return "\n".join(parts)

    def cleaningInfoGuide(self, data: dict) -> str:
        parts = []
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                # 精准捕获
                return data
            except Exception as e:
                logger.error(f"cleaningInfoGuide 发生未知异常: {e}")
                return data

        # 引导问题
        if data.get("guideQuestion"):
            q = data["guideQuestion"]
            parts.append(q.get("question", ""))
            parts.append("")

        # # 核心结论
        # if data.get("coreConclusion"):
        #     for item in data["coreConclusion"]:
        #         parts.append(f"- {item.get('content', '')}")
        #     parts.append("")

        return "\n".join(parts)


    def run_round(self, data_list, title, batch_id, materials, zone_name="通用医学专区"):
        """
        运行带有【反思重试环】和【动态树记忆】的迭代流水线
        """
        print(f"\n========== 开始新一轮迭代，专区：【{zone_name}】，共 {len(data_list)} 条数据 ==========")

        passed_data = []
        failed_data = []
        MAX_RETRIES = 1  # 每条数据最多抢救2次

        # 1. 精准提取当前专区的避坑规则
        local_rules = self.retrieve_local_memories(zone_name)
        pass_score_threshold = 70 if (not batch_id or batch_id == "GLOBAL_ALL") else 80

        for item in data_list:
            current_title = item["title"]
            retry_count = 0
            is_passed = False

            # 使用副本进行内部处理
            current_info = self.cleaningInfo(item.get("info", ""))

            # ======= 进入 反思抢救循环 =======
            while retry_count <= MAX_RETRIES:
                print(f"\n[尝试 {retry_count + 1}/{MAX_RETRIES + 1}] 审核: {current_title}")
                feedback_reasons = []
                has_fatal_issue = False

                # 1. 检查时效性 (仅全库挖掘时做)
                if not batch_id or batch_id == "GLOBAL_ALL":
                    r1 = check_timeliness(current_title, current_info)
                    if r1.get("has_issue"):
                        has_fatal_issue = True
                        feedback_reasons.append(f"【时效性硬伤】: {r1.get('reason')}")

                # 2. 检查断裂
                r2 = check_content_interruption(current_title, current_info)
                if r2.get("has_issue"):
                    has_fatal_issue = True
                    feedback_reasons.append(f"【标题内容断裂】: {r2.get('reason')}")

                # 3. 本专区记忆树拦截
                if local_rules:
                    rd = self.check_local_rules(current_title, current_info, local_rules)
                    if rd.get("has_issue"):
                        has_fatal_issue = True
                        feedback_reasons.append(f"【违背专区硬规】: {rd.get('reason')}")

                # 4. 整体质量打分
                score_res = score_data(current_title, current_info)
                score = score_res.get("score", 0)
                score_reason = score_res.get("reasoning", "")

                if score < pass_score_threshold:
                    feedback_reasons.append(f"【得分过低({score}分)】: {score_reason}")

                # --- 综合判定 ---
                if not has_fatal_issue and score >= pass_score_threshold:
                    print(f"✅ 审核通过！质量得分: {score}")
                    is_passed = True
                    item["score"] = score
                    item["score_reason"] = score_reason
                    item["info"] = current_info  # 保存最终成果
                    break
                else:
                    print(f"❌ 审核失败。发现 {len(feedback_reasons)} 个问题。")
                    if retry_count < MAX_RETRIES:
                        print(">> 启动 Reflexion 反思引擎，大模型正在自我重写修复...")
                        feedback_str = "\n".join(feedback_reasons)
                        # 让大模型重写并覆盖 current_info 进入下一轮循环
                        current_info = self.rewrite_info_with_feedback(current_title, current_info, feedback_str,
                                                                       local_rules)

                    retry_count += 1

            # 记录数据去向
            if is_passed:
                passed_data.append(item)
            else:
                item["score_reason"] = "经过多次重试依然不合格: " + " | ".join(feedback_reasons)
                failed_data.append(item)

        # ================== 阶段：总结错题，进化记忆树 ==================
        if failed_data:
            print(">> 触发动态规则树进化...")
            self.evolve_memory_tree(failed_data, zone_name)

            # 🌟 修改点 1：废弃数据状态更新
            conn = get_db_connection()
            try:
                cursor = conn.cursor()
                for item in failed_data:
                    if item.get("id"):
                        cursor.execute("UPDATE evidence_card SET status = 0 WHERE id = %s", (item["id"],))
                conn.commit()
            except Exception as e:
                conn.rollback()
            finally:
                cursor.close()
                conn.close()

        # ================== 阶段：最终制卡 (含局部记忆树规避) ==================
        if passed_data:
            # 制卡时同样注入专区避坑规则，防患未然
            dynamic_bans = "\n".join([f"- {rule}" for rule in local_rules]) if local_rules else "无"

            for item in passed_data:
                pass_title = item["title"]
                pass_info = item["info"]

                if batch_id and item.get("id"):
                    # 🌟 修改点 2：批次 ID 更新
                    conn = get_db_connection()
                    try:
                        cursor = conn.cursor()
                        cursor.execute("UPDATE evidence_card SET batch_id = %s WHERE id = %s", (batch_id, item["id"],))
                        conn.commit()
                    except Exception as e:
                        pass
                    finally:
                        cursor.close()
                        conn.close()

                print(f">> 正在生成高分专区卡片: {pass_title}")
                formatted_materials = f"{materials}\n" if materials else ""
                reference_list = ""

                system_prompt = '你现在是“医学证据卡片选题总编 + 临床知识编辑”。'
                user_prompt = f"""
                你的任务不是写一篇普通医学科普，也不是机械整理资料，而是基于我提供的全部材料、选定的题目，为医生生产一张“可读、可用、可检索、可沉淀为个人知识资产”的单张证据卡片。

                本次选定的卡片题目是：【 {pass_title} 】

                你的最高目标有五个：
                1. 本卡片必须服务于一个明确的临床决策，而不是泛泛介绍疾病。
                2. 本卡片必须同时吸收：医生资料（若有）、思维导图/决策树（若有）、医案（若有）、书籍（若有）、指南、文献、联网搜索和学术搜索结果。
                3. 本卡片必须有明显循证属性，尤其是量化指标、结局指标、分层条件、适用人群、证据等级或证据强弱。
                4. 不能只是把症状、证型、方药替换一下，必须体现深度。
                5. 产出不仅要能给医生每天阅读，也要能沉淀为医生自己的知识资产，未来可外挂给大模型用于垂类问答与临床辅助决策。

                一、必须使用的资料范围与使用原则
                你必须优先并显式整合以下来源，但当上传材料与外部证据不一致时，必须明确区分“医生经验逻辑”“资料原文观点”“外部循证结论”，不能混写。不得只根据我上传的一份资料直接成文，必须交叉验证。
                我已将资料资料筛选出精华片段，如下：
                {materials_part}{pass_info}

                1. 决策树 / 思维导图：提取风险因素分层、症状识别路径、辅助检查路径、证候判定条件、治法选择逻辑、对症加减规则。这决定“临床推理顺序”，不是单纯摘抄内容。
                2. 医案与书籍：提取核心病机、主要/兼夹病机、理法方药对应、加减思路、剂量逻辑、动态调整、误治漏治转折点。这决定“为什么这样治”，不是只提方名药名。
                3. 指南与文献：作为“校准器”和“证据增强器”。提取推荐意见、适应证/不适用人群、证据等级、结局指标、安全性信息。

                二、联网搜索与学术搜索的硬要求
                每次写卡片前，你必须做外部检索，并完成以下动作：
                1. 搜最新指南和共识（国内外相关学会、近3年更新、规范等）。
                2. 搜近5年高质量论文（系统综述、Meta分析、RCT、高质量队列研究、真实世界研究）。
                3. 搜争议点（主动找与医生经验相吻合/不同的证据，尚无定论的问题）。
                4. 搜相似症状或路径的鉴别文献（主诉、检查异常、关键病机、相关风险因素）。
                5. 检索后做证据分层（区分为强证据、中等证据、弱证据、经验性信息）。

                三、证据卡片的制作原则
                每张证据卡片必须满足以下要求：
                1. 只解决一个核心决策：一张卡片只回答一个核心临床决策，不要贪多。
                2. 读者是医生，不是患者：语言必须书面、学术、克制。不要出现过多解释性废话，不要反复讲常识。
                3. 不写“临床问题”字段：正文里不要单独出现“临床问题”这个小标题。
                4. 核心结论要像结论：不要像病历复述。不要写“根据患者症状和病史……”这类开头。应直接进入判断、结论和建议。
                5. 必须包含鉴别诊断或鉴别分析：至少回答为什么不是另外几种常见路径、哪些信息最能拉开鉴别、哪些情况下治疗思路会改变。
                6. 必须体现动态决策：不能写成静态百科。要写清初始判断、关键分叉、加减触发条件、疗效不佳时的调整方向、复查与再判断节点。
                7. 必须有强循证属性：不能只说“研究显示有效”。必须尽量写出样本量、研究类型、随访时长、主要终点、相对/绝对风险、症状/胃镜/病理评分变化、安全性指标。如果原始文献未提供完整指标，必须如实写“未报告”。
                8. 必须区分证据来源：正文中要尽量分清医生经验/医案启发、决策树规则、指南建议、文献数据、学术推断。不能把不同层级证据混成一段话。
                9. 必须可供大模型外挂调用：段落职责清晰、结论明确、条件边界明确、可提取关键词明确、便于转化为问答知识单元。

                四、证据卡片的推荐结构（控制在 1000—2000 字，不含引用文献列表）
                建议结构如下，可微调但不要僵化：
                1. 标题：陈述句或判断式，有临床意味，能体现决策价值。
                2. 核心结论：用 3—5 句话概括该卡片支持的核心判断、关键分层点、主要处理原则、一句话提示适用边界。
                3. 决策依据：来自医生资料的核心病机逻辑、决策树的分叉依据、检查病理的支持要点、文献指南的校准信息。
                4. 鉴别分析（必写）：最容易混淆的 2—4 条路径、区分依据、误判后治疗风险。
                5. 治疗与处置建议：治法、选方/药群/加减方向、哪些情况下加强清热/偏向养阴/活血通络/先处理风险异常，体现“路径感”。
                6. 详细数据支持（重点，必须量化）：
                   - 先写研究类型（指南、Meta、RCT、队列、决策树经验等）
                   - 再写研究对象（人群、证候、主诉等）
                   - 再写关键指标（症状评分变化、胃镜/病理/Hp评分变化、复发率、不良事件、结局指标等。若中医文献写“总有效率”，需说明判定标准和局限性）
                   - 再写证据解释（指标改善意味着什么、是否改变长期风险、样本量偏倚等）
                7. 随访与调护：哪些患者需密切复查、生活方式关键干预点、情志饮食作息对决策的影响、何种情况需重新辨证。
                8. 证据边界与未解决问题：当前证据不足之处、经验与证据未完全对齐之处、哪些结论更适合“辅助决策”。
                9. 正文文献角标（核心强制要求）：在正文（尤其是“核心结论”、“决策依据”和“详细数据支持”模块）中，每当你陈述了一个具体的指南推荐、文献数据或研究结论时，必须在句子末尾直接标出对应的参考文献角标（例如：[1]、[2]、[1][3]）。
                10. 参考文献：必须在文章最末尾单独列出“参考文献”模块。严格按照标准格式输出所有引用的文献，编号必须与正文中的角标严格一一对应！
                   格式示例：
                   [1] 顾燕(综述);曾燕(审校).肝癌磁共振分子影像诊断的研究进展[J]. 重庆医学,2014
                   [2] 袁惊雷;谢晓桐;张佩娜;马立恒.基于CT和MRI影像组学的机器学习模型预测肝癌早期复发的研究进展[J]. 磁共振成像,2022
                   [3] 秦建民;顾新刚.超声造影成像技术在肝癌早期诊断与治疗中应用价值[J]. 肝胆外科杂志,2015(0)

                五、输出前自检清单
                在输出最终证据卡片前，你必须自检并确保：
                1. 这张卡片是否真的只服务一个核心决策？
                2. 标题是否有阅读吸引力，又不流于营销化？
                3. 是否已经显式使用上传资料，而不是只做外部综述？
                4. 是否真正吸收了决策树逻辑和核心病机？
                5. 是否有鉴别诊断或鉴别分析？
                6. 是否有明确的量化指标并说明了证据边界？
                7. 是否适合未来被结构化沉淀为医生知识资产？
                8. 正文中是否准确标注了文献引用角标（如 [1]、[2]）？
                9. 文末的“参考文献”编号是否与正文的角标完全对应，且格式标准？

                【六、特别禁令与专属避坑】
                绝对禁止以下写法：
                - 机械模板化，用空话代替结论。
                - 把医案直接改写成科普，把指南内容堆砌成综述。

                【当前专区历史踩坑教训（绝对避免）】：
                {dynamic_bans}
                """

                output_dir = os.path.join("card_md", f"evidence_cards_{pass_title}")
                os.makedirs(output_dir, exist_ok=True)

                messages = [{"role": "user", "content": user_prompt}]
                card_res = requestQwenMultiTurn(system_prompt, messages)

                safe_title = re.sub(r'[\\/:*?"<>|]', '_', pass_title)[:40]
                with open(os.path.join(output_dir, f"_{safe_title}.md"), "w", encoding="utf-8") as f:
                    f.write(card_res)


        return True

