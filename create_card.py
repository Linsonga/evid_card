# -*- coding: utf-8 -*-
# @Time    : 25/03/2026
# @File    : create_card.py.py
# @Software: PyCharm

"""
create_card.py - Description of the file/module
"""

from openai import OpenAI
from pymilvus import MilvusClient
import requests
from urllib.parse import quote
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import normalize

from sklearn.metrics import pairwise_distances
from datetime import datetime
import os
import re


# # 测试
# client = MilvusClient(
#     uri="https://in03-7fcec72ace350c2.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn",
#     token="5a0ff0669f3e06665dd5761d0d8acb3e57571142dfe6bbebe3d652fa6be3d3a4a9e02e119818744cce729692e9f8cdf457ccfaf2"
# )
# bigdata库
# 正式连接客户端
milvus_client = MilvusClient(
    uri="https://in01-f141153d490dbb6.ali-cn-beijing.vectordb.zilliz.com.cn:19530",
    token="3ff40260e78eeaa2320a3da0f8c7a58244bf64347fd3c54ac7bbc3fbd18a952f31463b6ad1b7f6eb9f39527dcb1b6a17d3f8d402"
)



def requestQwencontentInterruption(system_prompt, user_prompt):
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-bb48765d7ed540d49639b1bd5e4dd82b",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        # model="qwen3-235b-a22b-instruct-2507",
        model="qwen3.5-plus-2026-02-15",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        extra_body={
            "enable_thinking": True,
            "enable_search": True
        }
    )
    return completion.choices[0].message.content


def requestQwenMultiTurn(system_prompt, messages):
    """支持多轮对话，messages 为完整的对话历史列表"""
    client = OpenAI(
        api_key="sk-bb48765d7ed540d49639b1bd5e4dd82b",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen3.5-plus-2026-02-15",
        messages=[{"role": "system", "content": system_prompt}] + messages,
        extra_body={
            "enable_thinking": True,
            "enable_search": True
        }
    )
    return completion.choices[0].message.content



def vector_4b(text):
    """获取向量"""
    url = "https://u48781-8e4f-c6691756.westx.seetacloud.com:8443/embedding"
    params = {"user_input": text}
    response = requests.get(url, params=params, verify=False)
    vector_str = response.text.strip()
    return [float(x) for x in vector_str.split(',')]


def main(materials):


    system_peompt = '你现在是“医学证据卡片选题总编 + 临床知识编辑 + 循证医学研究员 + 医生知识资产工程师”。'
    user_prompt = f"""
    你的任务不是写一篇普通医学科普，也不是机械整理资料，而是基于我提供的全部材料，为医生持续生产“可读、可用、可检索、可沉淀为个人知识资产”的证据卡片。
    你的工作范围覆盖两部分：第一部分：证据卡片选题设计,第二部分：单张证据卡片制作。

    你的最高目标有五个：
    1. 每张卡片必须服务于一个明确的临床决策，而不是泛泛介绍疾病。
    2. 每张卡片必须同时吸收：医生资料（若有）、思维导图/决策树（若有）、医案（若有）、书籍（若有）、指南、文献、联网搜索和学术搜索结果。
    3. 每张卡片必须有明显循证属性，尤其是量化指标、结局指标、分层条件、适用人群、证据等级或证据强弱。
    4. 卡片之间必须避免千篇一律，不能只是把症状、证型、方药替换一下。
    5. 产出不仅要能给医生每天阅读，也要能沉淀为医生自己的知识资产，未来可外挂给大模型用于垂类问答与临床辅助决策。

    一、必须使用的资料范围
    你必须优先并显式整合以下来源：
    A. 我上传的医生资料（若有）包括但不限于：
    - 决策树/思维导图
    - 医案
    - 书籍
    - 证据卡片示例
    - 研究资料/指南整理稿/症状评分表/胃镜病理评分表
    我已将医生资料资料筛选出精华片段，如下：
    {materials}
    
    B. 联网搜索
    你必须检索最新公开资料，尤其包括：
    - 最新国内外指南
    - 专家共识
    - 系统综述
    - Meta分析
    - RCT
    - 高质量观察性研究
    - 真实世界研究
    - 疾病管理规范
    - 近3-5年高质量综述
    
    C. 学术搜索
    优先检索：
    - PubMed
    - Google Scholar
    - Cochrane
    - 国内核心数据库可公开获得部分
    - 指南发布机构官网
    - 学协会官网
    请注意：
    - 上传资料不是“背景材料”，而是你的一级知识源。
    - 联网搜索与学术搜索不是可选项，而是必须补充的二级知识源。
    - 当上传材料与外部证据不一致时，必须明确区分“医生经验逻辑”“资料原文观点”“外部循证结论”，不能混写。
    - 不得只根据我上传的一份资料直接成文，必须交叉验证。

    二、你对上传资料的使用原则
    1. 决策树 / 思维导图的作用
       把它视为“医生显性化诊疗逻辑”。
       你必须提取：
    - 风险因素分层
    - 症状识别路径
    - 辅助检查路径
    - 证候判定条件
    - 治法选择逻辑
    - 对症加减规则
    - 生活方式与调护建议
      这一层决定“临床推理顺序”，不是单纯摘抄内容。
    
    2. 医案与书籍的作用
       把它们视为“医生隐性经验的文本化载体”。
       你必须提取：
    
    - 核心病机
    - 主要病机与兼夹病机
    - 理法方药对应关系
    - 加减思路
    - 剂量逻辑
    - 病程演变中的动态调整
    - 误治、漏治、疗效转折点
      这一层决定“为什么这样治”，不是只提方名药名。
    
    3. 指南与文献的作用
       把它们视为“校准器”和“证据增强器”。
       你必须提取：
    
    - 推荐意见
    - 适应证 / 不适用人群
    - 证据等级
    - 主要结局指标
    - 安全性信息
    - 与医生经验一致和不一致之处
      这一层决定“卡片能不能站得住”。
    
    4. 既有证据卡片样例的作用
       只可借用：
    
    - 成品感
    - 医生阅读友好度
    - 栏目意识
      不可照搬：
    - 标题句式
    - 固定段落套路
    - 空泛表达
    - 无指标的数据支持
    - 弱鉴别诊断
    - 模板化写法
    
    三、选题机制：先选题，再写卡片
    在正式写证据卡片前，你必须先做“选题推演”，不能直接写。
    每次选题时，先在我上传资料、联网结果和学术搜索结果中提取选题池，然后按照以下九类角度生成候选题目：
    
    1. 决策冲突型:围绕医生最容易犹豫的点来选题。
       例如：
    - 什么时候以清热化湿为先，什么时候先健脾和胃
    - 某症状背后究竟更偏气滞、湿热、阴虚还是瘀阻
    - 同类表现如何区分不同治法路径
    
    2. 鉴别诊断型:围绕“看起来像，但处理不同”的问题来选题。
       重点写：
    - 相似主诉
    - 关键分叉点
    - 舌脉/胃镜/病理/病程/诱因的区别
    - 误判的临床后果
    
    3. 病机拆解型:围绕一个复杂病机或关键病机来选题。
       重点写：
    - 核心病机是什么
    - 当前主要病机是什么
    - 为什么不能只盯一个证型
    - 病机转化后治法如何变化
    
    4. 症状切入口型:围绕临床高频、但背后病机不单一的症状来选题。
       例如：
    - 烧心
    - 胃脘胀满
    - 胃脘刺痛
    - 嗳气
    - 咽堵
    - 纳呆
    - 背沉背痛
      但必须写出“同一症状，不同处理路径”。
    
    5. 检查结果驱动型:围绕胃镜、病理、幽门螺杆菌、胆汁反流、萎缩、肠化、异型增生等结果来选题。
       重点写：
    - 某检查发现如何改变辨证权重
    - 某病理结果如何改变随访和干预优先级
    - 某结果出现后治法是否应升级或调整
    
    6. 用药决策型:围绕“什么情况下加什么、减什么、避什么”来选题。
       重点写：
    - 加减触发条件
    - 药物配伍逻辑
    - 与基础方的关系
    - 剂量侧重
    - 安全边界
      不能只罗列药物。
    
    7. 预后与随访型:围绕“什么时候需要更密切随访、什么时候可观察、什么指标提示风险上升”来选题。
       重点写：
    - 病程风险
    - 复查触发条件
    - 干预有效性的判定指标
    
    8. 生活方式干预型:围绕饮食、情志、作息、分餐、戒烟、低盐、劳逸等内容来选题。
       但必须把它做成“能影响决策质量的内容”，而不是泛泛养生建议。
       
    9. 经验-证据碰撞型:围绕医生经验与现代循证之间：
    - 一致的地方
    - 不一致的地方
    - 互补的地方
      来选题。
      这类题目最容易形成差异化内容和讨论价值。
    
    四、选题去同质化规则
    你必须把“避免千篇一律”当作硬约束。
    每个专区内连续生成多张卡片时，禁止出现以下情况：
    
    - 连续多张都用同一题目句式
    - 连续多张都以“某某证治疗方案”命名
    - 连续多张都只是在症状、证型、方药上做替换
    - 连续多张都采用同一种切入角度
    - 连续多张的核心段落结构几乎一致
    - 连续多张都只讲治疗，不讲鉴别与边界
    - 连续多张都没有明确的证据指标
    
    你必须主动拉开差异，至少从以下维度错开：
    
    - 题目句式
    - 切入角度
    - 问题层级
    - 病机层级
    - 检查层级
    - 决策层级
    - 证据层级
    - 适用人群层级
    - 写作重心
    
    题目允许的风格应该是多样的，例如：
    
    - 决策判断式
    - 鉴别分析式
    - 证据比较式
    - 病机拆解式
    - 治疗时机式
    - 误区纠偏式
    - 随访管理式
    - 药物加减式
    - 检查结果解读式
    但无论题目风格如何变化，都必须最终落到可执行的临床决策上。
    
    五、证据卡片的制作原则
    每张证据卡片必须满足以下要求：
    1. 只解决一个核心决策
       一张卡片只回答一个核心临床决策，不要贪多。
       可以有延伸信息，但主轴必须单一明确。
    
    2. 读者是医生，不是患者
       语言必须书面、学术、克制，不要写成面向大众的医学科普。
       不要出现过多解释性废话，不要反复讲常识。
    
    3. 不写“临床问题”字段
       正文里不要单独出现“临床问题”这个小标题。
    
    4. 核心结论要像结论，不要像病历复述
       不要写“根据患者症状和病史……”这类开头。
       应直接进入判断、结论和建议。
    
    5. 必须包含鉴别诊断或鉴别分析
       至少回答：
    
    - 为什么不是另外几种常见路径
    - 哪些信息最能拉开鉴别
    - 哪些情况下治疗思路会改变
    
    6. 必须体现动态决策
       不能写成静态百科。
       要写清：- 初始判断、- 关键分叉、- 加减触发条件、- 疗效不佳时的调整方向、- 复查与再判断节点
    
    7. 必须有强循证属性
       不能只说“研究显示有效”。
       必须尽量写出：
    
    - 样本量
    - 研究类型
    - 随访时长
    - 主要终点
    - 相对风险/绝对风险/OR/HR/均值差
    - 症状评分变化
    - 胃镜评分变化
    - 病理分级变化
    - 安全性指标
      如果原始文献未提供完整指标，必须如实写“未报告”。
    
    8. 必须区分证据来源
       正文中要尽量分清：
    - 医生经验/医案启发
    - 决策树给出的路径规则
    - 指南建议
    - 文献数据
    - 学术推断
      不能把不同层级证据混成一段话。
    
    9. 必须可供大模型外挂调用
       卡片写法要便于后续结构化抽取。
       隐含要求是：
    
    - 段落职责清晰
    - 结论明确
    - 条件边界明确
    - 可提取关键词明确
    - 便于转化为问答知识单元
    
    六、证据卡片的推荐结构
    每张卡片控制在 1000—2000 字，不含引用文献列表。
    建议结构如下，但可根据题型微调，不要僵化：
    1. 标题
       要求：
    - 陈述句或判断式，避免口号化
    - 有临床意味
    - 不宜过宽
    - 不宜千篇一律
    - 能体现决策价值，而不是只是“某病最新进展”
    
    2. 核心结论
       用 3—5 句话概括：
    - 该卡片支持的核心判断
    - 关键分层点
    - 主要处理原则
    - 一句话提示适用边界

    3. 决策依据
       这一部分写：
    - 来自医生资料的核心病机逻辑
    - 来自决策树的分叉依据
    - 来自检查或病理的支持要点
    - 来自文献或指南的校准信息
    
    4. 鉴别分析
       必须写。
       至少覆盖：
    - 最容易混淆的 2—4 条路径
    - 区分依据
    - 误判后治疗风险
    
    5. 治疗与处置建议
       写清：
    - 治法
    - 选方 / 药群 / 加减方向
    - 哪些情况下加强清热
    - 哪些情况下偏向养阴
    - 哪些情况下活血通络
    - 哪些情况下先处理风险因素与检查异常
      这里要体现“路径感”。
    
    6. 详细数据支持
       这是重点。
       必须尽量量化。
       优先写：
    - 症状评分前后变化
    - 胃镜评分前后变化
    - 病理评分变化
    - Hp转阴率
    - 复发率
    - 不良事件
    - 有效率
    - 缓解率
    - 主要结局指标
    - 次要结局指标
      如果是中医文献常见的“总有效率”，不能只写总有效率，需同时说明其判定标准和局限性。
    
    7. 随访与调护
       简要写：
    - 哪些患者需要更密切复查
    - 生活方式最关键的干预点
    - 情志、饮食、作息对决策的影响
    - 什么情况下需要重新辨证
    
    8. 证据边界与未解决问题
       必须写出：
    - 当前证据不足之处
    - 医生经验与现代证据尚未完全对齐之处
    - 哪些结论更适合“辅助决策”，不能被写成绝对结论
    
    七、详细数据支持的写法要求
    “详细数据支持”必须尽量做到以下几点：
    1. 先写研究类型
       例如：
    - 指南
    - 系统综述
    - Meta分析
    - RCT
    - 队列研究
    - 病例系列
    - 决策树经验规则
    
    2. 再写研究对象
       例如：
    - 胃癌前病变
    - 慢性萎缩性胃炎伴肠化
    - Hp阳性伴胆汁反流
    - 某证候人群
    - 某症状主导人群
    
    3. 再写关键指标
       例如：
    - 治疗前后症状积分
    - 萎缩评分
    - 肠化评分
    - 异型增生评分
    - 糜烂、胆汁反流、充血评分
    - 不良反应发生率
    - 复发率
    - 依从性
    - 随访结局
    
    4. 再写证据解释
       例如：
    - 指标改善意味着什么
    - 是否真正改变长期风险
    - 是否只能说明短期缓解
    - 是否样本量有限
    - 是否存在偏倚
    
    八、联网搜索与学术搜索的硬要求
    每次写卡片前，你必须做外部检索，并完成以下动作：

    1. 搜最新指南和共识
       至少检索：
    - 国内相关学会/协会指南
    - 国际相关指南
    - 近3年更新版本
    - 与疾病管理、癌前病变、随访、Hp处理、胃镜病理分层相关的规范
    
    2. 搜近5年高质量论文
       优先：
    - 系统综述
    - Meta分析
    - RCT
    - 高质量队列研究
    - 真实世界研究
    
    3. 搜争议点
       主动找：
    - 与医生经验相吻合的证据
    - 与医生经验不同的证据
    - 尚无定论的问题
    
    4. 搜相似症状或路径的鉴别文献
       不要只搜主病名，也要搜：
    - 主诉
    - 检查异常
    - 关键病机
    - 相关风险因素
    
    5. 检索后做证据分层
       把外部信息区分为：
    - 强证据
    - 中等证据
    - 弱证据
    - 经验性信息

    九、输出前自检清单
    在输出最终证据卡片前，你必须自检并确保：
    1. 这张卡片是否真的只服务一个核心决策？
    2. 标题是否有阅读吸引力，又不流于营销化？
    3. 是否已经显式使用上传资料，而不是只做外部综述？
    4. 是否真正吸收了决策树逻辑？
    5. 是否体现了核心病机，而非只套证型？
    6. 是否有鉴别诊断或鉴别分析？
    7. 是否有明确的量化指标？
    8. 是否说明了证据边界？
    9. 是否与同专区其他卡片明显不同？
    10. 是否适合未来被结构化沉淀为医生知识资产？
    
    十、最终输出要求
    输出时分两步：
    第一步：先输出选题清单
    每次先给出一组候选题目，并为每个题目标注：
    - 选题角度
    - 临床决策价值
    - 差异化理由
    - 主要资料来源
    - 是否适合做成高阅读量卡片
    - 是否适合做成高决策价值卡片
    
    十一、特别禁令
    禁止以下写法：
    - 机械模板化
    - 所有标题长得一样
    - 全部写成”某某证治疗方案”
    - 反复使用同一段落结构
    - 没有鉴别分析
    - 没有数据指标
    - 把医案直接改写成科普
    - 把指南内容堆砌成综述
    - 用空话代替结论
    - 用”显著改善””疗效较好”但不给任何指标
    - 只讲中医经验，不做外部校准
    - 只讲外部证据，不体现医生自己的逻辑
    - 在正文中出现任何形如”（见医生资料群组X）””（见群组X）””（参见片段X）””（来源：库内记录ID）”等内部引用标注；所有材料内容必须直接融入正文，禁止以括号注释形式暴露数据来源编号

    请把自己当成一个长期运营“医生学术内容系统”的总编辑，而不是一次性文案写手。
    你输出的不是文章，而是未来可持续更新、可检索调用、可支持大模型问答、可沉淀为医生知识资产的高质量证据卡片。
"""



    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"evidence_cards_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # ── 第一轮：只生成选题清单 ──────────────────────────────────────────
    topic_request = (
        user_prompt
        + "\n\n【本轮任务】请只完成第一步：输出5个证据卡片候选选题清单，每个选题标注角度、决策价值、差异化理由、主要资料来源。暂不写卡片正文，等待我逐一确认后再写。"
        + "\n\n【格式要求】每个选题的标题行必须严格按照以下格式输出，方便程序提取：\n"
        + "【选题1】<标题>\n【选题2】<标题>\n【选题3】<标题>\n【选题4】<标题>\n【选题5】<标题>\n"
        + "标题后可跟详细说明，但标题行本身必须单独一行，以【选题N】开头。"
    )

    messages = [{"role": "user", "content": topic_request}]

    print("=" * 60)
    print("【第一轮】正在生成5个候选选题...")
    print("=" * 60)

    topics_res = requestQwenMultiTurn(system_peompt, messages)
    print(topics_res)

    # 保存选题清单
    topics_file = os.path.join(output_dir, "00_选题清单.md")
    with open(topics_file, "w", encoding="utf-8") as f:
        f.write(topics_res)
    print(f"\n选题清单已保存至: {topics_file}\n")

    # 从第一轮回复中解析5个选题标题
    topic_titles = re.findall(r'【选题\d+】(.+)', topics_res)
    topic_titles = [t.strip() for t in topic_titles]

    if len(topic_titles) != 5:
        print(f"⚠️  警告：解析到 {len(topic_titles)} 个选题标题，预期5个。请检查选题清单格式。")
        print(f"已解析到的标题：{topic_titles}")
        # 用序号兜底，避免流程中断
        while len(topic_titles) < 5:
            topic_titles.append(f"第{len(topic_titles) + 1}个选题")
    else:
        print("已解析到5个选题标题：")
        for idx, title in enumerate(topic_titles, 1):
            print(f"  {idx}. {title}")

    # 将 assistant 回复加入历史
    messages.append({"role": "assistant", "content": topics_res})

    # ── 后续轮次：逐张生成卡片 ────────────────────────────────────────
    for i, title in enumerate(topic_titles, 1):
        card_request = (
            f"请现在根据第{i}个选题【{title}】，输出完整的证据卡片全文。"
            f"严格按照证据卡片制作原则撰写，只输出这一张卡片，不要输出其他选题的卡片。"
        )

        messages.append({"role": "user", "content": card_request})

        print("=" * 60)
        print(f"【第{i + 1}轮】正在生成第 {i} 张证据卡片：{title}")
        print("=" * 60)

        card_res = requestQwenMultiTurn(system_peompt, messages)
        print(card_res)

        # 保存单张卡片（文件名含选题标题，方便识别）
        safe_title = re.sub(r'[\\/:*?"<>|]', '_', title)[:40]
        card_file = os.path.join(output_dir, f"{i:02d}_{safe_title}.md")
        with open(card_file, "w", encoding="utf-8") as f:
            f.write(card_res)
        print(f"\n第 {i} 张卡片已保存至: {card_file}\n")

        # 将当前卡片加入历史，供下一轮参考
        messages.append({"role": "assistant", "content": card_res})

    print(f"\n全部完成！所有文件保存在目录: {output_dir}/")


def is_garbage_text(text):
    """
    判断是否为乱码/无意义文本
    返回 True 表示是垃圾数据，应该丢弃；False 表示正常数据
    """
    # 1. 过滤太短的文本
    if len(text) < 50:
        return True

    # 2. 计算中文字符占比 (假设你的目标语料主要是中文)
    # 匹配所有中文字符
    zh_chars = re.findall(r'[\u4e00-\u9fa5]', text)
    zh_ratio = len(zh_chars) / len(text)
    # 如果中文字符占比低于 30%（可根据实际情况微调），判定为乱码
    if zh_ratio < 0.3:
        return True

    # 3. 计算异常符号的密度
    # 乱码中常出现的符号集合
    weird_symbols = re.findall(r'[#\|\$_\~\^\/\\<>\*\}]', text)
    symbol_ratio = len(weird_symbols) / len(text)
    # 如果特殊符号占比超过 10%，判定为公式乱码或OCR错误
    if symbol_ratio > 0.1:
        return True

    # 4. 计算空格密度
    space_ratio = text.count(' ') / len(text)
    # 如果空格占比超过 25%，通常是 OCR 碎片化导致的
    if space_ratio > 0.25:
        return True

    return False




def milvus():

    # ==========================================
    # 第一步：拉取数据，并增加“数据清洗”逻辑
    # ==========================================
    print("正在从 Milvus 向量库提取数据...")

    results = milvus_client.query(
        collection_name="evidence_card_test",
        filter='batch_id == "c162f24a-096a-4c8f-99cc-5cb2afde358a"',
        output_fields=["vector", "text"],
        limit=2000  # 稍微拉大一点，保证过滤后有足够的数据
    )

    embeddings_list = []
    texts = []
    metadata = []

    for item in results:
        text_content = item["text"].strip()

        # 🌟 优化1：使用增强版的正则特征过滤乱码
        if is_garbage_text(text_content):
            continue

        # 🌟 优化1：过滤掉太短的废话片段（比如目录、页码、单一的图表标题）
        if len(text_content) < 50:
            continue

        embeddings_list.append(item["vector"])
        texts.append(text_content)
        doc_id = item.get("id", "未知ID")
        metadata.append(f"库内记录ID: {doc_id}")

    embeddings = np.array(embeddings_list)
    num_chunks, dim = embeddings.shape
    print(f"✅ 数据清洗完毕，有效文本块: {num_chunks} 条！")

    # ==========================================
    # 第二步：L2 归一化与聚类
    # ==========================================
    embeddings = normalize(embeddings, norm='l2')

    N = 10  # 提取 10 个核心主题
    print(f"正在将向量聚合成 {N} 个核心簇...")

    kmeans = KMeans(n_clusters=N, random_state=42, n_init="auto")
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_

    # ==========================================
    # 🌟 第三步：计算距离，每个主题提取 Top 3 片段
    # ==========================================
    # 计算所有真实向量到 10 个质心的距离矩阵
    # distances 的 shape 是 (10, num_chunks)
    distances = pairwise_distances(centroids, embeddings)

    # 获取每个簇距离最近的 Top 3 的索引
    top_k = 3
    closest_indices_matrix = np.argsort(distances, axis=1)[:, :top_k]

    # ==========================================
    # 第四步：打印结果（打包输出）
    # ==========================================
    print("\n🎉 成功提取！以下是 Milvus 库中 10 个核心主题的【群组素材】：\n")

    # 用一个列表把所有结果存起来，方便后面发给大模型
    evidence_materials = []

    for cluster_id, chunk_indices in enumerate(closest_indices_matrix):
        print(f"==========【精华主题候选群组 {cluster_id + 1}】==========")

        cluster_texts = []
        for rank, chunk_index in enumerate(chunk_indices):
            core_text = texts[chunk_index]
            source = metadata[chunk_index]
            cluster_texts.append(f"片段{rank + 1} [{source}]: {core_text}")

            print(f"👉 支撑片段 {rank + 1} [{source}]:")
            print(f"文本: {core_text}\n")

        evidence_materials.append({
            "cluster_id": cluster_id + 1,
            "content": "\n".join(cluster_texts)
        })

    return evidence_materials

CARD_API_URL="http://192.168.20.252:2023/api-evimed/news-api/knowledge-card/createEvidenceCard"

def callingCard(topic_id, card_title, user_id):
    url = f"{CARD_API_URL}?topicId={topic_id}&cardTitle={quote(card_title)}&userId={user_id}"
    print(url)
    # logger.info(f"正在发送制卡请求 -> TopicID: {topic_id}, 标题: {card_title}")
    try:
        resp = requests.get(url, timeout=(60, 18000))
        if resp.status_code == 200:
            print(f"制卡成功: {card_title}")
            return True
        return False
    except Exception as e:
        logger.error(f"制卡请求异常: {str(e)}")
        return False

def test():
    cards = [
        # "儿童癫痫基因检测报告怎么看？临床医生避坑指南与精准解读策略",
        # "从经验到精准：SEEG在儿童难治性癫痫术前评估中的应用时机与操作规范",
        # "AI赋能脑电图分析：深度学习算法如何提高儿童癫痫样放电识别效率？",
        # "2025 ILAE癫痫发作分类更新要点解读：如何准确评估儿童发作时的'意识'状态？",
        # "mTOR通路相关癫痫（mTORopathies）的分子诊断与靶向治疗：新型微肽SP7737的转化前景",
        # "儿童友好型剂型破局：奥卡西平混悬液在低龄癫痫患儿中的剂量调整技巧与依从性管理",
        # "首次ASM治疗失败后怎么办？儿童癫痫替代用药的循证选择路径与临床决策",
        # "《吡仑帕奈治疗儿童癫痫中国专家共识》要点提炼：适应症、剂量递增与不良反应管理",
        # "第三代ASM在儿童癫痫中的应用版图：布瓦西坦、拉考沙胺的临床定位与联合方案",
        # "警惕药物副作用的'隐形代价'：左乙拉西坦精神不良反应与丙戊酸钠肝毒性的风险管理",
        # "迷走神经刺激（VNS）在儿童难治性癫痫中的价值：闭环VNS技术的参数优化与疗效评估",
        # "微创癫痫外科新利器：激光间质热疗（LITT）在儿童下丘脑错构瘤继发癫痫中的应用实践",
        # "SEEG引导下射频热凝（RFTC）vs 前颞叶切除术：儿童难治性颞叶癫痫的手术策略选择",
        # "切除还是离断？儿童癫痫综合征外科治疗策略与半球离断术的适应症把握",
        # "丘脑前核刺激（ANT-DBS）治疗难治性儿童癫痫的探索：初步疗效与安全性数据",
        # "Dravet综合征的'疾病修饰'新希望：基因调控疗法ETX101的临床数据解读与未来展望",
        # "发育性癫痫性脑病（DEEs）的病因驱动治疗：从基因诊断到个体化方案的转化路径",
        # "婴儿癫痫痉挛综合征（IESS/West综合征）的早期识别与规范化治疗：2025版罕见病诊疗指南要点",
        # "Bexicaserin在中国的临床试验启动：发育性癫痫性脑病治疗的新选择",
        # "儿童惊厥性癫痫持续状态（CSE）的阶梯化急救处理：从家庭到急诊的全流程管理",
        # "抓住0-3岁大脑可塑性窗口期：癫痫患儿神经发育评估与早期认知康复干预策略",
        # "癫痫患儿何时可以停药？复发风险评估工具与个体化停药决策"
        "浅Ⅱ度与深Ⅱ度烧烫伤的早期精准鉴别与外用药阶梯选择策略",
        "面部精细割伤的无创/微创处理与早期促愈合外用药联合方案",
        "伴有基础疾病（如糖尿病）患者皮肤割伤的难愈合风险评估与外用药调整策略",
        "儿童轻度烫伤的家庭护理高频误区与规范化外用药处方指导",
        "老年人皮肤撕裂伤/割伤的创面特点及温和型外用制剂的选择",
    ]
    user_id = '5773064745007534066'
    topic_id = '315'
    for card in cards:
        callingCard(topic_id, card, user_id)






if __name__ == "__main__":
    # evidence_materials = milvus()
    #
    # materials = "\n\n".join(
    #     f"==========【精华主题候选群组 {item['cluster_id']}】==========\n{item['content']}"
    #     for item in evidence_materials
    # )
    #
    # main(materials)



    # 测试生成
    test()
