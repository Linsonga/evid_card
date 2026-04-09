import re
import json
import urllib3
import requests
from openai import OpenAI, AsyncOpenAI
from config import (
    QWEN_API_KEY, QWEN_BASE_URL,
    QWEN_MODEL_DEFAULT, QWEN_MODEL_SEARCH, QWEN_MODEL_MULTI_TURN,
    EMBEDDING_URL,
)
from logger import logger

# 屏蔽 requests verify=False 引起的警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ================= 大模型调用 =================
def requestQwen(system_prompt, user_prompt):
    """单轮大模型调用（基础接口）"""
    client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
    response = client.chat.completions.create(
        model=QWEN_MODEL_MULTI_TURN,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def call_qwen(system_prompt, user_prompt, model=None, temperature=0.1, enable_search=False):
    """单轮大模型调用（支持联网搜索，审核模块统一接口）"""
    client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
    use_model = QWEN_MODEL_SEARCH if enable_search else (model or QWEN_MODEL_DEFAULT)
    kwargs = {
        "model": use_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    if enable_search:
        kwargs["extra_body"] = {"enable_search": True}
    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content


def requestQwenMultiTurn(system_prompt, messages, enable_thinking=True, enable_search=True):
    """多轮对话大模型调用，messages 为完整的对话历史列表"""
    client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
    completion = client.chat.completions.create(
        model=QWEN_MODEL_MULTI_TURN,
        messages=[{"role": "system", "content": system_prompt}] + messages,
        extra_body={
            "enable_thinking": enable_thinking,
            "enable_search": enable_search,
        },
    )
    return completion.choices[0].message.content

# def requestQwenMultiTurn(system_prompt, messages):
#     """多轮对话大模型调用，messages 为完整的对话历史列表"""
#     client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
#     completion = client.chat.completions.create(
#         model=QWEN_MODEL_MULTI_TURN,
#         messages=[{"role": "system", "content": system_prompt}] + messages,
#         extra_body={
#             "enable_thinking": True,
#             "enable_search":   True,
#         },
#     )
#     return completion.choices[0].message.content


async def request_qwen_async(system_prompt, user_prompt):
    """异步单轮大模型调用"""
    client = AsyncOpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
    completion = await client.chat.completions.create(
        model=QWEN_MODEL_DEFAULT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content


# ================= JSON 工具 =================

def extract_json_from_text(text):
    """从大模型返回文本中提取 JSON 对象（防止 ```json 外壳干扰）"""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def extract_json_array_from_text(text):
    """从大模型返回文本中提取 JSON 数组"""
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return []


# ================= 向量化 =================

def vector_4b(text):
    """获取向量（通过远程 Embedding API）"""
    params = {"user_input": text}
    try:
        response = requests.get(EMBEDDING_URL, params=params, verify=False, timeout=15)
        vector_str = response.text.strip()
        return [float(x) for x in vector_str.split(",")]
    except Exception as e:
        logger.error(f"获取向量失败 (API异常): {e}")
        return []
