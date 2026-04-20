import os
import re
import json
import urllib3
import requests
import threading
import base64      # 🌟 新增：用于图片 Base64 编码
import asyncio     # 🌟 新增：用于异步事件循环处理
import random
from logger import logger
from filelock import FileLock
from datetime import datetime
from openai import OpenAI, AsyncOpenAI
from config import (
    QWEN_API_KEY, QWEN_BASE_URL,
    QWEN_MODEL_DEFAULT, QWEN_MODEL_SEARCH, QWEN_MODEL_MULTI_TURN,
    EMBEDDING_URL,
)

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


# ================= 🌟 新增图片 OCR 识别方法 =================
async def extract_text_from_image(file_path: str) -> str:
    """调用通义千问视觉模型(qwen-vl-ocr)识别本地图片中的文本"""
    loop = asyncio.get_running_loop()

    def sync_ocr():
        # 获取后缀以决定 mime type
        ext = os.path.splitext(file_path)[-1].lower().replace('.', '')
        mime_type = "jpeg" if ext == "jpg" else ext

        # 将本地图片读取并转换为 Base64 格式
        with open(file_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        data_uri = f"data:image/{mime_type};base64,{base64_image}"

        client = AsyncOpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

        completion = client.chat.completions.create(
            model="qwen-vl-ocr-2025-11-20",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_uri
                            },
                        },
                        {"type": "text", "text": "请仅输出图像中的文本内容。"},
                    ],
                },
            ],
        )
        return completion.choices[0].message.content

    # 放入线程池执行，防止阻塞主线程
    return await loop.run_in_executor(None, sync_ocr)



def generate_ai_cover_dashscope_api(title, output_dir="images/covers"):
    """
    按照 DashScope 官方 REST API 示例修改的生图函数
    """
    # 10种围绕“专业极简扁平”的微调风格库
    style_pool = [
        # 1. 经典标准
        "风格：纯粹扁平插画 + 极简主义。无多余装饰。配色：明亮柔和（医疗蓝/薄荷绿/纯白）。背景：极浅的低饱和度冰蓝色或雾灰色。",

        # 3. 矢量细线描边
        "风格：矢量扁平。主体保留纯色填色，外边缘带有干净的单色极细轮廓线。配色：经典白蓝搭配，辅以浅灰。背景：柔和的高级燕麦色或米灰色。",

        # 4. 几何规则化
        "风格：几何化扁平插画。使用极度规则的几何图形去高度概括人体或医疗元素。配色：克制的低饱和冷色调（灰蓝、青绿）。背景：低饱和度的冷灰蓝色调。",

        # 6. 图层半透叠加
        "风格：扁平图层叠加风。利用半透明的纯色块交叠来展现器官或元素的内部逻辑关系。配色：清透的浅水蓝、浅草绿交叠。背景：极浅的紫灰色或丁香灰。",

        # 7. 暖意点缀极简
        "风格：扁平极简 + 极少暖意点缀。在经典的蓝绿冷色调主导下，加入极少量的一个暖色作为视觉焦点。配色：蓝绿为主，微暖色辅助。背景：温暖治愈的奶白色或极浅的珍珠粉色。",

        # 8. 信息图表风
        "风格：现代医疗信息图扁平风。类似高端体检报告或《柳叶刀》等医学期刊中的极简配图。配色：高明度、低饱和度的科技蓝与灰。背景：专业的低饱和度磨砂灰，或带有极浅的水印网格底色 (Light matte gray)。",

        # 9. 负空间剪影
        "风格：正负形极简插画。巧妙利用背景的底色空间来反向勾勒出主体的轮廓。配色：极致精简的双色块。背景：低饱和度的灰青色或浅石板灰。",

        # 10. 临床硬朗切面
        "风格：硬朗极简扁平风。边缘相对锐利，减少圆角过渡，强调果断的切面感和精确度。配色：极其干净的医用白、深灰与亮蓝。背景：无菌感的极浅冷青色或工业浅灰。"
    ]
    # 随机抽取一种风格
    selected_style = random.choice(style_pool)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 1. 配置信息 (建议将 API Key 存入环境变量)
    api_key = QWEN_API_KEY  # 或者直接使用你的 QWEN_API_KEY
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # 2. 构造提示词1
    prompt = f"""你是一名医学科普插画设计师。根据给定主题生成一张封面插画描述（用于AI绘图）。
    【输入主题】：{title}

   【通用核心要求】
    1.提取 1-2 个最核心医学关键词作为视觉线索。
    2.设置2个视觉元素，元素之间需有自然的连接或过渡。
    3.【宽屏构图要求】：
    - 元素尺寸：核心医学视觉元素必须设定为【小巧、精致】（仅占画面5%的比例），拒绝庞大臃肿。
    - 空间排布：严禁将所有元素缩在正中间。请采用左右呼应、黄金分割或对角线构图，将小尺寸元素有节奏地分散。
    4. 图片文字：
       - 仅保留一个核心医学关键词（位置可偏左中或偏右中或中间排列以平衡构图）
       - 严禁长句

    【本张插画的专属视觉设定】
    {selected_style}
    """

    # 3. 构造请求体 (完全参考你的 cURL 示例)
    payload = {
        "model": "wan2.7-image",  # 使用示例中的新模型
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"text": prompt}
                    ]
                }
            ]
        },
        "parameters": {
            "size": "1280*720",  # 也可以根据需求设为 "2K"
            "n": 1,
            "watermark": False,
            "thinking_mode": False  # 开启思考模式，生成质量更高
        }
    }

    try:
        # 4. 发起 POST 请求
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        result = response.json()

        # 5. 解析响应结果
        # 注意：REST API 返回的结果层级通常为 result['output']['results'][0]['url']
        if response.status_code == 200:
            # image_url = result.get("output", {}).get("results", [{}])[0].get("url")

            image_url = (
                result.get("output", {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", [{}])[0]
                .get("image")
            )

            if not image_url:
                print(f"⚠️ 未获取到图片 URL: {result}")
                return None

            # 6. 下载并保存（转 JPG 压缩，控制在 1M 以内）
            from PIL import Image
            import io
            img_data = requests.get(image_url).content
            safe_title = "".join(c for c in title if c.isalnum())[:20]
            file_name = f"{safe_title}.jpg"
            file_path = os.path.join(output_dir, file_name)

            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            quality = 85
            while quality >= 20:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality)
                if buf.tell() <= 1 * 1024 * 1024:
                    break
                quality -= 10
            with open(file_path, "wb") as f:
                f.write(buf.getvalue())

            return os.path.abspath(file_path)
        else:
            print(f"❌ API 请求失败: {response.status_code}, {response.text}")
            return None

    except Exception as e:
        print(f"❌ AI 生图函数异常: {e}")
        return None



# def generate_ai_cover_openai(title, output_dir="images/covers"):
#     """
#     使用 OpenAI SDK 兼容模式调用阿里云通义万相生成封面图
#     :param title: 文章标题，用于生成 Prompt
#     :param api_key: 阿里云 DashScope API Key
#     :param output_dir: 本地存储目录
#     :return: 成功返回图片的绝对路径，失败返回 None
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir, exist_ok=True)
#
#     # 1. 初始化兼容 OpenAI 的客户端
#     client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)
#
#     # 2. 构造适合医疗/科普背景的提示词
#     prompt = f"提取主题中关键词，生成一张扁平插画风格的健康科普封面，主题关于：{title}，色彩明亮柔和，极简主义，高清，无文字"
#
#     try:
#         # 3. 发起生图请求
#         response = client.images.generate(
#             model="wan2.1-t2i-plus",
#             prompt=prompt,
#             n=1,
#             size="1440x612"  # 通义万相支持的尺寸
#         )
#
#         image_url = response.data[0].url
#         if not image_url:
#             return None
#
#         # 4. 下载并保存
#         img_data = requests.get(image_url).content
#         # 处理文件名：移除非法字符
#         safe_title = "".join(c for c in title if c.isalnum())[:20]
#         timestamp = int(datetime.now().timestamp())
#         file_name = f"cover_{safe_title}_{timestamp}.png"
#         file_path = os.path.join(output_dir, file_name)
#
#         with open(file_path, "wb") as f:
#             f.write(img_data)
#
#         return os.path.abspath(file_path)
#
#     except Exception as e:
#         print(f"❌ AI 生图函数异常: {e}")
#         return None

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



VECTOR_CACHE_FILE = "local_vector_cache.jsonl"
VECTOR_CACHE_LOCK = f"{VECTOR_CACHE_FILE}.lock"

# 全局内存缓存，仅在启动时或第一次调用时加载一次
_global_vector_cache = None
_vector_memory_lock = threading.Lock() #新增：内存级别的锁

def load_vector_cache():
    """将 JSONL 文件一次性加载到内存字典中"""
    global _global_vector_cache
    if _global_vector_cache is None:
        _global_vector_cache = {}
        if os.path.exists(VECTOR_CACHE_FILE):
            with open(VECTOR_CACHE_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # data 格式约定为: {"title": [0.1, 0.2, ...]}
                        _global_vector_cache.update(data)
                    except Exception as e:
                        print(f"解析缓存行失败，跳过: {e}")
        print(f"✅ 向量本地缓存加载成功，共载入 {len(_global_vector_cache)} 条记录。")
    return _global_vector_cache


def get_cached_vector(title):
    if not title:
        return []

    cache = load_vector_cache()

    # 1. 无锁极速查询（命中率最高）
    if title in cache:
        return cache[title]

    # 2.修改：未命中时加锁，并进行二次检查
    with _vector_memory_lock:
        # 双重检查：防止在等待锁的期间，别的线程已经把这个 title 请求回来并存入 cache 了
        if title in cache:
            return cache[title]

        print(f"🚀 触发大模型向量接口: {title[:20]}...")
        vec = vector_4b(title)

        if vec:
            cache[title] = vec
            # 文件锁依然保留，处理多进程或安全写入
            with FileLock(VECTOR_CACHE_LOCK, timeout=10):
                with open(VECTOR_CACHE_FILE, "a", encoding="utf-8") as f:
                    record = {title: vec}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return vec




DIST_LOG_FILE = os.path.join(os.getcwd(), 'release_md.jsonl')
dist_lock = threading.Lock()

def get_distributed_filenames(platform='wechat'):
    """获取已成功分发的文件名列表"""
    if not os.path.exists(DIST_LOG_FILE):
        return set()

    distributed = set()
    with dist_lock:
        with open(DIST_LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # 只有状态为 success 的才算已发送，失败的第二天还可以重试
                    if data.get('type') == platform and data.get('status') == 'success':
                        distributed.add(data.get('filename'))
                except:
                    continue
    return distributed


def record_distribution(platform, filename, status, note=""):
    """记录分发结果追加到 jsonl 文件"""
    log_entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": platform,
        "filename": filename,
        "status": status,
        "note": note
    }
    with dist_lock:
        with open(DIST_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
