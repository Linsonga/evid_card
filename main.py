import requests
from config import (
    DB_CONFIG, CARD_API_URL, EVID_DESC_URL,
    MILVUS_MAIN_URI, MILVUS_MAIN_TOKEN, MILVUS_COLLECTION_MAIN, MILVUS_REFINEDDATA
)
import pymysql
import logging

# 假设日志配置和常量已定义
logger = logging.getLogger(__name__)


def login():

    # url = "https://second.evimed.com/api-evimed/auth/login"
    url = "https://m.evimed.com/api-evimed/auth/login"
    data = {
        "username": "xxxxx",
        "password": "xxxxx"
    }

    # 使用 json 参数传递字典时，requests 会自动将字典序列化为 JSON 字符串，
    # 并且自动在请求头中设置 'Content-Type': 'application/json'
    try:
        response = requests.post(url, json=data)

        # 检查请求是否成功（状态码为 200）
        if response.status_code == 200:
            print("登录成功！")
            # 打印返回的 JSON 响应数据
            print(response.json())
            # 3. 提取 token
            # 推荐使用 .get() 方法，这样如果找不到 'token' 键，会返回 None 而不是报错退出
            token = response_data.get("token")

            print("登录成功！")
            print(f"获取到的 Token 是: {token}")

        else:
            print(f"请求失败，状态码：{response.status_code}")
            print("错误信息：", response.text)

    except requests.exceptions.RequestException as e:
        print(f"发生网络请求错误：{e}")



def main():
    print("Hello from evidence-card!")


def generate_and_transfer_image(topic_id, title):
    # 接口地址准备
    generate_url = "https://www.evimed.com/api-evimed/news-api/knowledge-card/createPicture"

    # ⚠️ 强烈建议核对这个地址是否有 /api-evimed/ 前缀，这通常是 405 的罪魁祸首
    # upload_url = "https://www.evimed.com/news-api/evidenceWindowController/topic/create"
    upload_url = "https://m.evimed.com/api-evimed/news-api/evidenceWindowController/topic/create"

    print(f"正在根据标题 '{title}' 请求生成图片...")

    gen_response = requests.get(generate_url, params={'title': title})

    if gen_response.status_code != 200:
        print(f"图片生成失败，状态码: {gen_response.status_code}")
        return

    image_url = gen_response.text.strip()
    print(f"成功获取图片链接: {image_url}")

    print("正在以【文件流】模式下载该图片...")

    # 🔴 修改 1：加上 stream=True，告诉 requests 不要立即把文件全部读进内存
    img_response = requests.get(image_url, stream=True)
    print(img_response)

    # 假设你的 title 和上一步的 img_response 已经准备好
    # title = "你的标题"
    # img_response = requests.get(image_url)

    upload_url = "https://m.evimed.com/api-evimed/news-api/evidenceWindowController/topic/create"

    # 🔴 关键修改 1：移除 "Content-Type"，只保留 token
    headers = {
        "token": "07ce8e73bd41544c89bd11c69a4021394"
    }

    # 🔴 关键修改 2：把纯文本信息放在 data 字典里 (注意修复了 users 后面的逗号缺失)
    data = {
        "title": title,
        "describe": "高龄双胎妊娠分娩时机与方式决策策略是妇产科学针对35岁以上孕妇怀有双胎时，如何选择最佳分娩时间和分娩方式的前沿研究领域。",
        "isPub": "0",
        "cardCorrelation": "0",
        "publishAuth": "0",
        "unit": "灵犀医疗",
        "type": "0",
        "users": "16710810141"
    }

    # 🔴 关键修改 3：把图片单独提取到 files 字典里
    # 格式为 '表单字段名': ('文件名', 文件二进制内容, 'MIME类型')
    files = {
        "image": ("cover_image.jpg", img_response.content, "image/jpeg")
    }

    print("正在提交图文数据到服务器...")

    # 发送 POST 请求，同时传入 data 和 files
    upload_response = requests.post(
        upload_url,
        headers=headers,
        data=data,
        files=files
    )

    # 打印最终结果
    if upload_response.status_code == 200:
        print("上传成功！服务器返回：")
        print(upload_response.text)

        res_data = upload_response.json()

        # 情况B：如果返回体是 {"data": {"id": "123"}} （最常见）
        latest_id = res_data.get("data", {}).get("id")

        if latest_id:
            print(f"[{topic_id}] 上传成功！最新 ID 为: {latest_id}")
            return latest_id  # 🔴 关键点：将获取到的新 ID 返回给调用的地方
        else:
            print(f"[{topic_id}] 上传成功，但未在返回数据中找到 ID 字段:", res_data)
            return None

    else:
        print(f"上传失败，状态码: {upload_response.status_code}")
        print(upload_response.text)



def update_card():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    try:
        # 1. 查出需要处理的旧数据
        sql_select = "SELECT id, title FROM evidence_topic WHERE id >= 336 AND id <= 336"
        cursor.execute(sql_select)
        results = cursor.fetchall()
        print(f"从数据库获取到 {len(results)} 条待处理数据。")
    except Exception as e:
        logger.error(f"查询数据库失败: {e}")
        cursor.close()
        conn.close()
        return

    # 2. 遍历每一条旧数据
    for row in results:
        old_topic_id = row[0]
        new_topic_name = row[1]
        print(f"\n================ 开始处理 旧ID: {old_topic_id} ================")
        print(f"\n================ 开始处理 旧: {new_topic_name} ================")

        # 调用函数获取新的 ID
        latest_id = generate_and_transfer_image(old_topic_id, new_topic_name)

        # 3. 如果成功拿到了新 ID，执行数据库连环更新
        if latest_id:
            try:
                # 【第一步】：更新 evidence_card 表
                sql_update_card = "UPDATE evidence_card SET topic_id = %s WHERE topic_id = %s"
                affected_rows_card = cursor.execute(sql_update_card, (latest_id, old_topic_id))

                # 【第二步】：查询 evidence_card 表中刚刚更新过的 info_id
                sql_select_info = "SELECT info_id FROM evidence_card WHERE topic_id = %s"
                cursor.execute(sql_select_info, (latest_id,))
                info_records = cursor.fetchall()

                # 【第三步】：遍历查询到的 info_id，更新 evidence_info 表
                affected_rows_info = 0
                if info_records:
                    sql_update_info = "UPDATE evidence_info SET topic_id = %s WHERE id = %s"
                    for info_row in info_records:
                        info_id = info_row[0]
                        # 增加一层判断，防止数据库里存在 info_id 为 NULL 的脏数据导致报错
                        if info_id is not None:
                            affected_rows_info += cursor.execute(sql_update_info, (latest_id, info_id))

                # 【第四步】：所有修改执行完毕，统一提交事务
                conn.commit()

                print(f"[{old_topic_id}] 数据库双表更新成功！")
                print(f" ├─ evidence_card: 受影响 {affected_rows_card} 行，topic_id 已更新为 {latest_id}")
                print(f" └─ evidence_info: 受影响 {affected_rows_info} 行，共处理 {len(info_records)} 个 info_id")

            except Exception as e:
                # 只要中途任何一条 SQL 报错，两张表的修改都会全部撤销，保证数据安全
                conn.rollback()
                logger.error(f"[{old_topic_id}] 数据库更新失败，已回滚: {e}")
        else:
            print(f"[{old_topic_id}] 由于未能获取到 latest_id，跳过数据库更新。")


    # 4. 全部循环结束后，安全关闭数据库连接
    cursor.close()
    conn.close()
    print("\n所有任务处理完毕！数据库连接已关闭。")





def update_describe():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    try:
        # 1. 查出需要处理的旧数据
        sql_select = "SELECT id, title FROM evidence_topic WHERE id >= 335 AND id <= 416"
        cursor.execute(sql_select)
        results = cursor.fetchall()
        print(f"从数据库获取到 {len(results)} 条待处理数据。")
    except Exception as e:
        logger.error(f"查询数据库失败: {e}")
        cursor.close()
        conn.close()
        return

    # 2. 遍历每一条旧数据
    for row in results:
        topic_id = row[0]  # 建议重命名为 topic_id，更清晰
        topic_title = row[1]

        print(f"\n开始处理 Topic ID: {topic_id}, 标题: {topic_title}")

        try:
            # 加入 timeout 防止网络死锁
            response = requests.get(EVID_DESC_URL, params={"desc": topic_title}, verify=False, timeout=10)

            if response.status_code == 200:
                data_text = response.json().get("data", "")
                describe = data_text.split("\n\n")[0] if data_text else "智能生成的专区描述"
                print(f"[{topic_id}] 成功获取描述: {describe[:20]}...")  # 打印前20个字看看

                # 🔴 核心修复：1. describe 加反引号  2. WHERE 条件改为了 topic_id（请根据实际表结构确认）
                sql_update_card = "UPDATE evidence_topic SET `describe` = %s WHERE id = %s"

                # 🔴 核心修复：传参改为 describe
                affected_rows = cursor.execute(sql_update_card, (describe, topic_id))

                # 每次更新完单条立刻提交（防止后续网络报错导致前面的白跑）
                conn.commit()
                print(f"[{topic_id}] 数据库更新成功，受影响行数: {affected_rows}")
            else:
                print(f"[{topic_id}] 接口请求失败，状态码: {response.status_code}")

        except Exception as e:
            # 捕获单次循环的错误，保证脚本不会因为某一条失败而彻底中断
            logger.error(f"[{topic_id}] 处理发生异常: {e}")
            conn.rollback()

            # 3. 循环结束后，安全关闭连接
    cursor.close()
    conn.close()
    print("\n所有描述更新任务处理完毕！数据库连接已关闭。")


if __name__ == "__main__":
    update_card()
    # update_describe()
