# -*- coding: utf-8 -*-
# @Time    : 07.04.2026
# @File    : create_zone.py.py
# @Software: PyCharm
import requests
import pymysql
from config import (
    DB_CONFIG, CARD_API_URL, EVID_DESC_URL,
    MILVUS_MAIN_URI, MILVUS_MAIN_TOKEN, MILVUS_COLLECTION_MAIN, MILVUS_REFINEDDATA
)

"""
create_zone.py - Description of the file/module
"""
# 开始创建新专区
def creationZone(title):
    # logger.info(f"开始创建新专区: 【{title}】")
    try:
        response = requests.get("https://u48781-9d52-ad09442b.westx.seetacloud.com:8443/evidDesc", params={"desc": title}, verify=False)
        data_text = response.json().get("data", "")
        describe = data_text.split("\n\n")[0] if data_text else "智能生成的专区描述"
    except Exception as e:
        describe = "智能生成的专区描述"


    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    new_topic_id = None
    try:
        sql = """INSERT INTO evidence_topic (title, `describe`, classification, is_pub, publish_auth, card_correlation, `type`) VALUES (%s, %s, 0, 0, 0, 0, 0)"""
        cursor.execute(sql, (title, describe))
        conn.commit()
        new_topic_id = cursor.lastrowid
        # logger.info(f"新专区创建成功，Topic ID: {new_topic_id}")
    except Exception as e:
        conn.rollback()
        logger.error(f"新专区插入数据库失败: {e}")
    finally:
        cursor.close()
        conn.close()

    return new_topic_id

def main():

    titles = [
        '伏诺拉生根除难治性Hp胃炎策略',
        '大剂量双联方案治疗Hp胃炎路径',
        '萎缩性胃炎伴肠化生的逆转治疗',
        '肠道微生态靶向干预慢性胃炎策略',
        '大模型辅助慢性胃炎病理分级应用'
        # '自身免疫性胃炎合并贫血诊疗路径',
     ]
    for title in titles:
        new_topic_id = creationZone(title)
        print(new_topic_id)
    # title = '大模型辅助慢性胃炎病理分级应用'
    # res = creationZone(title)
    # print(res)



if __name__ == "__main__":
    main()
