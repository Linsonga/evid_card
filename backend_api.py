# -*- coding: utf-8 -*-
# @Time    : 16.04.2026
# @File    : api.py.py
# @Software: PyCharm

"""
api.py - Description of the file/module
"""


import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

import pymysql
from config import DB_CONFIG

app = FastAPI(title="文件处理回调接口")

# 1. 定义单条数据的 Pydantic 模型
class FileCallbackItem(BaseModel):
    # 注意：前端传来的是 "0000001"，我们先用 str 接收，后续存库时再转为 int
    id: int = Field(..., description="文件ID")
    file_url: str = Field(..., description="文件存储链接")
    file_name: str = Field(..., description="文件名称")
    user_id: str = Field(..., description="用户ID")

# 2. 定义回调接口：接收 List[FileCallbackItem] 作为请求体
@app.post("/api/v1/analysis_file", status_code=200)
async def update_file_status(files: List[FileCallbackItem]):
    """
    接收文件处理完成的回调，更新数据库状态
    """
    if not files:
        raise HTTPException(status_code=400, detail="请求数据不能为空")

    # 提取并转换文件 ID（将字符串 "0000001" 转换为 整数 1 以匹配 bigint）
    valid_ids = []
    for file in files:
        try:
            valid_ids.append(file.id)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"文件 ID [{file.id}] 格式错误，无法转换为数字")

    if not valid_ids:
        return {"code": 200, "message": "没有需要更新的数据"}

    # 3. 模拟数据库批量更新操作
    # 对应的 SQL 逻辑: UPDATE evidence_file_info SET file_status = 1 WHERE id IN (...)
    try:
        mock_db_execute_update(files)
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"数据库更新失败: {str(e)}")

    # 4. 返回 200 成功响应
    return {
        "code": 200,
        "message": "success",
        # "data": {
        #     "updated_count": len(valid_ids),
        #     "updated_ids": valid_ids
        # }
    }


def mock_db_execute_update(files: List[FileCallbackItem]):
    """
    使用 PyMySQL 执行实际的数据库写入和更新操作
    """
    if not files:
        return

    # 1. 准备批量执行的数据
    insert_card_data = []
    update_status_data = []

    for file in files:
        file_id = int(file.id)
        user_id = int(file.user_id)  # 数据库中 user_id 是 bigint，需要强转

        # 你的表结构中 card_name 是 varchar(64)
        # 为防止文件名过长导致 SQL 报错，这里做一个安全截断
        # card_name = file.file_name[:64] if len(file.file_name) > 64 else file.file_name
        card_name = "安神抗癫方联合西药治疗癫痫中血清 SOD 水平回升幅度作为氧自由基清除达标及减药"

        # 准备插入 evidence_file_card_name 的数据元组: (file_id, card_name, user_id)
        insert_card_data.append((file_id, card_name, user_id))

        # 准备更新 evidence_file_info 状态的数据元组: (id,)
        update_status_data.append((file_id,))

    conn = None
    cursor = None
    try:
        # 建立连接
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # --- 开启事务操作 ---

        # 2. 批量写入 evidence_file_card_name 表
        insert_sql = """
            INSERT INTO evidence_file_card_name (file_id, card_name, user_id) 
            VALUES (%s, %s, %s)
        """
        cursor.executemany(insert_sql, insert_card_data)
        print(f"[DB LOG] 成功插入 {cursor.rowcount} 条记录到 evidence_file_card_name")

        # # 3. 批量更新 evidence_file_info 表的状态 (结合你之前的需求)
        # update_sql = """
        #     UPDATE evidence_file_info
        #     SET file_status = 1
        #     WHERE id = %s
        # """
        # cursor.executemany(update_sql, update_status_data)
        # print(f"[DB LOG] 成功更新 {cursor.rowcount} 条记录的 file_status")

        # 4. 提交事务
        conn.commit()

    except Exception as e:
        # 发生异常时回滚，确保数据不会只插入一半
        if conn:
            conn.rollback()
        print(f"[DB ERROR] 数据库操作失败，已回滚: {e}")
        # 重新抛出异常，让 FastAPI 路由捕获并返回 500 错误给客户端
        raise e

    finally:
        # 无论成功失败，都必须关闭游标和连接，释放资源
        if cursor:
            cursor.close()
        if conn:
            conn.close()

if __name__ == "__main__":
    # 启动命令: python main.py
    # uvicorn.run(app, host="0.0.0.0", port=6008, workers=1)
    uvicorn.run(app, host="0.0.0.0", port=5911, workers=1)
