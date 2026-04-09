# -*- coding: utf-8 -*-
# @Time    : 09.04.2026
# @File    : database.py.py
# @Software: PyCharm

"""
database.py - Description of the file/module
"""

import pymysql
from dbutils.pooled_db import PooledDB
from config import DB_CONFIG

# 初始化全局的、线程安全的 MySQL 连接池
db_pool = PooledDB(
    creator=pymysql,      # 使用链接数据库的模块
    maxconnections=20,    # 连接池允许的最大连接数
    mincached=5,          # 初始化时，链接池中至少创建的空闲的链接
    blocking=True,        # 连接池满时阻塞等待
    ping=1,               # ping MySQL 服务，检查连接是否可用
    **DB_CONFIG
)

def get_db_connection():
    """从连接池中获取一个连接"""
    return db_pool.connection()
