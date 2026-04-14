# -*- coding: utf-8 -*-
# @Time    : 14.04.2026
# @File    : publish_test.py.py
# @Software: PyCharm

"""
# 知乎发布文章
publish_test.py - Description of the file/module
"""

from zhihu_mcp_server.server import create_atticle


def main():
    # 准备发布的内容
    title = "测试在Linux服务器通过自动脚本发布知乎文章"  # 文章标题，不多于100个字符
    content = "这是一篇测试文章的内容。测试使用 Cookies 在 Linux 无头浏览器环境下自动发布文章是否成功。"  # 文章内容，不少于9个字
    images = []  # 如果有封面图片，填入本地绝对路径，例如 ["/root/images/cover.jpg"]
    topic = "测试"  # 最好输入已存在的话题，否则可能无法成功发帖

    print(f"准备发布文章: {title}")

    # 调用发文接口 (注意代码中的原方法名为 create_atticle)
    result = create_atticle(title=title, content=content, images=images, topic=topic)

    print("执行结果:", result[0].text)


if __name__ == "__main__":
    main()