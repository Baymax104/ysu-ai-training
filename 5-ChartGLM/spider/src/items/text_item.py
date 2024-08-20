# -*- coding: utf-8 -*-
"""
Created on 2024-08-20 22:52:55
---------
@summary:
---------
@author: 小振
"""

from feapder import Item


class TextItem(Item):
    """
    This class was generated by feapder
    command: feapder create -i text
    """

    __table_name__ = "text"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.id = None  # 主键
        self.title = kwargs.get('title')  # 标题
        self.content = kwargs.get('content')  # 文章内容
        self.type = kwargs.get('type')  # 文本类别