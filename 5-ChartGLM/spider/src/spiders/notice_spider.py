# -*- coding: utf-8 -*-
"""
Created on 2024-08-21 02:11:44
---------
@summary:
---------
@author: 小振
"""

import feapder
from feapder.utils.log import log

from items.text_item import TextItem


class NoticeSpider(feapder.Spider):
    """
    通知公告爬虫
    https://mec.ysu.edu.cn/index/tzgg.htm
    """

    def start_requests(self):
        urls = (['https://mec.ysu.edu.cn/index/tzgg.htm'] +
                [f'https://mec.ysu.edu.cn/index/tzgg/{i}.htm' for i in range(1, 118)])
        for url in urls:
            yield feapder.Request(url, callback=self.parse_index)

    def parse_index(self, _, response: feapder.Response):
        log.info(f'正在爬取: {response.url}')
        index_urls = response.xpath(r'//div[@class="list list-dashed"]/ul//@href').extract()
        for url in index_urls:
            yield feapder.Request(url)

    def parse(self, request, response):
        log.info(f'正在解析: {response.url}')
        bs = response.bs4()
        title = bs.find('div', class_='title').string
        content = bs.find('div', class_='v_news_content').get_text(separator=';', strip=True)
        item = TextItem(title=title, content=content, type='notice')
        yield item
