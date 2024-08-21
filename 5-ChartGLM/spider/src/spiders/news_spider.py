# -*- coding: utf-8 -*-
"""
Created on 2024-08-18 20:09:30
---------
@summary:
---------
@author: 小振
"""
import feapder
from feapder.utils.log import log

from items.text_item import TextItem


class NewsSpider(feapder.BaseParser):
    """
    新闻动态爬虫
    https://mec.ysu.edu.cn/xwdt.htm
    """

    def start_requests(self):
        # urls of page 1 to 209
        urls = ['https://mec.ysu.edu.cn/xwdt.htm'] + [f'https://mec.ysu.edu.cn/xwdt/{i}.htm' for i in range(1, 209)]
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
        title = bs.find('div', class_='title').string.replace('\u200B', '')
        content = bs.find('div', class_='v_news_content').get_text(strip=True)
        item = TextItem(title=title, content=content, type='新闻动态')
        yield item
