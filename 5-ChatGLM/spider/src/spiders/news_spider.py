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


class NewsSpider(feapder.Spider):
    """
    新闻动态爬虫
    党群新闻：https://mec.ysu.edu.cn/xwdt/dqxw.htm
    行政新闻：https://mec.ysu.edu.cn/xwdt/xzxw.htm
    教学信息：https://mec.ysu.edu.cn/xwdt/jxxx.htm
    科研动态：https://mec.ysu.edu.cn/xwdt/kydt.htm
    招生就业：https://mec.ysu.edu.cn/xwdt/zsjy.htm
    教师信息：https://mec.ysu.edu.cn/xwdt/jsxx.htm
    学工动态：https://mec.ysu.edu.cn/xwdt/xgdt.htm
    """

    def start_requests(self):
        section_urls = {
            'https://mec.ysu.edu.cn/xwdt/dqxw.htm': (35, '党群新闻'),
            'https://mec.ysu.edu.cn/xwdt/xzxw.htm': (22, '行政新闻'),
            'https://mec.ysu.edu.cn/xwdt/jxxx.htm': (22, '教学信息'),
            'https://mec.ysu.edu.cn/xwdt/kydt.htm': (13, '科研动态'),
            'https://mec.ysu.edu.cn/xwdt/zsjy.htm': (5, '招生就业'),
            'https://mec.ysu.edu.cn/xwdt/jsxx.htm': (3, '教师信息'),
            'https://mec.ysu.edu.cn/xwdt/xgdt.htm': (101, '学工动态')
        }
        for url in section_urls.keys():
            yield feapder.Request(url, callback=self.parse_section, section_urls=section_urls)

    def parse_section(self, request, response):
        log.info(f'正在解析章节: {response.url}')
        section_urls = request.section_urls
        pages, section = section_urls[request.url]
        page_urls = [request.url] + [f'{request.url.rstrip(".htm")}/{i}.htm' for i in range(1, pages)]
        for url in page_urls:
            yield feapder.Request(url, callback=self.parse_index, section=section)

    def parse_index(self, request, response):
        log.info(f'正在解析索引页: {response.url}')
        index_urls = response.xpath('//div[@class="list list-dashed"]/ul//@href').extract()
        for url in index_urls:
            yield feapder.Request(url, section=request.section)

    def parse(self, request, response):
        log.info(f'正在解析文章: {response.url}')
        bs = response.bs4()
        title = bs.find('div', class_='title').string.replace('\u200B', '')
        content = bs.find('div', class_='v_news_content').get_text(strip=True)
        section = request.section
        item = TextItem(title=title, content=content, section=section)
        yield item
