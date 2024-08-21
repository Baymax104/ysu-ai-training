# -*- coding: UTF-8 -*-
from feapder import Spider

from spiders import *

if __name__ == '__main__':
    spider = Spider(redis_key='ysu', thread_count=10)
    spider.add_parser(news_spider.NewsSpider)
    spider.add_parser(notice_spider.NoticeSpider)
    spider.start()
