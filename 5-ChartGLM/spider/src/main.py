# -*- coding: UTF-8 -*-

from spiders import *


def crawl_news():
    news_spider.NewsSpider(redis_key='ysu:news', thread_count=4).start()


def crawl_notice():
    notice_spider.NoticeSpider(redis_key='ysu:notice', thread_count=4).start()


if __name__ == '__main__':
    # crawl_news()
    crawl_notice()
