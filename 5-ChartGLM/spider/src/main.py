# -*- coding: UTF-8 -*-

from spiders.news_spider import NewsSpider


def crawl_news():
    NewsSpider(redis_key='ysu:news', thread_count=4).start()


if __name__ == '__main__':
    crawl_news()
