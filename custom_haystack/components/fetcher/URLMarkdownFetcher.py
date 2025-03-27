# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
# SPDX-License-Identifier: Apache-2.0

from haystack import Document, component, logging
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, BrowserConfig
import asyncio
from typing import List, Optional
import concurrent.futures
import traceback


logger = logging.getLogger(__name__)
@component
class URLMarkdownFetcher:
    """
    将URL转换为Markdown文档的抓取器

    使用示例：
    ```python
    from custom_haystack.components.fetcher.url_to_markdown import URLMarkdownFetcher

    fetcher = URLMarkdownFetcher()
    results = fetcher.run(sources=["https://example.com"])
    documents = results["documents"]
    ```
    """
    def __init__(self,
                 timeout: int = 5000,
                 ):
        browser_cfg = BrowserConfig(
            light_mode=True,
            text_mode=True
        )
        self.crawler = AsyncWebCrawler(config=browser_cfg)
        self.crawler_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            page_timeout=timeout 
        )

    async def _async_crawl(self, url: str) -> Optional[Document]:
        """异步抓取单个URL并转换为Markdown文档"""
        logger.info(f"开始抓取 {url}")
        try:
            result = await self.crawler.arun(url=url, config=self.crawler_config)
            logger.info(f"抓取完成 {url} result: {result}")
            return Document(
                content=result.markdown.markdown_with_citations,
                meta={
                    "url": url,
                    "title": result.metadata.get("title", ""),
                    "description": result.metadata.get("description", ""),
                    "author": result.metadata.get("author", "")
                }
            )
        except Exception as e:
            logger.warning(f"抓取失败 {url}. 错误: {str(e)}")
            return None

    async def _gather_tasks(self, urls: list):
        """异步任务聚合执行"""

        tasks = [ self._async_crawl(url) for url in urls ]
        async with self.crawler:
            return await asyncio.gather(*tasks, return_exceptions=True)

    def _thread_pool_run(self, urls: List[str]):
        """线程池运行"""
        def run_coroutine(coro):
            """运行协程并返回结果"""
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            loop.create_task(coro)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_query = {
                executor.submit(run_coroutine(self._async_crawl(url))) for url in urls
            }
            crawl_results = []
            for future in concurrent.futures.as_completed(future_to_query):
                url = future
                try:
                    logger.info(f"抓取成功 '{url}'")
                    result = future.result()
                    crawl_results.append(result)
                except Exception as e:
                    logger.error(f"抓取失败 '{url}' 失败: {str(e)}")
                    traceback.print_exc()
                    crawl_results.append([])
        logger.info(f"抓取完成，共抓取 {len(crawl_results)} 个URL crawl_results: {crawl_results}")
        return crawl_results
    
    @component.output_types(documents=List[Document])
    def run(self, sources: List[str]):
        """
        异步并发爬取多个URL并转换为Markdown文档
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 提交所有任务到线程池
            future_to_query = {
                executor.submit(self._async_crawl, url): url 
                for url in sources
            }
            # 获取结果
            crawl_results = []
            for future in concurrent.futures.as_completed(future_to_query):
                url = future_to_query[future]
                try:
                    result = future.result()
                    crawl_results.append(result)
                except Exception as e:
                    logger.error(f"抓取失败 '{url}' 失败: {str(e)}")
                    crawl_results.append([])
        documents = [doc for doc in crawl_results if isinstance(doc, Document)]
        return {"documents": documents}

# 作为主脚本运行时的测试代码
if __name__ == "__main__":
    fetcher = URLMarkdownFetcher()
    results = fetcher.run(sources=[
        "https://almanac.ximizi.com/"
        ])
    documents = results["documents"]
    with open("documents.md", "w") as f:
        f.write(documents[0].content) 