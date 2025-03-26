# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
# SPDX-License-Identifier: Apache-2.0

from haystack import Document, component, logging
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
import asyncio
from typing import List, Optional

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
                 timeout: int = 60000,
                 ):
        self.logger = logging.getLogger(__name__)
        self.crawler = AsyncWebCrawler()
        self.crawler_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            page_timeout=timeout 
        )

    async def _async_crawl(self, url: str) -> Optional[Document]:
        """异步抓取单个URL并转换为Markdown文档"""
        try:
            result = await self.crawler.arun(url=url, config=self.crawler_config)
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
            self.logger.warning(f"抓取失败 {url}. 错误: {str(e)}")
            return None

    async def _gather_tasks(self, tasks: list):
        """异步任务聚合执行"""
        async with self.crawler:
            return await asyncio.gather(*tasks, return_exceptions=True)

    @component.output_types(documents=List[Document])
    def run(self, sources: List[str]):
        """
        异步并发爬取多个URL并转换为Markdown文档
        """
        tasks = [self._async_crawl(url) for url in sources]
        results = asyncio.run(self._gather_tasks(tasks))
        documents = [doc for doc in results if isinstance(doc, Document)]
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