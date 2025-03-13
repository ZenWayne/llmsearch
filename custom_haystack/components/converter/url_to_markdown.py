# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
# SPDX-License-Identifier: Apache-2.0

from haystack import Document, component, logging
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
import asyncio
from typing import List
logger = logging.getLogger(__name__)

@component
class URLToMarkdownDocument:
    """
    Converts a URL into a Markdown Document.

    Usage example:
    ```python
    from custom_haystack.components.converter.url_to_markdown import URLToMarkdownDocument

    converter = URLToMarkdownDocument()
    results = converter.run(sources=["https://example.com/sample.md"])
    documents = results["documents"]
    print(documents[0].content)
    # 'This is a text from the markdown file at the URL.'
    ```
    """
    def __init__(self):
        self.crawler = AsyncWebCrawler()
        self.crawler_config = CrawlerRunConfig(
            #cache_mode=CacheMode.BYPASS,
        )

    async def _async_crawl(self, url: str):
        try:
            
            result = await self.crawler.arun(url=url, config=self.crawler_config)
            return Document(content=result._markdown.markdown_with_citations, meta=
                            {
                                "url": url, 
                                "title": result.metadata.get("title", ""),
                                "description": result.metadata.get("description", ""),
                                "author": result.metadata.get("author", "")
                            }
                            )
        except Exception as e:
            logger.warning(f"Could not fetch {url}. Skipping it. Error: {str(e)}")
            return None

    @component.output_types(documents=List[Document])
    def run(self, sources: List[str]):
        """
        异步并发爬取多个URL并转换为Markdown文档
        """
        # 创建异步任务列表
        tasks = [self._async_crawl(url) for url in sources]
        
        async def _gather_tasks():
            async with self.crawler:
                return await asyncio.gather(*tasks, return_exceptions=True)
                
        # 并发执行所有任务
        results = asyncio.run(_gather_tasks())
        # 过滤有效结果
        documents = [doc for doc in results if isinstance(doc, Document)]
        
        return {"documents": documents}

if __name__ == "__main__":
    converter = URLToMarkdownDocument()
    results = converter.run(sources=[
        #"https://docs.crawl4ai.com/",
        #"https://docs.haystack.deepset.ai/docs/intro"
        "https://almanac.ximizi.com/"
        ])
    documents = results["documents"]
    #wirte first to file
    with open("documents.md", "w") as f:
        f.write(documents[0].content)
    # 'This is a text from the markdown file at the URL.'
