# SPDX-FileCopyrightText: 2024-present DeepSeek
# SPDX-License-Identifier: Apache-2.0

from haystack import Document, component, logging
import aiohttp
import asyncio
from typing import List, Dict
from urllib.parse import urljoin
import json

logger = logging.getLogger(__name__)

@component
class SearXNGQueryFetcher:
    """
    通过SearXNG搜索引擎获取搜索结果的组件
    
    Usage example:
    ```python
    from custom_haystack.components.queryfetcher.searxng_fetcher import SearXNGQueryFetcher
    
    fetcher = SearXNGQueryFetcher(
        searxng_url="http://localhost:8080",
        max_results=3
    )
    results = fetcher.run(queries=["最新AI技术发展", "大语言模型应用场景"])
    documents = results["documents"]
    print(f"获取到{len(documents)}条结果")
    ```
    """
    
    def __init__(
        self,
        searxng_url: str = "http://localhost:8080",
        max_results: int = 5,
        timeout: float = 30.0,
        safe_search: int = 1,
        language: str = "zh-CN"
    ):
        self.base_url = searxng_url
        self.max_results = max_results
        self.timeout = timeout
        self.safe_search = safe_search
        self.language = language
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
            "Accept": "*/*;"
        }
        logger.info(f"searxng_url: {self.base_url}")

    async def _fetch_single_query(self, session: aiohttp.ClientSession, query: str) -> List[Dict]:
        """异步获取单个查询的结果"""
        params = {
            "q": query,
            "format": "json",
            "language": self.language,
            "safesearch": self.safe_search,
            "pageno": 1,
            "time_range": None,
            "categories": "general"
        }
        
        # 过滤掉值为None的参数
        params = {k: v for k, v in params.items() if v is not None}
        
        try:
            async with session.get(
                url=urljoin(self.base_url, "/search"),
                params=params,  # 将参数放在请求体中
                timeout=self.timeout
            ) as response:
                
                if response.status != 200:
                    logger.warning(f"搜索失败: HTTP {response.status} - {query} - {await response.text()}")
                    return []
                    
                data = await response.json(loads=json.loads)
                return data.get("results", [])[:self.max_results]
                
        except Exception as e:
            log_exception(e, f"搜索异常: {str(e)} - {query}")
            return []

    def _result_to_document(self, result: Dict) -> Document:
        """将搜索结果转换为Haystack文档格式"""
        logger.info(f"result: {result}")
        content = f"{result.get('content', '')}"
        metadata = {
            "url": result.get("url", ""),
            "title": result.get("title", ""),
            "source_engine": result.get("engine", "unknown"),
            "category": result.get("category", "general"),
            "score": float(result.get("score", 0)),
            #"query_time": result.get("engines", [{}])[0].get("time", 0)
        }
        return Document(content=content, meta=metadata)

    @component.output_types(documents=List[Document])
    def run(self, queries: List[str]):
        """
        执行批量搜索查询
        
        :param queries: 搜索关键词列表
        :return: 包含Document对象的字典
        """
        async def _async_run():
            async with aiohttp.ClientSession(headers=self.headers) as session:
                tasks = [self._fetch_single_query(session, q) for q in queries]
                results = await asyncio.gather(*tasks)
                return results
                
        # 执行异步任务
        all_results = asyncio.run(_async_run())
        
        # 转换结果格式
        documents = []
        for query_results in all_results:
            for result in query_results:
                documents.append(self._result_to_document(result))
                
        return {"documents": documents}

if __name__ == "__main__":
    # 添加模块搜索路径
    import sys
    import os
    # custom_haystack/components/fetcher/searxng_fetcher.py
    sys.path.append(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(__file__)  # 假设logger.py在项目根目录
                )
            )
        )
    )
  
    # 初始化日志配置
    import logging
    from logger import CustomFormatter, ContextFilter
    
    # 创建根logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # 添加控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    root_logger.addHandler(console_handler)
    
    # 添加上下文过滤器
    context_filter = ContextFilter()
    root_logger.addFilter(context_filter)
    
    # 设置haystack组件的日志级别
    logging.getLogger("haystack").setLevel(logging.INFO)
    
    # 测试用例
    fetcher = SearXNGQueryFetcher(
        searxng_url="http://localhost:8080/",  # 替换为实际SearXNG实例地址
        max_results=2
    )
    
    results = fetcher.run(queries=["2024年AI发展趋势", "深度学习最新进展"])
    for idx, doc in enumerate(results["documents"]):
        print(f"结果 {idx+1}:")
        print(f"标题: {doc.meta.get('title', '')}")
        print(f"摘要: {doc.content}")
        print(f"链接: {doc.meta['url']}\n") 