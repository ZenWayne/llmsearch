# SPDX-FileCopyrightText: 2024-present DeepSeek
# SPDX-License-Identifier: Apache-2.0

from haystack import component, logging
from haystack.dataclasses import Document
import requests
from typing import List, Dict
from urllib.parse import urljoin
import time
from .URLMarkdownFetcher import URLMarkdownFetcher


logger = logging.getLogger(__name__)

@component
class SearXNGQueryFetcher(URLMarkdownFetcher):
    """
    通过SearXNG搜索引擎获取搜索结果的组件
    
    Usage example:
    ```python
    from custom_haystack.components.fetcher.searxng_fetcher import SearXNGQueryFetcher
    
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
        result_per_query: int = 5,
        timeout: float = 30.0,
        safe_search: int = 1,
        language: str = "zh-CN"
    ):
        # 显式调用父类初始化
        URLMarkdownFetcher.__init__(self)
        self.base_url = searxng_url
        self.result_per_query = result_per_query
        self.timeout = timeout
        self.safe_search = safe_search
        self.language = language
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
            "Accept": "*/*;"
        }
        logger.info(f"searxng_url: {self.base_url} language: {self.language}")

    def _fetch_single_query(self, query: str) -> List[Dict]:
        """同步获取单个查询的结果"""
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
            response = requests.get(
                url=urljoin(self.base_url, "/search"),
                params=params,
                headers=self.headers,
                timeout=self.timeout
            )
                
            if response.status_code != 200:
                logger.warning(f"搜索失败: HTTP {response.status_code} - {query} - {response.text}")
                return []
                    
            data = response.json()
            return data.get("results", [])[:self.result_per_query]
                
        except Exception as e:
            logger.exception(f"搜索异常: {str(e)} - {query}")
            return []

    def _result_to_document(self, result: Dict) -> Document:
        """将搜索结果转换为Haystack文档格式"""
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
        pass

    @component.output_types(documents=List[Document])
    async def run_async(self, queries: List[str]):
        """
        执行批量搜索查询
        
        :param queries: 搜索关键词列表
        :return: 包含Document对象的字典
        """
        # 执行任务
        time_start = time.time()
        
        search_results = []
        for query in queries:
            search_results.append(self._fetch_single_query(query))

        # 2. 提取所有结果URL
        urls = []
        for sublist in search_results:
            for result in sublist:
                if isinstance(result, dict) and "url" in result:
                    urls.append(result["url"])
        
        # 3. 使用基类爬取能力
        all_results = await self._gather_tasks(urls)
        
        time_end = time.time()
        logger.info(f"完成搜索及爬虫，耗时: {time_end - time_start}秒")
        return {"documents": [doc for doc in all_results if isinstance(doc, Document)]}

# 如果作为主脚本运行
if __name__ == "__main__":
    # 添加模块搜索路径
    import sys
    import os
    # 添加项目根目录到路径中，确保能找到所有模块
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    sys.path.append(project_root)
    print(f"project_root: {project_root}")
  
    # 初始化日志配置
    import logging
    try:
        from utils.logger import CustomFormatter, ContextFilter
        
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
    except ImportError:
        # 如果找不到自定义logger，使用标准配置
        logging.basicConfig(level=logging.ERROR)
    
    # 设置haystack组件的日志级别
    logging.getLogger("haystack").setLevel(logging.ERROR)
    
    # 测试用例
    fetcher = SearXNGQueryFetcher(
        searxng_url="http://localhost:8080/",  # 替换为实际SearXNG实例地址
        result_per_query=20
    )
    
    results = fetcher.run(queries=["深度学习最新进展"])
    for idx, doc in enumerate(results["documents"]):
        print(f"结果 {idx+1}:")
        print(f"标题: {doc.meta.get('title', '')}")
        print(f"文本: {doc.content[0:100]}")
        print(f"链接: {doc.meta['url']}\n") 