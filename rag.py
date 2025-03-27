from haystack import AsyncPipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import MarkdownToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

from custom_haystack.components.fetcher.SearxngFetcher import SearXNGQueryFetcher
from custom_haystack.components.embedders import SiliconFlowTextEmbedder, SiliconFlowDocumentEmberdder
from custom_haystack.components.builders import DocsPromptBuilder
from custom_haystack.components.generators import CustomOpenAIGenerator

import time
import json
import logging
from typing import Callable

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(
        self,
        split_lines: int = 10,
        searxng_url: str = "http://127.0.0.1:8080/",
        result_per_query: int = 5,
        use_siliconflow_embedder: bool = True,
        streaming_callback: Callable = None,
        model: str = "qwen-qwq-32b",
        language: str = "zh-CN"
    ):
        self.split_lines = split_lines
        self.searxng_url = searxng_url
        self.result_per_query = result_per_query
        self.use_siliconflow_embedder = use_siliconflow_embedder
        self.streaming_callback = streaming_callback
        self.language = language
        if self.language == "en":
            self.template_path = "./template/query_template.en.md"
        else:
            self.template_path = "./template/query_template.md"
        # 初始化文档存储
        self.document_store = InMemoryDocumentStore()
        self.model = model
        
        # 初始化嵌入器
        if self.use_siliconflow_embedder:
            self.siliconflow_api_key = Secret.from_env_var("SILICONFLOW_API_KEY").resolve_value()
            self.embedder = SiliconFlowDocumentEmberdder(api_key=self.siliconflow_api_key)
        else:
            self.embedder = SentenceTransformersDocumentEmbedder(model="BAAI/bge-m3")
            self.embedder.warm_up()
        
        self.api_key = ""
        self.api_base_url = ""
        if Secret.from_env_var("GROQ_API_KEY"):
            self.api_key = Secret.from_env_var("GROQ_API_KEY")
            self.api_base_url = Secret.from_env_var("OPENAI_API_BASE_URL").resolve_value() or "https://api.groq.com/openai/v1"
        elif Secret.from_env_var("SILICONFLOW_API_KEY"):
            self.api_key = Secret.from_env_var("SILICONFLOW_API_KEY")
            self.api_base_url = Secret.from_env_var("OPENAI_API_BASE_URL").resolve_value() or "https://api.siliconflow.com/v1"
        elif Secret.from_env_var("OPENAI_API_KEY"):
            self.api_key = Secret.from_env_var("OPENAI_API_KEY")
            self.api_base_url = Secret.from_env_var("OPENAI_API_BASE_URL").resolve_value() or "https://api.openai.com/v1"
        else:
            raise ValueError("No API key found")
        # 初始化管道
        self._init_pipeline()
        self._init_query_pipeline()
        
    def split_by_passage(self, content: str):
        lines = content.splitlines()
        passages = []
        for i in range(0, len(lines), self.split_lines):
            passage = "\n".join(lines[i:i + self.split_lines])
            passages.append(passage)
        return passages
    
    def _init_pipeline(self):
        self.pipeline = AsyncPipeline()
        self.pipeline.add_component("fetcher", SearXNGQueryFetcher(
            searxng_url=self.searxng_url,
            result_per_query=self.result_per_query,
            language=self.language
        ))
        self.pipeline.add_component("cleaner", DocumentCleaner())
        self.pipeline.add_component("splitter", DocumentSplitter(
            split_by="function",
            splitting_function=self.split_by_passage
        ))
        self.pipeline.add_component("embedder", self.embedder)
        self.pipeline.add_component("writer", DocumentWriter(
            document_store=self.document_store,
            policy=DuplicatePolicy.OVERWRITE
        ))
        
        # 连接组件
        self.pipeline.connect("fetcher", "cleaner")
        self.pipeline.connect("cleaner", "splitter")
        self.pipeline.connect("splitter", "embedder")
        self.pipeline.connect("embedder", "writer")
        
    def _init_query_pipeline(self):
        self.retriever = InMemoryEmbeddingRetriever(self.document_store)
        if self.use_siliconflow_embedder:
            self.query_embedder = SiliconFlowTextEmbedder(api_key=self.siliconflow_api_key)
        else:
            self.query_embedder = SentenceTransformersTextEmbedder(model="BAAI/bge-m3")
        
        # 读取模板
        with open(self.template_path, "r", encoding="utf-8") as f:
            template = f.read()
            
        #logger.info(f"template: {template}")
        self.prompt_builder = DocsPromptBuilder(template=template)
        


        self.query_pipeline = AsyncPipeline()
        self.query_pipeline.add_component("embedder", self.query_embedder)
        self.query_pipeline.add_component("retriever", self.retriever)
        self.query_pipeline.add_component("prompt_builder", self.prompt_builder)
        self.query_pipeline.add_component("llm", 
            CustomOpenAIGenerator(
                api_key=self.api_key,
                api_base_url=self.api_base_url,
                model=self.model
            ))
            
        # 连接组件
        self.query_pipeline.connect("embedder.embedding", "retriever.query_embedding")
        self.query_pipeline.connect("retriever.documents", "prompt_builder.documents")
        self.query_pipeline.connect("prompt_builder", "llm")
        
    async def process_query(self, query_str: str, streaming_callback: Callable = None):
        # 处理查询并获取文档
        result = await self.pipeline.run_async(
            {"fetcher": {"queries": [query_str]}},
            include_outputs_from={"splitter"}
        )
        
        # 保存分割结果
        # with open("./tmp/splite_result.json", "w", encoding="utf-8") as f:
        #     f.write(json.dumps(
        #         [{"content": doc.content} for doc in result["splitter"]["documents"]],
        #         indent=4,
        #         ensure_ascii=False
        #     ))
            
        # 执行查询
        query_result = await self.query_pipeline.run_async(
            data={
                "embedder": {"text": query_str},
                "prompt_builder": {"question": query_str},
                "llm": {"streaming_callback": streaming_callback if streaming_callback else self.streaming_callback}
            }
        )
        
        # 保存查询结果
        with open("./tmp/query_result.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(query_result, indent=4, ensure_ascii=False))
            
        return query_result['llm']['replies'][0]

# 使用示例
if __name__ == "__main__":
    try:
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
    except ImportError:
        # 如果找不到自定义logger，使用标准配置
        logging.basicConfig(level=logging.ERROR)
    # 设置haystack组件的日志级别
    logging.getLogger("haystack").setLevel(logging.ERROR)
    start_time = time.time()
    
    # 创建RAG系统实例
    rag_system = RAGSystem(
        split_lines=10,
        searxng_url="http://127.0.0.1:8080/",
        result_per_query=5
    )
    
    # 执行查询
    query_str = "今天星期几"
    result = rag_system.process_query(query_str)
    print(f"result: {result}")
    
    print(f"time cost: {time.time() - start_time}")