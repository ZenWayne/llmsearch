from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import MarkdownToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter


import os

from custom_haystack.components.fetcher.SearxngFetcher import SearXNGQueryFetcher
from custom_haystack.components.embedders import SiliconFlowTextEmbedder, SiliconFlowDocumentEmberdder
from custom_haystack.components.builders import DocsPromptBuilder

import time

from haystack.components.retrievers import InMemoryEmbeddingRetriever
import logging


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

## split by passage every 10 lines "\r\n"
def split_by_passage(content: str):
    lines = content.splitlines()  # 按行分割内容
    passages = []
    
    #print("lines: ",lines)
    for i in range(0, len(lines), 10):  # 每10行分割一次
        passage = "\n".join(lines[i:i + 10])  # 组合成一个段落
        passages.append(passage)
    #print("passages: ",passages)
    return passages  # 返回分割后的段落列表

# time cost
start_time = time.time()

# read siliconflow api key from .env file

siliconflow_api_key = os.getenv("SILICONFLOW_API_KEY")
print(f"siliconflow_api_key: {siliconflow_api_key}")


query_str = "今天星期几"

document_store = InMemoryDocumentStore()

use_siliconflow = True
embedder = SiliconFlowDocumentEmberdder(api_key=siliconflow_api_key)

if not use_siliconflow:
    embedder = SentenceTransformersDocumentEmbedder(model="BAAI/bge-m3")
    embedder.warm_up()

pipeline = Pipeline()
pipeline.add_component("fetcher", SearXNGQueryFetcher(searxng_url="http://10.10.44.47:11436/", result_per_query=20))
pipeline.add_component("cleaner", DocumentCleaner())
pipeline.add_component("splitter", DocumentSplitter(split_by="function", splitting_function=split_by_passage))
pipeline.add_component("embedder", embedder)
pipeline.add_component("writer", DocumentWriter(document_store=document_store))
pipeline.connect("fetcher", "cleaner")
pipeline.connect("cleaner", "splitter")
pipeline.connect("splitter", "embedder")
pipeline.connect("embedder", "writer")

result=pipeline.run({"fetcher": {"queries": [query_str]}},
             include_outputs_from={"splitter"})

#print(result)
#print([ {"content":doc.content} for doc in result["splitter"]["documents"]])
#write json to file add indent
# with open("./tmp/splite_result.json", "w", encoding="utf-8") as f:
#     f.write(json.dumps([ {"content":doc.content} for doc in result["splitter"]["documents"]], indent=4))

retriever = InMemoryEmbeddingRetriever(document_store)

embedder = SiliconFlowTextEmbedder(api_key=siliconflow_api_key)

#query
query_pipeline = Pipeline()

template = """
## Input Data

### 【Web Page】
{{contents}}

### 【References】
{{references}}

### 【Question】
{{question}}
"""

prompt_builder = DocsPromptBuilder(template=template)

query_pipeline.add_component("prompt_builder",prompt_builder)
query_pipeline.add_component("embedder",embedder)
query_pipeline.add_component("retriever", retriever)

query_pipeline.connect("embedder.embedding", "retriever.query_embedding")
query_pipeline.connect("retriever.documents", "prompt_builder.documents")

result = query_pipeline.run(data={"prompt_builder": {"question": query_str}})

print(f"query result: {result}")
print(f"time cost: {time.time() - start_time}")