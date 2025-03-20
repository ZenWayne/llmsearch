
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import MarkdownToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.utils import ComponentDevice, Device
from haystack.components.builders import PromptBuilder


from custom_haystack.components.fetcher.searxng_fetcher import SearXNGQueryFetcher
import time

from haystack.components.retrievers import InMemoryEmbeddingRetriever
import json

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
document_store = InMemoryDocumentStore()
doc_embedder = SentenceTransformersDocumentEmbedder(model="BAAI/bge-m3",
                            device = ComponentDevice.from_single(Device.gpu(id=0))
                                 )
doc_embedder.warm_up()

pipeline = Pipeline()
pipeline.add_component("fetcher", SearXNGQueryFetcher(searxng_url="http://localhost:8080/", result_per_query=20))
pipeline.add_component("converter", MarkdownToDocument())
pipeline.add_component("cleaner", DocumentCleaner())
pipeline.add_component("splitter", DocumentSplitter(split_by="function", splitting_function=split_by_passage))
pipeline.add_component("embedder", doc_embedder)
pipeline.add_component("writer", DocumentWriter(document_store=document_store))
pipeline.connect("converter", "cleaner")
pipeline.connect("cleaner", "splitter")
pipeline.connect("splitter", "embedder")
pipeline.connect("embedder", "writer")

result=pipeline.run({"converter": {"sources": ["./source/crawl_result_1.md"]}},
             include_outputs_from={"splitter"})

print(result)
print([ {"content":doc.content} for doc in result["splitter"]["documents"]])
#write json to file add indent
with open("./tmp/splite_result.json", "w", encoding="utf-8") as f:
    f.write(json.dumps([ {"content":doc.content} for doc in result["splitter"]["documents"]], indent=4))


retriever = InMemoryEmbeddingRetriever(document_store)

embedder = SentenceTransformersTextEmbedder(
                                 model="BAAI/bge-m3",
                                 device = ComponentDevice.from_single(Device.gpu(id=0))
                                 )
#query
query_pipeline = Pipeline()


query_pipeline.add_component("embedder",embedder)
query_pipeline.add_component("retriever", retriever)

query_pipeline.connect("embedder.embedding", "retriever.query_embedding")


result = query_pipeline.run(data={"embedder": {"text": "今天星期几"}})

print(result["retriever"]["documents"][0].content)
print("time cost: ", time.time() - start_time)