from haystack import component, logging 
from haystack import Document
from typing import List, Dict, Any
import aiohttp
import asyncio
from haystack.core.serialization import default_to_dict, default_from_dict

logger = logging.getLogger(__name__)

@component
class SiliconFlowDocumentEmberdder:
    """
    使用SiliconFlow进行文本嵌入的组件
    """
    def __init__(self, 
                 api_key: str,
                 model: str = "BAAI/bge-large-zh-v1.5"
                 ):
        self.siliconflow_url = "https://api.siliconflow.cn/v1/embeddings"
        self.api_key = api_key
        self.model = model

    async def async_embed_text(self, text):
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model,
                    "input": text,
                    "encoding_format": "float"
                }
                async with session.post(self.siliconflow_url, headers=headers, json=payload) as response:
                        response.raise_for_status()
                        obj = await response.json()
                        return obj['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return None

    async def _gather_tasks(self, texts: list):
        """异步任务聚合执行"""
        tasks = [self.async_embed_text(text) for text in texts]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        serialization_dict = default_to_dict(
            self,
            siliconflow_url=self.siliconflow_url,
            model=self.model,
            api_key=self.api_key,
        )

        return serialization_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SiliconFlowDocumentEmberdder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """

        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        pass

    @component.output_types(documents=List[Document])
    async def run_async(self, documents: List[Document]):
        """
        Embed a list of documents.

        :param documents:
            Documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: Documents with embeddings.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "SiliconFlowDocumentEmberdder expects a list of Documents as input."
                "In case you want to embed a list of strings, please use the SiliconFlowTextEmbedder."
            )
        
        if len(documents) == 0:
            raise ValueError("SiliconFlowDocumentEmberdder expects a list of Documents as input.")

        texts_to_embed = [doc.content for doc in documents]

        embeddings = []
            
        embeddings = await self._gather_tasks(texts_to_embed)
        
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
        
        logger.info(f"documents: {documents}")
        
        return {"documents": documents}
    
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

    api_key = os.getenv("SILICONFLOW_API_KEY")
    embedder = SiliconFlowDocumentEmberdder(api_key=api_key, model="BAAI/bge-large-zh-v1.5")
    result = embedder.run(documents=[Document(content="你好"), Document(content="再见")])
    print(result)
