from haystack import component, logging
import requests
from typing import List

logger = logging.getLogger(__name__)

@component
class SiliconFlowTextEmbedder:
    """
    使用SiliconFlow进行文本嵌入的组件
    """
    def __init__(self, 
                 api_key: str,
                 model: str = "BAAI/bge-large-zh-v1.5",
                 ):
        self.logger = logging.getLogger(__name__)
        self.siliconflow_url = "https://api.siliconflow.cn/v1/embeddings"
        self.api_key = api_key
        self.model = model

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        """
        使用SiliconFlow进行文本嵌入
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": text,
            "encoding_format": "float"
        }
        
        try:
            response = requests.post(
                self.siliconflow_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            obj = response.json()
            # 提取第一个embedding结果
            embedding = obj['data'][0]['embedding']
            return {"embedding": embedding}
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Embedding请求失败: {str(e)}")
            raise
        except (KeyError, IndexError) as e:
            self.logger.error(f"响应格式解析错误: {str(e)}")
            raise

if __name__ == "__main__":
    # 添加模块搜索路径
    import sys
    import os
    # 添加项目根目录到路径中，确保能找到所有模块

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

    api_key = os.getenv("SILICONFLOW_API_KEY")
    embedder = SiliconFlowTextEmbedder(api_key=api_key, model="BAAI/bge-large-zh-v1.5")
    result = embedder.run(text="你好")
    print(result)
