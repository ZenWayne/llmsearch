from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from haystack.dataclasses import StreamingChunk
from openai.types.chat import ChatCompletionChunk
from haystack.dataclasses.chat_message import ChatMessage
from haystack.components.generators.openai import OpenAIGenerator
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import asyncio
from rag import RAGSystem
import logging
import nest_asyncio
import json
from dotenv import load_dotenv

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
logger = logging.getLogger("haystack")

app = FastAPI()

model : Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str
    stream: bool = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

async def stream_response(response_queue: asyncio.Queue[ChatCompletionChunk]):
    while True:
        try:
            streaming_chunk : ChatCompletionChunk = await response_queue.get()
            if streaming_chunk is None:  # 结束信号
                break
            json_data = streaming_chunk.model_dump(mode="json")
            logger.info(f"data: {json_data}")
            yield f"data: {json.dumps(json_data, ensure_ascii=False)}\n\n".encode('utf-8')
        except asyncio.CancelledError:
            break

# 创建该请求专用的RAG系统实例
request_rag : Optional[RAGSystem] = None

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    # 获取最后一条用户消息
    user_message = next((msg for msg in reversed(request.messages) if msg.role == "user"), None)
    if not user_message:
        return {"error": "No user message found"}
    
    query = user_message.content
    logger.info(f"query: {query}")
    
    # 如果是流式请求
    if request.stream:
        # 为每个请求创建独立的响应队列
        request_queue = asyncio.Queue()
        
        # 启动后台任务处理查询
        async def process_query():

            try:
                await request_rag.process_query(query, streaming_callback=lambda x: request_queue.put_nowait(x))
                await request_queue.put(None)  # 结束信号
            except Exception as e:
                exception(e, f"Error processing query: {e}")
                await request_queue.put(None)
        
        # 启动后台任务
        loop = asyncio.get_event_loop()
        task = loop.create_task(process_query())
        try:
            await task  # Wait for the task to complete
        except Exception as e:
            print(f"Caught exception: {e}")
            exception(e, f"Error processing query: {e}")
        else:
            print("Task succeeded")
        finally:
            if not task.cancelled():
                # Check for unhandled exceptions
                if task.exception() is not None:
                    print(f"Unhandled exception: {task.exception()}")
        # 返回流式响应
        return StreamingResponse(
            stream_response(request_queue),
            media_type="text/event-stream",
            headers={"Transfer-Encoding":"chunked"}
        )
    
    # 非流式请求
    try:
        result = request_rag.process_query(query)
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1694268190,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result
                },
                "finish_reason": "stop"
            }]
        }
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    nest_asyncio.apply()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    load_dotenv(".env")
    model = os.getenv("MODEL")
    language = os.getenv("LANGUAGE")
    logger.info(f"language: {language}")
    request_rag = RAGSystem(
        split_lines=10,
        searxng_url=os.getenv("SEARXNG_URL", "http://127.0.0.1:8080/"),
        result_per_query=5,
        model=model,
        use_siliconflow_embedder=os.getenv("USE_SILICONFLOW_EMBEDDER", "true") == "true",
        language=language
    )

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port) 