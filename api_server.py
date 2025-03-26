from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from haystack.dataclasses import StreamingChunk
from openai.types.chat import ChatCompletionChunk
from haystack.dataclasses.chat_message import ChatMessage
from haystack.components.generators.openai import OpenAIGenerator
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import asyncio
from rag import RAGSystem
import logging
import nest_asyncio
nest_asyncio.apply()

try:
    from logger import CustomFormatter, ContextFilter, exception
    
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

logger = logging.getLogger(__name__)

app = FastAPI()

model = "qwen-qwq-32b"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str
    stream: bool = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

async def stream_response(response_queue: asyncio.Queue):
    while True:
        try:
            streaming_chunk : StreamingChunk = await response_queue.get()
            completion_chunk : ChatCompletionChunk = ChatCompletionChunk()
            completion_chunk.model = model
            completion_chunk.usage = {}
            chat_message: ChatMessage=OpenAIGenerator._create_message_from_chunks(completion_chunk, [streaming_chunk])
            if streaming_chunk is None:  # 结束信号
                break
            logger.info(f"data: {chat_message.meta}")
            yield f"data: {chat_message.meta}\n\n"
        except asyncio.CancelledError:
            break

# 创建该请求专用的RAG系统实例
request_rag = RAGSystem(
    split_lines=10,
    searxng_url="http://127.0.0.1:8080/",
    result_per_query=5,
    model=model
)

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
        loop.create_task(process_query())
        
        # 返回流式响应
        return StreamingResponse(
            stream_response(request_queue),
            media_type="text/event-stream"
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
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    uvicorn.run(app, host="0.0.0.0", port=8001) 