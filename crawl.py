import asyncio
from crawl4ai import AsyncWebCrawler

async def crawl_single(url: str):
    async with AsyncWebCrawler() as crawler:
        return await crawler.arun(url=url)

async def main():
    # 要爬取的URL列表
    urls = [
        "https://crawl4ai.com",
        "https://example.com",
        "https://www.python.org"
    ]
    
    # 创建异步任务列表
    tasks = [crawl_single(url) for url in urls]
    
    # 使用gather并发执行并收集结果
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 处理结果
    for url, result in zip(urls, results):
        if isinstance(result, Exception):
            print(f"爬取 {url} 失败: {str(result)}")
        else:
            print(f"=== {url} 的爬取结果 ===")
            print(result.markdown[:500] + "...")  # 只打印前500字符避免刷屏

asyncio.run(main())