from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8009/v1", api_key="EMPTY")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "写一句随机的名言警句。"},
]


def send_request(i):
    try:
        response = client.chat.completions.create(
            model="/AIdata/JW/Qwen3-30B-A3B",
            messages=messages,
            max_tokens=10000,
        )
        return f"[{i}] {response.choices[0].message.content}"
    except Exception as e:
        return f"[{i}] 请求失败: {e}"


# 10 个线程并发，每个线程跑 100 次 = 总共 1000 次
with ThreadPoolExecutor(max_workers=100) as executor:
    futures = [executor.submit(send_request, i) for i in range(1000)]
    for future in as_completed(futures):
        print(future.result())
