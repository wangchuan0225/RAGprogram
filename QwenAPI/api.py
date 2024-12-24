from openai import OpenAI
import os
import json
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-68aa756d1aac410797b7adee53499e14",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
    model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': '我想打人'}],
)

json_str = completion.model_dump_json()

# 将 JSON 字符串解析为 Python 字典
response_dict = json.loads(json_str)

# 打印整个响应字典（用于调试）
print(response_dict)
print(response_dict['choices'][0].get('message', {}).get('content'))