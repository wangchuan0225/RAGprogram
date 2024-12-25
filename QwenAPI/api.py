from openai import OpenAI
import re
import json
from openai import OpenAI
def call_api(message):
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-714231764e0443e595ec60c7c7659f51",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': message}],
    )

    json_str = completion.model_dump_json()

    # 将 JSON 字符串解析为 Python 字典
    response_dict = json.loads(json_str)

    # 打印整个响应字典（用于调试）
    print(f"total_tokens: {response_dict['usage'].get('total_tokens')}")
    return response_dict['choices'][0].get('message', {}).get('content')

sentence=input("输入要查询的句子：")
message="请从以下描述中提取关键词，确保这些词汇能够精准反映查询意图：- 提取名词、形容词和动词，这些词汇应直接与查询内容相关。" \
        f"- 避免提取过于通用的或不太可能作为查询关键词的词（如“一个”、“我想”等），如果有介词（例如在，at，in等）后面跟了名词，将介词和名词合并为一个关键短语。" \
        f"- 如果有名词动词在句中相邻且连起来意思通顺，可以作为短语，则将二者合并成为一个关键词短语返回" \
        f"- 可以对加上对句子中提取出来的关键词进行合理联想，但都要是意思相近或者同一方面的词（例如输入ai可以联想到大模型，计算机视觉等），只用联想三四个就行。如果关键词是英文，联想词也要是英文" \
        f"将联想词和关键词都返回。用户输入：{sentence}返回格式：提取关键词：[关键词1,关键词2,关键词3] 联想关键词：[关键词1,关键词2]"\

responce=call_api(message)
match = re.findall(r'$[.*?]$', responce)
# 将匹配到的结果分割成单独的元素，并去除可能存在的多余空格
match_array = re.findall(r'\[(.*?)\]', responce)

# 初始化两个列表来保存提取的内容
extraction_keywords = []
association_keywords = []

# 处理匹配结果
if len(match_array) == 2:
    # 去掉方括号并分割字符串变成列表
    extraction_keywords = match_array[0].split(', ')
    association_keywords = match_array[1].split(', ')

# 输出结果
print("提取关键词列表:", extraction_keywords)
print("联想关键词列表:", association_keywords)