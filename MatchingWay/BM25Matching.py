import time
from utils.base import message,TOKEN_LIMIT
from utils.base import extract_number
import json
from PDFproduce.Lemmatization import load_from_json
from QwenAPI.api import return_key_word, return_answer
from collections import defaultdict
from pathlib import Path
from utils.base import read_pdf_sentence
pdf_sentences_path=Path("../output")
inverted_path=Path("../inverted_index")

message1=""
'''
关键词匹配
'''
def real_keyword_match(question):
    extract_keyword, associate_keyword = return_key_word(question)
    # 读取所有文章的句子列表，以及倒排索引列表
    sentences_list = list(pdf_sentences_path.glob("*.txt"))
    inverted_index_list = list(inverted_path.glob("*.json"))
    # 以下为按照关键词匹配的句子进行拼接并生成答案
    # 先提取共有的，包含尽量多的关键词的句子
    sentence_keyword_count = defaultdict(int)  # 存储句子及其关键词匹配数量

    for index in range(len(inverted_index_list)):
        now_inverted = inverted_index_list[index]
        now_pdf = sentences_list[index]
        print(now_pdf)
        # 读取当前文件的句子和倒排索引
        now_sentences = read_pdf_sentence(now_pdf)
        internal_list = load_from_json(now_inverted)

        for keyw in extract_keyword:
            if keyw in internal_list[0]:
                # 获取关键词对应的句子索引列表
                sentence_indices = internal_list[0][keyw]
                # 根据索引记录句子关键词匹配数量
                for i in sentence_indices:
                        sentence = now_sentences[i]
                        sentence_keyword_count[sentence] += 1
        print(f"sentence:{sentence_keyword_count.keys()}")

    # 根据关键词匹配数量排序句子（从多到少）
    sorted_sentences = sorted(sentence_keyword_count.keys(), key=lambda s: sentence_keyword_count[s], reverse=True)
    print(sorted_sentences)
    # 拼接包含关键词数量最多的句子到 message1
    message1 = ""
    for sentence in sorted_sentences:
        if len(message1) > 10000:
            # 控制长度
            print("sentences is more than 10000 word.\n")
            break
        else:
            message1 = message1 + sentence
    prompt = message + message1 + " 问题是：" + question


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
def optimized_keyword_match(question):
    start_time=time.time()
    extract_keyword, associate_keyword = return_key_word(question)
    #extract_keyword=['laptop', 'screen', 'black', 'issue', 'resolve']
    # 获取所有文件路径（按文件名排序确保顺序一致）
    sentences_files = sorted(pdf_sentences_path.glob("processed*.json"), key=extract_number)
    inverted_files = sorted(inverted_path.glob("*.json"), key=extract_number)

    # 确保文件数量一致
    assert len(sentences_files) == len(inverted_files), "句子文件和倒排索引文件数量不匹配！"
    print(len(sentences_files),len(inverted_files))
    # 综合评分存储 (score, sentences_file, sentences, inverted_index)
    article_scores = []
    total=0
    # 遍历每篇文章
    for sentences_file, inverted_file in zip(sentences_files, inverted_files):
        sentences = load_json(sentences_file)  # 加载句子
        inverted_index = load_json(inverted_file)  # 加载倒排索引

        # 1. 统计关键词覆盖数量
        matching_keywords = set()
        keyword_total_count = 0

        for keyw in extract_keyword:
            if keyw in inverted_index:
                matching_keywords.add(keyw)  # 记录出现的关键词
                keyword_total_count += len(inverted_index[keyw])  # 累加关键词出现总次数

        # 关键词覆盖数量和总次数
        coverage_count = len(matching_keywords)

        # 2. 综合评分计算 (70% 关键词覆盖 + 30% 总出现次数)
        score = 0.8 * coverage_count + 0.05 * keyword_total_count
        total+=score
        article_scores.append((score, sentences_file, sentences, inverted_index))

    # 按综合评分从高到低排序文章
    article_scores.sort(reverse=True, key=lambda x: x[0])

    # 输出最相关的文章
    most_matching_article = article_scores[0] if article_scores else None
    if most_matching_article:
        print(
            f"最相关的文章是：{most_matching_article[1]}，综合评分：{most_matching_article[0]/total*100}"
        )
    end_time=time.time()
    print(f"find sentences use {end_time-start_time}s")
    # 拼接最相关的句子
    message1 = ""
    sentence_keyword_count = defaultdict(int)

    # 从最相关的文章开始寻找句子
    for _, _, sentences, inverted_index in article_scores:
        # 统计每个句子的关键词匹配数量
        for idx, sentence in enumerate(sentences):
            matching_keywords = 0
            for keyw in extract_keyword:
                if keyw in inverted_index and idx in inverted_index[keyw]:
                    matching_keywords += 1
            sentence_keyword_count[sentence] = matching_keywords

        # 按句子匹配数量排序
        sorted_sentences = sorted(
            sentence_keyword_count.keys(),
            key=lambda s: sentence_keyword_count[s],
            reverse=True,
        )

        # 优先从当前文章中拼接句子
        for sentence in sorted_sentences:
            if len(message1) > TOKEN_LIMIT:  # 控制拼接长度
                break
            if sentence_keyword_count[sentence] > 0:  # 只拼接匹配度大于 0 的句子
                message1 += sentence + " "

        # 如果拼接的内容已经足够，退出
        if len(message1) > TOKEN_LIMIT:
            break

    # 构造最终 prompt
    prompt = message + message1.strip() + " 问题是：" + question

    # Latino registered voters' views on whether they want Trump to remain a national political figure and potentially run for president in 2024
    # some detail about macbook screen battery touch board
    # some detail about econophysics models equivalent market exchanges redistribution nonequivalent mutual aid exchanges wealth inequality economic flow

    '''
    In the realm of econophysics, models are developed to explore both equivalent market exchanges and nonequivalent mutual aid exchanges. These models 
    simulate traditional market transactions where goods and services are exchanged at equivalent value, as well as non-traditional mutual aid scenarios 
    where exchanges may not be of equal value but contribute significantly to wealth redistribution. By analyzing these dynamics, researchers can better 
    understand how different types of exchanges impact economic flow and wealth inequality. Notably, nonequivalent mutual aid exchanges often play a crucial 
    role in mitigating wealth disparities by promoting more equitable resource distribution. This research highlights the importance of diverse exchange mechanisms 
    in shaping economic systems and offers insights into potential policies for reducing inequality and fostering a more balanced economic environment.
    '''
    return return_answer(prompt)



def keyword_match(question):
    extract_keyword, associate_keyword = return_key_word(question)
    # 读取所有文章的句子列表，以及倒排索引列表
    sentences_list = list(pdf_sentences_path.glob("*.json"))
    inverted_index_list = list(inverted_path.glob("*.json"))
    # 以下为按照关键词匹配的句子进行拼接并生成答案
    # 先提取共有的，包含尽量多的关键词的句子
    sentence_keyword_count = defaultdict(int)  # 存储句子及其关键词匹配数量

    for index in range(len(inverted_index_list)):
        now_inverted = inverted_index_list[index]
        now_pdf = sentences_list[index]
        print(f"Processing file: {now_pdf}")

        # 读取当前文件的句子和倒排索引
        content_list = load_from_json(now_inverted)  # 假设此函数返回句子列表
        internal_list = load_from_json(now_inverted)  # 假设此函数返回倒排索引

        if not content_list or not internal_list:
            print(f"Warning: Empty content_list or internal_list for file {now_pdf}")
            continue

        # 初始化集合交集
        matching_indices = None

        # 计算关键词的索引交集
        for keyw in extract_keyword:
            if keyw in internal_list[0]:
                if matching_indices is None:
                    matching_indices = set(internal_list[0][keyw])
                else:
                    matching_indices &= set(internal_list[0][keyw])  # 求交集

        # 如果没有匹配到任何句子，跳过当前文件
        if not matching_indices:
            print(f"No matching sentences found for file: {now_pdf}")
            continue

        # 更新匹配结果的句子
        for idx in matching_indices:
            if idx >= len(content_list):
                print(f"Index {idx} is out of range for content_list with length {len(content_list)}")
                continue
            sentence = content_list[idx].strip()
            sentence_keyword_count[sentence] += 1

        # 输出匹配的句子及其计数
    print(f"sentence: {sentence_keyword_count.keys()}")

    # 根据关键词匹配数量排序句子（从多到少）
    sorted_sentences = sorted(
        sentence_keyword_count.keys(),
        key=lambda s: sentence_keyword_count[s],
        reverse=True
    )
    print(sorted_sentences)

    # 拼接包含关键词数量最多的句子到 message1
    message1 = ""
    for sentence in sorted_sentences:
        if len(message1) > 10000:
            print("sentences is more than 10000 words.\n")
            break
        message1 += sentence

    # 构造最终的 prompt
    prompt = message + message1 + " 问题是：" + question
    print(prompt)
if __name__ == '__main__':
    #question=input("输入问题：")
    start_time=time.time()
    question='In the realm of econophysics, models are developed to explore both equivalent market exchanges and nonequivalent mutual aid exchanges. These models simulate traditional market transactions where goods and services are exchanged at equivalent value, as well as non-traditional mutual aid scenarios  where exchanges may not be of equal value but contribute significantly to wealth redistribution. By analyzing these dynamics, researchers can better understand how different types of exchanges impact economic flow and wealth inequality. Notably, nonequivalent mutual aid exchanges often play a crucial role in mitigating wealth disparities by promoting more equitable resource distribution. This research highlights the importance of diverse exchange mechanisms in shaping economic systems and offers insights into potential policies for reducing inequality and fostering a more balanced economic environment.'
    print(optimized_keyword_match(question))
    end_time = time.time()  # 记录结束时间
    print("运行时间: ", end_time - start_time, "秒")
    #topic_match(question)






