import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
import os
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict
from pathlib import Path
from utils.base import load_from_json
from MatchingWay.BM25Matching import TOKEN_LIMIT
from QwenAPI.api import return_key_word,return_answer
from utils.base import split_sentences
all_pdf_content=Path('../output')
topic_pakage="../topic"
nlp = spacy.load("en_core_web_sm")
sentence_internal_dict = "../inverted_index"
message = "我会给你一大段话关于所查询的内容以及，所要查询的问题。总结所给的话中的数据信息和文字信息，" \
          f"并结合你自身所知道的信息输出结果，结果中要包括所给的话中的信息，所给的文段中可能有不相关的句子，可以进行排除。如果相关信息较少，不要在答案中说出来，自行补充即可。如果有所不清楚或者哪里需要用户提供更多细节，可以在回答的最后一个板块中加入对用户追问，是对用户问题的追问，问题中不清楚的点都可以追问。所给文段："




'''
词型还原函数
'''
def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

'''
句型划分函数，将每个传入的文本进行按句子划分得到每句
'''
def solid_size(content):
    for i in range(0, len(content), 500000):
        yield content[i:i + 500000]


def merge_inverted_indices(list_inverted_index):
    """
    对于一些文段需要分开为多个部分进行处理，在处理后会得到多个倒排索引合并多个倒排索引，确保同一个词条只保留一个，并合并所有句子索引。
    """
    merged_index = {}

    for index in list_inverted_index:
        for word, sentence_indices in index.items():
            if word not in merged_index:
                merged_index[word] = set(sentence_indices)  # 使用集合避免重复
            else:
                merged_index[word].update(sentence_indices)  # 合并句子索引

    # 将集合转换回列表以便保存为 JSON
    for word in merged_index:
        merged_index[word] = sorted(list(merged_index[word]))  # 按索引排序

    return merged_index
'''
保存倒排索引到JSON
'''
def save_in_json(internal_index,file_name):
    with open(file_name, 'w') as f:
        json.dump(internal_index, f)

'''
从JSON加载倒排索引
'''


'''
构建基于过滤词汇的倒排索引
'''
def build_filtered_inverted_index(sentences):
    inverted_index = defaultdict(list)

    for idx, sentence in enumerate(sentences):
        # 基于空格分词
        words = sentence.split()

        for word in words:
            # 转为小写
            word = word.lower()

            # 过滤条件：
            # 1. 必须包含字母（防止全是数字或符号的情况）
            # 2. 不包含无意义的符号（如 '67%'、'###'）
            # 3. 允许中间有连字符（如 'covid-19'）
            if re.match(r"^[a-zA-Z]+[a-zA-Z0-9\-]*[a-zA-Z0-9]+$", word):
                if idx not in inverted_index[word]:
                    inverted_index[word].append(idx)

    return inverted_index

'''
按句划分流程：读取文本、按句子分块，处理并构建倒排索引
'''
def sentence_create_basic(file_path,index):
    output_file = file_path
    # 读取文件内容
    with open(output_file, 'r') as f:
        content = f.read()
        if not content.strip():  # `strip` 去掉空白字符，确保内容确实为空
            f.close()  # 确保文件关闭后删除
            os.remove(output_file)
            return

    chunks=list(solid_size(content))
    # 词型还原
    list_inverted_index = []
    all_sentences=[]
    for chunk_idx, chunk in enumerate(chunks):
        # 词型还原
        lemmatized_content = lemmatize_text(chunk)

        # 按句分块
        sentences = split_sentences(lemmatized_content)

        # 保存分块后的句子到 JSON
        all_sentences.extend(sentences)
        # 构建倒排索引
        inverted_index = build_filtered_inverted_index(sentences)
        list_inverted_index.append(inverted_index)
    json_output_path = os.path.join(f"../temp_output/processed_file_{index}.json")
    save_in_json(all_sentences,json_output_path)
    json_output_path = os.path.join(f"../output/processed_file_{index}.json")
    save_in_json(all_sentences, json_output_path)
    # 删除原文件
    #os.remove(file_path)
    #print(f"Deleted original file: {file_path}")

    # 合并倒排索引
    merged_inverted_index = merge_inverted_indices(list_inverted_index)

    # 保存合并后的倒排索引到 JSON 文件
    sentence_internal_file = os.path.join(sentence_internal_dict, f"pdf_{index}_inverted_index.json")
    save_in_json(merged_inverted_index, sentence_internal_file)
    sentence_internal_file = os.path.join('../temp_output', f"pdf_{index}_inverted_index.json")
    save_in_json(merged_inverted_index, sentence_internal_file)

    # 输出倒排索引结果

'''
按主题进行分块
'''
def split_by_topic(n_topics):
    # 1. 分句
    pdf_path = all_pdf_content.glob("*.json")
    sentences = [load_from_json(f) for f in pdf_path]
    flat_sentences = [sentence.strip() for sublist in sentences for sentence in sublist if
                      isinstance(sentence, str) and sentence.strip()]

    if not flat_sentences:
        raise ValueError("No valid sentences found for vectorization. Please check your input files.")

    # 2. 使用TF-IDF向量化
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(flat_sentences)

    # 词汇表和 IDF 值
    vocabulary = vectorizer.vocabulary_
    idf_values = vectorizer.idf_.tolist()

    # 3. 使用 K-Means 进行主题聚类
    kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(tfidf_matrix)

    # 4. 按聚类结果分块
    topics = defaultdict(list)
    for i, cluster_id in enumerate(clusters):
        topics[int(cluster_id)].append(flat_sentences[i])

    centroids = {}
    for topic_id in range(n_topics):
        indices = [i for i, cluster_id in enumerate(clusters) if cluster_id == topic_id]
        cluster_vectors = tfidf_matrix[indices]
        centroid = cluster_vectors.mean(axis=0).A1
        centroids[topic_id] = centroid.tolist()

    # 5. 保存到文件
    if not os.path.exists(topic_pakage):
        os.makedirs(topic_pakage)

    # 存储每个主题的数据
    for topic_id, topic_sentences in topics.items():
        topic_data = {
            "topic_id": topic_id,
            "sentences": topic_sentences,
            "centroid": centroids[topic_id],
            "vocabulary": vocabulary,
            "idf_values": idf_values  # 存储 IDF 值
        }
        topic_file_path = os.path.join(topic_pakage, f"topic_{topic_id}.json")
        with open(topic_file_path, "w") as f:
            json.dump(topic_data, f, indent=4)

    print(f"Topics saved in directory: {topic_pakage}")
    return topics
def match_question_to_topics_directory(question):
    """
    根据输入问题，遍历 topic_pakage 目录中所有 JSON 文件，与中心向量进行匹配，找到最相似的主题。

    参数：
        question (str): 用户输入的问题。
        topic_pakage (str): 存储各主题 JSON 文件的目录。
        lemmatize_text (function): 用于对输入问题进行词型还原的函数。

    返回：
        dict: 包含最佳匹配主题 ID、相似度、主题中的句子。
    """
    if not os.path.exists(topic_pakage):
        raise ValueError("The topic directory does not exist")
    extract_keyword, associate_keyword = return_key_word(question)

    lemmatized_question = lemmatize_text(question)

    best_match = {
        "topic_id": None,
        "similarity": -1,
        "sentences": []
    }

    for file_name in os.listdir(topic_pakage):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(topic_pakage, file_name)

        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON file: {file_path}")
            continue

        topic_id = data.get("topic_id")
        sentences = data.get("sentences", [])
        centroid = np.array(data.get("centroid"))
        vocabulary = data.get("vocabulary")
        idf_values = data.get("idf_values")

        if not centroid.any() or not sentences or not vocabulary or not idf_values:
            print(f"Skipping topic {topic_id} due to missing data.")
            continue

        # ** 重新初始化 TF-IDF Vectorizer **
        vectorizer = TfidfVectorizer(stop_words="english", vocabulary=vocabulary)
        vectorizer.fit([lemmatized_question])  # 先 `fit`，避免 `NotFittedError`
        vectorizer.idf_ = np.array(idf_values)  # 加载存储的 IDF 值

        # ** 转换问题向量 **
        question_vector = vectorizer.transform([lemmatized_question])

        centroid_vector = centroid.reshape(1, -1)
        similarity = cosine_similarity(question_vector, centroid_vector)[0][0]

        if similarity > best_match["similarity"]:
            best_match = {
                "topic_id": topic_id,
                "similarity": similarity,
                "sentences": sentences
            }

    if best_match["topic_id"] is None:
        raise ValueError("No matching topics found.")
    sentences=list(best_match['sentences'])

    return sentences,extract_keyword
def topic_sentences_match(sentences,keywords):
    pattern = re.compile(r'\b(' + "|".join(map(re.escape, keywords)) + r')\b')

    result = []
    for s in sentences:
        matches = pattern.findall(s)
        count = len(matches)
        if count > 0:
            result.append((s, count))

    result.sort(key=lambda x: x[1], reverse=True)
    res=[]
    for sentence, count in result:
        res.append(sentence)
    return res
def topic_match(question):
    start_time=time.time()
    sentences,keyw=match_question_to_topics_directory(question)
    sentences=topic_sentences_match(sentences,keyw)
    end_time=time.time()
    print(f"find sentence use{end_time-start_time}")
    message1=message
    for sentence in sentences:
        if len(message1)>TOKEN_LIMIT:
            break
        else:
            message1+=sentence
    print(message1)
    #print(len(message1))
    return return_answer(message1)

'''
if __name__=="__main__":
    pdf_file=list(all_pdf_content.glob("*.txt"))
    # 按照文件名中的序号排序
    pdf_file = sorted(pdf_file, key=extract_number)

    index = 0
    for file in pdf_file:
        print(file)
        sentence_create_basic(file, index)
        index += 1
'''









if __name__=="__main__":
    #topics=split_by_topic(60)
    start_time = time.time()
    topic_match("detail about MacBook,hardware,screen,dashboard")
    end_time = time.time()  # 记录结束时间
    print("运行时间: ", end_time - start_time, "秒")

#Latino registered voters' views on whether they want Trump to remain a national political figure and potentially run for president in 2024
#detail about lenovo thinkpad battery,and other part, like screen, cpu
#chromosome in cell genome




