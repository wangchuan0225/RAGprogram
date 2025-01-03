from lib2to3.fixes.fix_input import context
from pathlib import Path
import spacy
from collections import defaultdict
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict
nlp = spacy.load("en_core_web_sm")
sentence_internal_dict = "../output/sentence_internal_json.json"
sentences_table="../output/sentences_table.json"
'''
词型还原函数
'''
def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

'''
句型划分函数，将每个传入的文本进行按句子划分得到每句
'''
def split_sentences(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

'''
保存倒排索引到JSON
'''
def save_in_json(internal_index,file_name):
    with open(file_name, 'w') as f:
        json.dump(internal_index, f)

'''
从JSON加载倒排索引
'''
def load_from_json(file_path):
    with open(file_path, 'r') as file:
        inverted_index = json.load(file)
    return inverted_index

'''
构建基于过滤词汇的倒排索引
'''
def build_filtered_inverted_index(sentences):
    inverted_index = defaultdict(list)
    for idx, sentence in enumerate(sentences):
        doc = nlp(sentence)
        for token in doc:
            if not token.is_stop and token.is_alpha and token.pos_ in {"NOUN", "VERB"}:
                word = token.lemma_.lower()
                if idx not in inverted_index[word]:
                    inverted_index[word].append(idx)
    return inverted_index

'''
按句划分流程：读取文本、按句子分块，处理并构建倒排索引
'''
def sentence_create_basic():
    content = ''
    output_file = '../output/output_result.txt'
    list_inverted_index = []

    # 读取文件内容
    with open(output_file, 'r') as f:
        content = f.read()

    # 词型还原
    lemmatized_content = lemmatize_text(content)

    # 写回词型还原后的内容
    with open(output_file, 'w') as f:
        f.write(lemmatized_content)

    # 按句分块
    sentences = split_sentences(lemmatized_content)
    #print("分句结果：", sentences)
    save_in_json(sentences,sentences_table)
    # 构建倒排索引（过滤停用词，仅保留名词和动词）
    inverted_index = build_filtered_inverted_index(sentences)
    list_inverted_index.append(inverted_index)

    # 保存到JSON文件
    save_in_json(list_inverted_index, sentence_internal_dict)

    # 输出倒排索引结果
    print("倒排索引：", list_inverted_index)

'''
按主题进行分块
'''
def split_by_topic(n_topics):
    # 1. 分句
    content_path='../output/output_result.txt'
    with open(content_path,'r') as f:
        text=f.read()
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    # 2. 对每句进行词型还原
    lemmatized_sentences = [lemmatize_text(sentence) for sentence in sentences]

    # 3. 使用TF-IDF向量化
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(lemmatized_sentences)

    # 4. 使用KMeans进行主题聚类
    kmeans = KMeans(n_clusters=n_topics, random_state=42)  # 此处 n_topics 来自函数参数
    clusters = kmeans.fit_predict(tfidf_matrix)

    # 5. 根据聚类结果分块
    topics = defaultdict(list)
    for i, cluster_id in enumerate(clusters):
        topics[cluster_id].append(sentences[i])
    inverted_index = build_filtered_inverted_index(sentences)

    # 保存到JSON文件
    save_in_json(inverted_index, sentence_internal_dict)
    print(inverted_index)

    return topics
def search_by_keyword(keyword, topics):
    """
    根据关键词进行模糊匹配，返回对应主题和句子
    """
    inverted_index=load_from_json(sentence_internal_dict)
    sentences=load_from_json(sentences_table)
    # 对关键词进行词形还原
    keyword = lemmatize_text(keyword)
    results = []

    for word in inverted_index:
        if keyword in word:  # 模糊匹配
            for idx in inverted_index[word]:
                # 找到对应句子的主题
                for topic_id, topic_sentences in topics.items():
                    if sentences[idx] in topic_sentences:
                        results.append((topic_id, sentences[idx]))

    # 去重并按主题分组
    unique_results = defaultdict(list)
    for topic_id, sentence in set(results):
        unique_results[topic_id].append(sentence)

    return unique_results

if __name__=="__main__":
    topics=split_by_topic(50)
    keyword = "trump"
    results = search_by_keyword(keyword,  topics)
    for topic_id, sentences in results.items():
        print(f"\n主题 {topic_id}:")
        for sentence in sentences:
            print(f"  - {sentence}")



