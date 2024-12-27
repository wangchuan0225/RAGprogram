from pathlib import Path
import spacy
from collections import defaultdict
import json

nlp = spacy.load("en_core_web_sm")
internal_dict = "../output/internal_json.json"
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
主流程：读取文本、处理并构建倒排索引
'''
def create_basic():
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
    save_in_json(list_inverted_index, internal_dict)

    # 输出倒排索引结果
    print("倒排索引：", list_inverted_index)
