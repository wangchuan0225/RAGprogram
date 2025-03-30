import faiss
import numpy as np
import torch
import spacy
from sklearn.preprocessing import normalize
import os
import json
from transformers import AutoModel, AutoTokenizer
nlp = spacy.load("en_core_web_sm")
model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name)
# 设定索引文件路径
from Semantic.AdaptivePCA import tokenizer

EMBEDDINGS_DIR = "../semantic_vector/"
TEXTS_DIR = "../output/"
FAISS_INDEX_PATH = "../semantic_vector/faiss_sentence.index"
TEXT_DATA_PATH = "../output/faiss_sentences.json"



def load_all_sentence_vectors():
    """ 加载所有 `.npy` 文件中的句子向量，并存储对应的句子文本 """
    all_embeddings = []
    all_sentences = []

    import re

    file_list = sorted(
        [f for f in os.listdir(EMBEDDINGS_DIR) if f.startswith("embeddings") and f.endswith(".npy")],
        key=lambda x: int(re.search(r'\d+', x).group())  # 提取数字部分排序
    )
    for file in file_list:
        # 加载语义向量
        embeddings_path = os.path.join(EMBEDDINGS_DIR, file)
        print(embeddings_path)
        sentence_embeddings = np.load(embeddings_path)
        all_embeddings.append(sentence_embeddings)

        # 获取对应的 JSON 文件名
        idx = file.split("embeddings")[1].split(".npy")[0]  # 提取索引
        sentences_path = os.path.join(TEXTS_DIR, f"processed_file_{idx}.json")
        print(sentences_path)
        if os.path.exists(sentences_path):
            with open(sentences_path, "r", encoding="utf-8") as f:
                sentences = json.load(f)
                all_sentences.extend(sentences)

    # 合并所有 `.npy` 文件的向量
    if all_embeddings:
        all_embeddings = np.vstack(all_embeddings)
    else:
        raise ValueError("No `.npy` embeddings found!")

    # 归一化向量，使 FAISS 余弦相似度计算更稳定
    all_embeddings = normalize(all_embeddings, axis=1)

    return all_embeddings, all_sentences


def build_faiss_index(vectors):
    """ 构建全局 FAISS 索引 """
    dimension = vectors.shape[1]  # 获取向量维度
    index = faiss.IndexFlatIP(dimension)  # 余弦相似度（内积）
    index.add(vectors)  # 添加所有语义向量
    return index


def save_faiss_index(index, sentence_texts):
    """ 保存 FAISS 索引和句子文本 """
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(TEXT_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(sentence_texts, f, ensure_ascii=False, indent=4)
    print("✅ FAISS 索引 & 句子文本已保存")


def load_faiss_index():
    """ 加载 FAISS 索引和句子文本 """
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(TEXT_DATA_PATH):
        print("加载已有的 FAISS 索引...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(TEXT_DATA_PATH, "r", encoding="utf-8") as f:
            sentence_texts = json.load(f)
        return index, sentence_texts
    else:
        print("未找到索引文件，重新构建...")
        return None, None





def query_match(question, top_n=10):
    """ 直接使用 FAISS 全局索引查找最相似的句子 """

    # **2️⃣ 计算查询句子的向量**
    query_input = tokenizer([question], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        query_embedding = model(**query_input).last_hidden_state.mean(dim=1).numpy()

    # 归一化查询向量，使其与索引匹配
    query_embedding = normalize(query_embedding, axis=1)

    # **3️⃣ 使用 FAISS 进行最近邻搜索**
    _, top_indices = faiss_index.search(query_embedding, top_n)

    # **4️⃣ 获取匹配的句子**
    final_top_sentences = [all_sentence_texts[i] for i in top_indices[0]]


    return final_top_sentences
if __name__=='__main__':
    faiss_index, all_sentence_texts = load_faiss_index()
    if faiss_index is None:
        # 需要重新加载语义向量并构建索引
        all_sentence_embeddings, all_sentence_texts = load_all_sentence_vectors()
        faiss_index = build_faiss_index(all_sentence_embeddings)
        save_faiss_index(faiss_index, all_sentence_texts)
    query="latino trump 2024"
    results = query_match(query, top_n=5)
    print("\n🔍 最相似的句子:")
    for i, sentence in enumerate(results):
        print(f"Top {i + 1}: {sentence}")

