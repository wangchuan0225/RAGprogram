import os
from utils.base import TOKEN_LIMIT,message
import time

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
from QwenAPI.api import return_answer
from utils.base import extract_number
import spacy
from PDFproduce.Lemmatization import split_sentences, load_from_json
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.neighbors import KernelDensity

from Semantic.SemanticVetor import load_sentences_by_indices
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.decomposition import PCA
import numpy as np
from pathlib import Path
pca_vector_cache = "../semantic_vector/adaptive_pca_vectors.npy"  # 存储所有 `.npy` 文件的均值向量
file_indices_cache = "../semantic_vector/adaptive_pca_file_indices.npy"  # 存储所有 `.npy` 文件编号
pca_vector_cache1 = "../semantic_vector/adaptive_pca_vectors.npy"  # 存储所有 `.npy` 文件的均值向量
file_indices_cache1 = "../semantic_vector/adaptive_pca_file_indices.npy"  # 存储所有 `.npy` 文件编号


def load_pca_vectors():
    """ 加载 `.npy` 文件的 PCA 代表向量 """
    pca_vectors = np.load(pca_vector_cache1)
    file_indices = np.load(file_indices_cache1)
    return pca_vectors, file_indices


def append_new_pca_vector(new_embedding_path):
    """ 计算新 embedding 的 PCA 并追加到已有的 PCA 向量缓存 """

    # 1️⃣ 加载已有 PCA 向量和索引
    if Path(pca_vector_cache).exists() and Path(file_indices_cache).exists():
        existing_pca_vectors = np.load(pca_vector_cache)
        existing_file_indices = np.load(file_indices_cache).tolist()
    else:
        raise FileNotFoundError("PCA 缓存文件不存在，请先计算已有的 PCA 结果！")

    # 2️⃣ 读取新 embedding
    new_embeddings = np.load(new_embedding_path)
    new_index = extract_number(new_embedding_path)

    # 3️⃣ 计算 PCA
    explained_variance_ratio = 0.99  # 目标方差比例
    pca = PCA(n_components=min(new_embeddings.shape[0], new_embeddings.shape[1]))  # 设最大
    pca.fit(new_embeddings)
    total_variance = np.cumsum(pca.explained_variance_ratio_)
    optimal_dim = np.searchsorted(total_variance, explained_variance_ratio) + 1

    # 4️⃣ 重新降维到最佳维度
    pca = PCA(n_components=optimal_dim)
    reduced_embeddings = pca.fit_transform(new_embeddings)

    # 5️⃣ 取第一主成分的均值作为代表向量
    pca_vector = np.mean(reduced_embeddings, axis=0)

    # 6️⃣ 还原回原始维度（768）
    final_vector = pca.inverse_transform(pca_vector)

    # 7️⃣ 追加新的 PCA 结果
    updated_pca_vectors = np.vstack([existing_pca_vectors, final_vector])
    updated_file_indices = existing_file_indices + [new_index]

    # 8️⃣ 存储更新后的 `.npy`
    np.save(pca_vector_cache, updated_pca_vectors)
    np.save(file_indices_cache, np.array(updated_file_indices))

    print(f"PCA vector for {new_embedding_path} appended successfully!")


def compute_and_store_pca_vectors(embeddings_dir):
    """ 计算所有 `.npy` 文件的 PCA 第一主成分并存储 """
    embeddings_paths = sorted(Path(embeddings_dir).glob('embeddings*.npy'), key=extract_number)
    file_indices = [extract_number(path) for path in embeddings_paths]

    pca_vectors = []

    for emb_path in embeddings_paths:
        embeddings = np.load(str(emb_path))  # 避免占用过多内存
        print(emb_path)
        # 1️⃣ 自适应选择 PCA 维度（保留 99% 信息）
        explained_variance_ratio = 0.99  # 目标方差比例
        pca = PCA(n_components=min(embeddings.shape[0], embeddings.shape[1]))  # 先设最大
        pca.fit(embeddings)
        total_variance = np.cumsum(pca.explained_variance_ratio_)
        optimal_dim = np.searchsorted(total_variance, explained_variance_ratio) + 1

        # 2️⃣ 重新降维到最佳维度
        pca = PCA(n_components=optimal_dim)
        reduced_embeddings = pca.fit_transform(embeddings)

        # 3️⃣ 取第一主成分的均值作为代表向量
        pca_vector = np.mean(reduced_embeddings, axis=0)

        # 4️⃣ 还原回原始维度（768）
        final_vector = pca.inverse_transform(pca_vector)
        pca_vectors.append(final_vector)

    # 5️⃣ 存储 `.npy`
    np.save(pca_vector_cache1, np.vstack(pca_vectors))
    np.save(file_indices_cache1, np.array(file_indices))
    print("PCA vectors saved successfully!")

def query_match_with_pca_adaptive_vectors(question, top_n=2):
    """ 使用自适应pca向量快速匹配 `.npy` 文件 """
    nlp = spacy.load("en_core_web_sm")
    model_name = "bert-base-uncased"
    model = AutoModel.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  # 设置为评估模式
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    start=time.time()
    # **1️⃣ 加载预计算的均值向量**
    mean_vectors, file_indices = load_pca_vectors()

    # **2️⃣ 计算查询句子的向量**
    query_input = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        query_embedding = model(**query_input).last_hidden_state.mean(dim=1).numpy()

    # **3️⃣ 计算查询句子与所有 `.npy` 文件均值向量的相似度**
    file_similarities = cosine_similarity(query_embedding, mean_vectors)[0]

    # **4️⃣ 取前 N 个最相关的 `.npy` 文件**
    top_indices = np.argsort(file_similarities)[::-1][:top_n]
    top_files = file_indices[top_indices]
    end=time.time()
    print(f"find sentences use{end-start}")
    print(top_files)
    sentence_embeddings_list = []
    sentence_texts = []
    for idx in top_files:
        embeddings_path = f"../semantic_vector/embeddings{idx}.npy"
        sentences_path = f"../output/processed_file_{idx}.json"  # 假设有对应的句子文件

        # 加载句子向量
        sentence_embeddings = np.load(embeddings_path)
        sentence_embeddings_list.append(sentence_embeddings)

        # 加载句子文本（假设 `load_from_json` 可以正确解析）
        sentences = load_from_json(sentences_path)
        sentence_texts.extend(sentences)

    # **6️⃣ 计算查询句子与这些句子的相似度**
    if sentence_embeddings_list:
        # 合并所有 `.npy` 文件中的句子向量
        all_sentence_embeddings = np.vstack(sentence_embeddings_list)

        # 计算相似度
        final_similarities = cosine_similarity(query_embedding, all_sentence_embeddings)[0]

        # 获取前 `top_n` 个最相似句子的索引
        final_top_indices = np.argsort(final_similarities)[::-1][:100]
        final_top_sentences = [sentence_texts[i] for i in final_top_indices]
        # **输出结果**

        message1=''
        for sentence in final_top_sentences:
            if len(message1) > TOKEN_LIMIT:
                break
            else:
                message1 += sentence
        prompt = message + message1.strip() + " 问题是：" + question
        return return_answer(prompt)
if __name__ == "__main__":
    #compute_and_store_pca_vectors('../semantic_vector')

    start=time.time()
    query_match_with_pca_adaptive_vectors("In the realm of econophysics, models are developed to explore both equivalent market exchanges and nonequivalent mutual aid exchanges. These models  simulate traditional market transactions where goods and services are exchanged at equivalent value, as well as non-traditional mutual aid scenarios where exchanges may not be of equal value but contribute significantly to wealth redistribution. By analyzing these dynamics, researchers can better understand how different types of exchanges impact economic flow and wealth inequality. Notably, nonequivalent mutual aid exchanges often play a crucial role in mitigating wealth disparities by promoting more equitable resource distribution. This research highlights the importance of diverse exchange mechanisms in shaping economic systems and offers insights into potential policies for reducing inequality and fostering a more balanced economic environment.")
    end=time.time()
    print(f"use {end-start}s")
