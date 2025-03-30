from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from pathlib import Path
import spacy
import torch
from sklearn.metrics.pairwise import cosine_similarity

from MatchingWay.BM25Matching import TOKEN_LIMIT
from QwenAPI.api import return_answer
from Semantic.AdaptivePCA import file_indices_cache
from utils.base import extract_number,extract_keywords
from transformers import AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from PDFproduce.Lemmatization import load_from_json
import jieba
from PDFproduce.Lemmatization import message
nlp = spacy.load("en_core_web_sm")
model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()  # 设置为评估模式
tokenizer = AutoTokenizer.from_pretrained(model_name)
kmean_vector_cache1 = "../semantic_vector/kmean_vectors.npy"  # 存储所有 `.npy` 文件的均值向量
file_indices_cache1 = "../semantic_vector/kmean_file_indices.npy"  # 存储所有 `.npy` 文件编号
kmean_vector_cache="../semantic_vector/kmean_vectors.npy"
file_indices_cache="../semantic_vector/kmean_file_indices.npy"




def find_best_k(embeddings, max_k=10):
    """ 使用肘部法则（Elbow Method）自动选择 K """
    n_samples = len(embeddings)

    # ✅ 如果样本数小于 2，直接返回 1
    if n_samples < 2:
        return 1

    possible_k = list(range(2, min(max_k, n_samples)))  # K 值范围

    # ✅ 确保 possible_k 不是空的
    if not possible_k:
        return 1

    sse = []

    for k in possible_k:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        sse.append(kmeans.inertia_)  # 计算 SSE（误差）

    # ✅ 确保 sse 至少有 2 个值
    if len(sse) < 2:
        return possible_k[0]  # 选择最小的 K 值

    k_diffs = np.diff(sse)  # 计算一阶导数

    # ✅ 确保 k_diffs 不是空的
    if len(k_diffs) == 0:
        return possible_k[0]

    k_best = possible_k[np.argmin(k_diffs) + 1]  # 找到拐点

    return k_best


def compute_and_store_pca_kmeans_vectors(embeddings_dir):
    """ 计算所有 `.npy` 文件的 PCA + K-Means 代表向量并存储 """
    embeddings_paths = sorted(Path(embeddings_dir).glob('embeddings*.npy'), key=extract_number)
    file_indices = [extract_number(path) for path in embeddings_paths]

    kmeans_vectors = []

    for emb_path in embeddings_paths:
        embeddings = np.load(str(emb_path), mmap_mode="r")
        n_samples, n_features = embeddings.shape

        # ✅ Check for empty embeddings
        if n_samples == 0:
            print(f"Skipping {emb_path}: empty embeddings")
            continue

        print(emb_path)

        # ✅ Ensure PCA components are valid
        n_components = max(1, min(10, n_samples, n_features))

        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)

        # ✅ Prevent empty input for K-Means
        if len(reduced_embeddings) == 0:
            print(f"Skipping {emb_path}: reduced embeddings empty after PCA")
            continue

        optimal_k = find_best_k(reduced_embeddings, max_k=10)
        print(f"Optimal K for {emb_path}: {optimal_k}")

        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans.fit(reduced_embeddings)
        cluster_centroids = kmeans.cluster_centers_

        cluster_weights = np.bincount(kmeans.labels_) / n_samples
        weighted_centroid = np.average(cluster_centroids, axis=0, weights=cluster_weights)

        final_vector = pca.inverse_transform(weighted_centroid)
        kmeans_vectors.append(final_vector)

    if kmeans_vectors:
        np.save(kmean_vector_cache1, np.vstack(kmeans_vectors))
        np.save(file_indices_cache1, np.array(file_indices))
        print("PCA + Adaptive K-Means vectors saved successfully!")
    else:
        print("No valid embeddings processed, skipping save.")
def append_new_pca_kmeans_vector(new_embedding_path):
    """ 计算新的 PCA + K-Means 向量并追加到已有的文件 """

    # 1️⃣ 读取已有的 PCA+K-Means 结果
    if Path(kmean_vector_cache).exists() and Path(file_indices_cache).exists():
        existing_kmeans_vectors = np.load(kmean_vector_cache)
        existing_file_indices = np.load(file_indices_cache).tolist()
    else:
        raise FileNotFoundError("PCA+K-Means 缓存文件不存在，请先计算已有的 PCA 结果！")

    # 2️⃣ 读取新 embedding 文件
    new_embeddings = np.load(new_embedding_path, mmap_mode="r")
    new_index = extract_number(new_embedding_path)

    n_samples, n_features = new_embeddings.shape

    # ✅ 确保 embedding 非空
    if n_samples == 0:
        print(f"Skipping {new_embedding_path}: empty embeddings")
        return

    print(f"Processing {new_embedding_path}...")

    # 3️⃣ 确定 PCA 维度
    n_components = max(1, min(10, n_samples, n_features))
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(new_embeddings)

    # ✅ 确保降维后数据不为空
    if len(reduced_embeddings) == 0:
        print(f"Skipping {new_embedding_path}: reduced embeddings empty after PCA")
        return

    # 4️⃣ 计算最佳 K 值
    optimal_k = find_best_k(reduced_embeddings, max_k=10)
    print(f"Optimal K for {new_embedding_path}: {optimal_k}")

    # 5️⃣ 进行 K-Means 聚类
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans.fit(reduced_embeddings)
    cluster_centroids = kmeans.cluster_centers_

    # 6️⃣ 计算加权质心
    cluster_weights = np.bincount(kmeans.labels_) / n_samples
    weighted_centroid = np.average(cluster_centroids, axis=0, weights=cluster_weights)

    # 7️⃣ 还原到原始维度
    final_vector = pca.inverse_transform(weighted_centroid)

    # 8️⃣ 追加到已有数据
    updated_kmeans_vectors = np.vstack([existing_kmeans_vectors, final_vector])
    updated_file_indices = existing_file_indices + [new_index]

    # 9️⃣ 保存 `.npy` 文件
    np.save(kmean_vector_cache, updated_kmeans_vectors)
    np.save(file_indices_cache, np.array(updated_file_indices))

    print(f"PCA + K-Means vector for {new_embedding_path} appended successfully!")
def load_pca_kde_vectors():
    """ 加载 `.npy` 文件的 PCA + KDE 代表向量 """
    kde_vectors = np.load(kmean_vector_cache1)
    file_indices = np.load(file_indices_cache1)
    return kde_vectors, file_indices


def query_match_with_pca_vectors(question, top_n=3):
    """ 使用均值向量快速匹配 `.npy` 文件，并从匹配的 `.npy` 文件中查询最相似的句子 """

    # **1️⃣ 加载预计算的均值向量**
    mean_vectors, file_indices = load_pca_kde_vectors()  # 加载均值向量和对应的文件索引

    # **2️⃣ 计算查询句子的向量**
    query_input = tokenizer([question], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        query_embedding = model(**query_input).last_hidden_state.mean(dim=1).numpy()

    # **3️⃣ 计算查询句子与所有 `.npy` 文件均值向量的相似度**
    file_similarities = cosine_similarity(query_embedding, mean_vectors)[0]

    # **4️⃣ 取前 N 个最相关的 `.npy` 文件**
    top_indices = np.argsort(file_similarities)[::-1][:top_n]
    top_files = file_indices[top_indices]  # 获取最相关的 `.npy` 文件编号
    print(top_files)

    # **5️⃣ 从匹配的 `.npy` 文件加载句子向量**
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
    else:
        print("No matching sentences found.")
        return []


if __name__=="__main__":
    #compute_and_store_pca_kmeans_vectors('../semantic_vector')

    start_time=time.time()
    query_match_with_pca_vectors("some detail about econophysics models equivalent market exchanges redistribution nonequivalent mutual aid exchanges wealth inequality economic flow")
    end_time=time.time()
    print(f"use {end_time-start_time} s")
    #print(extract_keywords("Latino registered voters' views on whether they want Trump to remain a national political figure and potentially run for president in 2024"))
