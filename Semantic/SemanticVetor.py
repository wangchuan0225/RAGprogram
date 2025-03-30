import os
import re
import json
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
from pathlib import Path
import spacy
from utils.base import split_sentences, load_from_json
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from utils.base import extract_number
embeddings_dir="../semantic_vector"
nlp = spacy.load("en_core_web_sm")
model_name = "bert-base-uncased"
model = AutoModel.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()  # 设置为评估模式
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 加载预训练模型
#model = SentenceTransformer('all-MiniLM-L6-v2')  # 或 'all-mpnet-base-v2' (精度更高但速度较慢)
def text2vetor(sentence_path,no):

    sentences=load_from_json(sentence_path)
    # 将句子转换为向量
    #sentence_embeddings = model.encode(sentences)  # sentences 是句子的列表


    # 将句子转换为嵌入向量
    embeddings_list = []

    # 批量大小（根据你的硬件性能调整）
    batch_size = 16
    print("start producing\n")
    # 按批次处理句子，并显示进度条
    for i in tqdm(range(0, len(sentences), batch_size), desc="Processing Sentences"):
        batch_sentences = sentences[i:i + batch_size]

        # Tokenize 每批句子
        inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        # 计算均值嵌入
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings_list.append(embeddings.numpy())

    # 将嵌入向量合并为一个 NumPy 数组
    embeddings_numpy = np.vstack(embeddings_list)

    # 保存到本地文件
    np.save(f"../temp_output/embeddings{no}.npy", embeddings_numpy)
    np.save(f"../semantic_vector/embeddings{no}.npy", embeddings_numpy)
mean_vector_cache = "../semantic_vector/mean_vectors.npy"  # 存储所有 `.npy` 文件的均值向量
file_indices_cache = "../semantic_vector/mean_file_indices.npy"  # 存储所有 `.npy` 文件编号



def compute_and_store_mean_vectors():
    """ 计算所有 `.npy` 文件的均值向量并存储 """
    embeddings_paths = sorted(Path(embeddings_dir).glob('embeddings*.npy'), key=extract_number)
    file_indices = [extract_number(path) for path in embeddings_paths]

    mean_vectors = []

    for emb_path in embeddings_paths:
        embeddings = np.load(str(emb_path), mmap_mode="r")  # 直接从磁盘加载，避免占用内存
        mean_vector = np.mean(embeddings, axis=0)  # 计算均值向量
        mean_vectors.append(mean_vector)

    # 存储为 `.npy` 文件
    np.save(mean_vector_cache, np.vstack(mean_vectors))
    np.save(file_indices_cache, np.array(file_indices))
    print("Mean vectors saved successfully!")

def load_sentences_by_indices(indices):
    """ 根据文件索引加载对应的 JSON 句子文件 """
    sentences = []
    for idx in indices:
        json_path = Path(f"../output/processed_file_{idx}.json")
        if json_path.exists():  # 确保文件存在
            with open(json_path, "r", encoding="utf-8") as f:
                sentences.extend(json.load(f))  # 追加句子
    return sentences
def load_mean_vectors():
    """ 加载 `.npy` 文件的均值向量 """
    mean_vectors = np.load(mean_vector_cache)
    file_indices = np.load(file_indices_cache)
    return mean_vectors, file_indices


def query_match_with_mean_vectors(question, top_n=3):
    """ 使用均值向量快速匹配 `.npy` 文件 """

    # **1️⃣ 加载预计算的均值向量**
    mean_vectors, file_indices = load_mean_vectors()

    # **2️⃣ 计算查询句子的向量**
    query_input = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        query_embedding = model(**query_input).last_hidden_state.mean(dim=1).numpy()

    # **3️⃣ 计算查询句子与所有 `.npy` 文件均值向量的相似度**
    file_similarities = cosine_similarity(query_embedding, mean_vectors)[0]

    # **4️⃣ 取前 N 个最相关的 `.npy` 文件**
    top_indices = np.argsort(file_similarities)[::-1][:top_n]
    top_files = file_indices[top_indices]

    # **5️⃣ 仅加载这些 `.json` 文件中的句子**
    selected_sentences = load_sentences_by_indices(top_files)

    # **6️⃣ 计算查询句子与这些句子的相似度**
    if selected_sentences:
        sentence_inputs = tokenizer(selected_sentences, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            sentence_embeddings = model(**sentence_inputs).last_hidden_state.mean(dim=1).numpy()

        final_similarities = cosine_similarity(query_embedding, sentence_embeddings)
        final_top_indices = np.argsort(final_similarities[0])[::-1][:top_n]
        final_top_sentences = [selected_sentences[i] for i in final_top_indices]

        # **输出结果**
        for i, sentence in enumerate(final_top_sentences):
            print(f"Top {i + 1} sentence: {sentence}\n")

        return final_top_sentences
    else:
        print("No matching sentences found.")
        return []

def faiss_match(question):
    # 加载嵌入
    embeddings = np.load("../output/embeddings.npy").astype("float32")  # FAISS 要求 float32
    with open("../output/output_result.txt", "r") as f:
        text = f.read()
    sentences = split_sentences(text)
    sentences = list(dict.fromkeys(sentences))
    dimension = embeddings.shape[1]

    # 初始化索引
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"已添加 {index.ntotal} 条向量到索引中。")

    # 查询句子转嵌入
    inputs = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"last_hidden_state 的数据类型: {outputs.last_hidden_state.dtype}")

    # 转换为 numpy 后检查数据类型
    query_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    print(f"转换为 numpy 后的数据类型: {query_embedding.dtype}")
    # 检查维度
    if query_embedding.shape[1] != dimension:
        raise ValueError(f"查询向量维度 {query_embedding.shape[1]} 与索引维度 {dimension} 不匹配。")

    # FAISS 搜索
    k = 5
    distances, indices = index.search(query_embedding, k)
    print(indices)
    # 显示结果


if __name__ =='__main__':
    '''
    # 进行语义向量的计算
    all_sentence_path = Path('../output')
    sentences_path = list(all_sentence_path.glob('*.json'))  # 获取所有匹配的 Path 对象
    sorted_sentences = sorted(sentences_path, key=extract_number)

    for i in sorted_sentences:
        no=extract_number(i)
        text2vetor(i,no)
    #question="Latino registered voters' views on whether they want Trump to remain a national political figure and potentially run for president in 2024"
    query_match_with_cosine(question)
    #faiss_match(question)
    '''
    query_match_with_mean_vectors("macbook hardware screen dashboard")







