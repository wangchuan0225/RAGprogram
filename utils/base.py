import re
import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
nlp = spacy.load("en_core_web_sm")
TOKEN_LIMIT=1000

message = "我会给你一大段话关于所查询的内容以及，所要查询的问题。总结所给的话中的数据信息和文字信息，如果给的信息里面有一些具体的数据或者名词解释，也要讲出来。" \
          f"并结合你自身所知道的信息输出结果，结果中要包括所给的话中的信息，所给的文段中可能有不相关的句子，可以进行排除。所给文段："
def extract_number(filename):
    # 提取文件名中的数字
    match = re.search(r'\d+', filename.stem)  # 提取文件名的主干部分并搜索数字
    return int(match.group()) if match else float('inf')  # 如果没找到数字，返回无穷大（放最后
def load_from_json(file_path):
    with open(file_path, 'r') as file:
        inverted_index = json.load(file)
    return inverted_index
def is_valid_sentence(tokens):
    """
    判断句子是否有效：必须包含主语和谓语，避免孤立单词。
    """
    has_subject = any(token.dep_ in {"nsubj", "nsubjpass"} for token in tokens)
    has_predicate = any(token.dep_ == "ROOT" for token in tokens)
    return has_subject and has_predicate
def split_sentences(text):
    doc = nlp(text)
    sentences = []
    current_sentence=[]

    # 遍历所有 token
    for token in doc:
        current_sentence.append(token)

        # 遇到句号或特殊分隔符时，尝试分句
        if token.text in {".", "!", "?"}:
            if is_valid_sentence(current_sentence):
                sentences.append(" ".join([t.text for t in current_sentence]).strip())
                current_sentence = []

        # 最后一个句子
    if current_sentence:
        if is_valid_sentence(current_sentence):
            sentences.append(" ".join([t.text for t in current_sentence]).strip())
        else:
            # 如果最后一句不完整，合并到上一句
            if sentences:
                sentences[-1] += " " + " ".join([t.text for t in current_sentence]).strip()

    return sentences

def read_pdf_sentence(pdf_path):
    with open(pdf_path,'r') as f:
        text=f.read()
    return split_sentences(text)
def extract_keywords(text, top_n=8):
    words = list(jieba.cut(text))
    tfidf = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    tfidf_matrix = tfidf.fit_transform([words])
    scores = tfidf_matrix.toarray().flatten()
    keywords = [word for word, score in sorted(zip(tfidf.get_feature_names_out(), scores), key=lambda x: -x[1])][:top_n]
    return " ".join(keywords)
