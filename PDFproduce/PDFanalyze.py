import pymupdf
import fitz
from pathlib import Path
floder_path = Path("../data")
'''
for page_num in range(1):
    page = doc[page_num]
    text = page.get_text()  # 提取文本
    print(f"--- Page {page_num + 1} ---")
    print(text)
'''
def extract_text_tables(pdf_path):
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("blocks")  # 提取所有文本块

        print(f"--- Page {page_num + 1} ---")
        for block in blocks:
            # 获取块内容中的文本部分
            text = block[4]  # 文本通常是第5个元素（索引从0开始）
            if text.strip():  # 过滤掉空文本块
                print(text)  # 打印纯文本



    doc.close()

# 使用示例
pdf_file = list(floder_path.glob('*.pdf'))
for file in pdf_file:
    print(file)
    extract_text_tables(file)
    break



