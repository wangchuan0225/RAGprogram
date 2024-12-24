import pymupdf
import fitz

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
            if len(block) == 5:
                x0, y0, x1, y1, text = block
            elif len(block) == 6:
                x0, y0, x1, y1, text, block_type = block
            else:
                print("Unexpected block format:", block)
                continue

            if text.strip():  # 过滤掉空内容块
                if "\n" in text or "\t" in text:  # 简单判断是否为表格
                    print(f"Block at ({x0}, {y0}, {x1}, {y1}):")
                    print(text)

    doc.close()

# 使用示例
pdf_path = "../data/data1.pdf"
extract_text_tables(pdf_path)



