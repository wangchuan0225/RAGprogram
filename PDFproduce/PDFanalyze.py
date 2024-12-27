import pymupdf
import fitz
from pathlib import Path
import camelot
floder_path = Path("../data")

def extract_text(pdf_path):
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

def extract_text_and_tables(pdf_path, output_dir):
        import os
        os.makedirs(output_dir, exist_ok=True)

        doc = fitz.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            print(f"--- Processing Page {page_num + 1} ---")

            # 1. 提取文字块
            text_blocks = page.get_text("blocks")  # 获取文字块
            text_output = os.path.join(output_dir, f"page_{page_num + 1}_text.txt")
            with open(text_output, "w", encoding="utf-8") as text_file:
                for block in text_blocks:
                    x0, y0, x1, y1, text, *_ = block
                    if text.strip():  # 过滤掉空文本块
                        text_file.write(f"{text.strip()}\n")
            print(f"Text saved to {text_output}")

            # 2. 提取表格
            try:
                # 使用 camelot 提取表格
                tables = camelot.read_pdf(pdf_path, pages=str(page_num + 1))
                table_output_dir = os.path.join(output_dir, f"page_{page_num + 1}_tables")
                os.makedirs(table_output_dir, exist_ok=True)
                print(tables)
                for i, table in enumerate(tables):
                    table_file = os.path.join(table_output_dir, f"table_{i + 1}.csv")
                    table.to_csv(table_file)
                    print(f"Table saved to {table_file}")
            except Exception as e:
                print(f"No tables found on page {page_num + 1}: {e}")

        print("--- Extraction Complete ---")
# 使用示例
# 以页为单位读取pdf中内容并保存到txt中
def produce_pdf():
    pdf_file = list(floder_path.glob('*.pdf'))
    for file in pdf_file:
        print(file)
        #extract_text_tables(file)
        extract_text_and_tables(file,'../output')
        break






