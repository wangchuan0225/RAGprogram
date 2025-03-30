import os
import re
from pathlib import Path
from PDFproduce.PDFanalyze import extract_text_and_tables, produce_pdf

output_result='../output'
import os
import re


def clean_text(file_path, no, output_result):
    # 定义正则表达式模式
    url_pattern = re.compile(
        r'https?://(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^/\s]*)*'
    )
    single_digit_pattern = re.compile(r'^\d+$')  # 只包含数字的行
    symbol_only_pattern = re.compile(r'^[^\w\s]+$')  # 只包含非字母数字字符（符号）的行

    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 清洗文本数据
    cleaned_lines = []
    for line in lines:
        stripped_line = line.strip()
        # 去除URL、单独的数字行以及单独的符号行
        if (not url_pattern.match(stripped_line) and
                not single_digit_pattern.match(stripped_line) and
                not symbol_only_pattern.match(stripped_line)):
            cleaned_lines.append(stripped_line.lower())

    # 替换换行符和回车符为单个空格，并移除多余的空白
    cleaned_lines = [' '.join(line.split()) for line in cleaned_lines]

    # 现在连接所有行，确保行间也只有一个空格
    combined_text = ' '.join(cleaned_lines)

    # 构建输出文件路径
    content_text = os.path.join(output_result, f"pdf_{no}.txt")

    # 如果目录不存在，则创建它
    os.makedirs(output_result, exist_ok=True)

    # 将清洗后的内容写入新文件
    with open(content_text, 'a', encoding='utf-8') as file:
        file.write(combined_text + '\n')  # 每次写入后添加一个换行符，方便多条记录

    # 如果需要删除原始文件，在这里进行
    os.remove(file_path)

def data_clean(no):
    txt_path=Path('../temp')
    txt_file=list(txt_path.glob('*.txt'))
    for file in txt_file:
        clean_text(file,no,'../temp_output')
'''
if __name__ == "__main__":
    produce_pdf()
    pdf_file = list(floder_path.glob('*.pdf'))
    no=0
    for file in pdf_file:
        print(file)
        extract_text_and_tables(file, '../temp')
        data_clean(no)
        no+=1

'''




