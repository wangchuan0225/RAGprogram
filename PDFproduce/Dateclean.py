import os
import re
from pathlib import Path

output_result='../output/output_result.txt'
def clean_text(file_path):
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        os.remove(file_path)

    # 定义正则表达式模式
    url_pattern = re.compile(r'^www\.[^\s]+')
    single_digit_pattern = re.compile(r'^\d+$')

    # 清洗文本数据
    cleaned_lines = []
    for line in lines:
        # 去除URL
        if not url_pattern.match(line.strip()):
            # 去除单独的数字行
            if not single_digit_pattern.match(line.strip()):
                cleaned_lines.append(line.strip().lower())
        cleaned_lines = [
            line.replace('\n', ' ').replace('\r', ' ')  # 替换换行符和回车符为单个空格
            for line in cleaned_lines
        ]

        # 现在连接所有行，确保行间也只有一个空格
        combined_text = ' '.join(cleaned_lines)

        # 如果需要进一步去除多余的空格（例如多个连续空格），可以使用以下方法：
        combined_text = ' '.join(combined_text.split())
        # 现在连接所有行，确保行间也只有一个空格
    combined_text = ' '.join(cleaned_lines)
    print(combined_text)
    # 将清洗后的内容写回文件或新文件
    with open(output_result, 'a', encoding='utf-8') as file:
        file.write(combined_text)

def data_clean():
    txt_path=Path('../output')
    txt_file=list(txt_path.glob('*.txt'))
    for file in txt_file:
        clean_text(file)
