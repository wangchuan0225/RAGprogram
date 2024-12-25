import os
import re
from pathlib import Path
def clean_text(file_path):
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

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
                cleaned_lines.append(line.lower())

    # 将清洗后的内容写回文件或新文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(cleaned_lines)


txt_path=Path('../output')
txt_file=list(txt_path.glob('*.txt'))
for file in txt_file:
    clean_text(file)
