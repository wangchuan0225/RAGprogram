import json
from PDFproduce.Lemmatization import load_from_json, sentences_table, sentence_internal_dict
from QwenAPI.api import return_key_word, return_answer

internal_list=load_from_json(internal_dict)
sentences=load_from_json(sentences_table)
message = "我会给你一大段话关于所查询的内容以及，所要查询的问题。总结所给的话中的数据信息和文字信息，" \
          f"并结合你自身所知道的信息输出结果，结果中要包括所给的话中的信息，所给的文段中可能有不相关的句子，可以进行排除。所给文段："
message1=""
if __name__ == '__main__':
    question=input("输入问题：")
    extract_keyword,associate_keyword=return_key_word(question)
    for keyw in extract_keyword:
        if keyw in internal_list[0]:
            for index in internal_list[0][keyw]:
                message1 = message1+sentences[index]
                if len(message)>10000:
                    break
        else:
            print(f"word:{keyw} can't find\n")
    prompt=message+message1+" 问题是："+question

    #return_answer(prompt)



