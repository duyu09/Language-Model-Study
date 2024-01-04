# _*_ coding:utf-8 _*_
"""
@Version  :
             1.0.0
@Time     :
             2024/01/03
@Author   :
             DuYu (@Duyu09, qluduyu09@163.com)
@File     :
             main.py
@Describe :
             自然语言处理 基于”词袋模型“的文本生成练手代码。Practicing code of text generation based on bag-of-word model of NLP.
@Copyright:
             Copyright (c) 2024 DuYu (No.202103180009), Faculty of Computer Science & Technology, Qilu University of Technology (Shandong Academy of Sciences).
@Note     :
             Last Modify: 2024/01/05 备注：数据集是第一列为文本语料的csv文件，若有txt文件需要转换或修改本代码。好的数据集来源参见Readme.md文档。
"""

import csv
import pickle
import random
from collections import Counter, defaultdict

# 定义测试数据集
corpus = ["我喜欢吃苹果", "我喜欢吃香蕉", "我爱中国", "他喜欢吃苹果", "他喜欢中国", "托尼爱吃香蕉", "我爱齐鲁工业大学", "托尼也在齐鲁工业大学", "我在齐鲁工业大学吃香蕉",
          '我喜欢齐鲁工业大学小璇子']


# 定义分词函数，中文语句中，每个汉字就是一个词。

def tokenizer(text):
    # return text.split(" ")  # 英文以空格作为词语间的分隔符
    return [c for c in text]


# 词语分组并计算词频

def count_ngrams(corpus, n):
    arr = []
    result = defaultdict(dict)
    for text in corpus:
        words = tokenizer(text)
        for e in range(len(words) - n + 1):
            arr.append(tuple(words[e:e + n]))
    count_arr = Counter(arr)
    for i in set(arr):
        result[i[0]][i[1]] = count_arr[i]
    return dict(result)


# 计算概率

def count_probability(count_dict):
    probability_dict = dict()
    for i in count_dict:
        item_dict = count_dict[i]
        s = sum(item_dict.values())
        temp_dict = dict()
        for j in item_dict:
            temp_dict[j] = item_dict[j] / s
        probability_dict[i] = temp_dict
    return probability_dict


# 给定一个词，计算下一个词，mode参数可以取'max'或'probability'。max指取最大概率值的词作为下一个词；probability指根据概率分布随机选择词语。

def next_word(probability_dict, word, mode='probability'):
    probability_dict_itemdict = probability_dict.get(word)
    if probability_dict_itemdict is None:
        return None
    if mode == 'probability':
        keys = list(probability_dict_itemdict.keys())
        probabilities = list(probability_dict_itemdict.values())
        return random.choices(population=keys, weights=probabilities)[0]
    else:
        max_value, max_word = 0, None
        for c in probability_dict_itemdict:
            if probability_dict_itemdict[c] > max_value:
                max_value = probability_dict_itemdict[c]
                max_word = c
        return max_word


# 生成连续的文本

# def generate_text(probability_dict, first_word): # 存到字符串变量里输出
#     text = [first_word]
#     n_word = next_word(probability_dict, first_word)
#     while True:
#         if n_word is None:
#             return ''.join(text)
#         text.append(n_word)
#         n_word = next_word(probability_dict, n_word)

def generate_text(probability_dict, first_word):  # 动态输出
    print(first_word, end='')
    n_word = next_word(probability_dict, first_word)
    while True:
        if n_word is None:
            break
        print(n_word, end='')
        n_word = next_word(probability_dict, n_word)


# 读取并处理数据集(数据集格式：文本存储在CSV文件的第一列中)

def load_dataset(filename):
    dataset = []
    with open(filename, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            text = row[0]
            w = ""
            for c in text:
                if not (0x4E00 <= ord(c) <= 0x9FFF):  # 只支持汉字，中文标点以及其他字符被排除。
                    if not w == "":
                        dataset.append(w + "。")
                    w = ""
                else:
                    w = w + c
            if not w == "":
                dataset.append(w + "。")
    return dataset


# 保存模型(字典)

def save_model(filename, probability_dict):
    with open(filename, 'wb') as file:
        pickle.dump(probability_dict, file)


# 加载模型

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


# 训练模型

def train():
    filename = r"C:\Users\Duyu\Desktop\testLM\output.csv"  # csv语料库文件名
    model_filename = r"./output.model"
    dataset = load_dataset(filename)  # 可以先使用测试数据集corpus进行测试。
    rd = count_ngrams(dataset, 2)
    probability_dict = count_probability(rd)
    save_model(model_filename, probability_dict)


# 测试模型

def test(first_word):
    model_filename = r"./output.model"
    probability_dict = load_model(model_filename)
    generate_text(probability_dict, first_word)


if __name__ == '__main__':
    # train()
    for _ in range(10):
        test('我')
        print()
