# _*_ coding:utf-8 _*_
"""
@Version  :
             1.0.0
@Time     :
             2024/01/05
@Author   :
             DuYu (@Duyu09, qluduyu09@163.com)
@File     :
             text-classification.py
@Describe :
             自然语言处理 基于“词袋模型”的文本分类练手代码。Practicing code of text classification based on bag-of-word model of NLP.
@Copyright:
             Copyright (c) 2024 DuYu (No.202103180009), Faculty of Computer Science & Technology, Qilu University of Technology (Shandong Academy of Sciences).
@Note     :
             Last Modify: 2024/01/05 备注：CSV数据集格式：第一列为文本数据，第二列为标签。好的数据集来源参见Readme.md文档。
"""

import os
import csv
import pickle
import numpy as np
from collections import Counter
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 定义标点符号表
punctuations = ['！', '，', '。', '？', '；', '：', '、', '（', '）', '【', '】', '‘', '’', '“', '”', '《', '》', '—',
                '…', ',', '.', '!', '?', ';', ':', '(', ')', '[', ']', '{', '}', '<', '>', '`', '~', '@', '#', '$',
                '%', '^', '&', '*', '_', '-', '=', '+', '|', '\\', '/', "'", '"'
                ]

# 定义常用停用词表
chinese_stopwords = [
    '个', '些', '次', '种', '件', '条', '支', '张', '篇', '本', '只', '位', '处', '名', '根', '颗', '座', '床', '间',
    '层', '幅', '和', '及', '了', '或', '等', '乃', '乃至', '与', '就', '以', '并', '而', '然后', '但是', '因为',
    '所以', '如果', '如此', '这样', '怎样', '啊', '呢', '吧', '哦', '嘛', '哈', '嗯', '咦', '唉', '噢', '嗬', '咚',
    '我', '你', '他', '她', '它', '我们', '你们', '他们', '自己', '这', '那', '谁', '哪', '哪儿', '哪里', '凡', '所有',
    '各', '另', '其', '此', '每', '全', '该', '一些', '一切', '有些', '是', '的', '这个', '那个', '哪个', '虽然',
    '但是', '但', '还是', '或者', '或', ' '
]

# 停用词合计
total_stopwords = set(punctuations + chinese_stopwords)


# 定义分词函数，中文语句中，每个汉字就是一个词。
def tokenizer(text):
    # return text.split(" ")  # 英文以空格作为词语间的分隔符
    return [c for c in text]


# 加载CSV数据集，数据集格式：第一列为文本，第二列为标签。
def load_csv_data(filename):
    result = []
    with open(filename, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            result.append((row[0], row[1]))
    return result


# 数据预处理
def preprocessing(data_list):
    result = []
    for item in data_list:
        text = item[0]
        text_new = "".join([c for c in text if ((0x4E00 <= ord(c) <= 0x9FFF) and (c not in total_stopwords))])
        if text_new != "":
            text_new_token = tokenizer(text_new)
            result.append((text_new_token, item[1]))
    return result


# 生成词典(一个包含数据集中所有词(仅出现一次)的列表)
def generate_set_list(pre_data_list):
    result = set()
    for item in pre_data_list:
        word_list = item[0]
        for word in word_list:
            result.add(word)
    result = list(result)
    return result


# 得到特征矩阵(ndarray)及对应的标签(词语)向量
def get_feature_matrix(pre_data_list, set_list, pca_n_components=0.96):
    feature_matrix, label_vector = [], []
    for item in pre_data_list:
        feature_vector = []
        word_list = item[0]
        word_list_len = len(word_list)
        cnt = Counter(word_list)
        for d in set_list:
            feature_vector.append(cnt[d] / word_list_len)
        feature_matrix.append(feature_vector)
        label_vector.append(item[1])
    pca = PCA(n_components=pca_n_components)
    pca_feature_matrix = pca.fit_transform(np.array(feature_matrix))
    return pca_feature_matrix, label_vector


# 训练模型
def train_model(feature_matrix, label_vector, model_save_path="./model.pkl"):
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, label_vector, test_size=0.1)
    svm_model = SVC(kernel='rbf', C=1.25, gamma='auto')  # 这里使用径向基核函数(RBF Kernel)，还可以使用其他核函数。
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确度：{accuracy}")

    path = os.path.abspath(model_save_path)
    with open(path, 'wb') as file:
        pickle.dump(svm_model, file)
    print(f"模型已保存到：{path}")


# 模型推理测试
def test_model(feature_matrix, label_vector, model_save_path="./model.pkl"):
    with open(model_save_path, 'rb') as file:
        model = pickle.load(file)
        label_vector_pred = model.predict(feature_matrix)
        accuracy = accuracy_score(label_vector, label_vector_pred)
        print(f"测试得模型准确度：{accuracy}")


if __name__ == '__main__':
    original_data = load_csv_data(r"D:\Downloads\n02\0.csv")
    pre_data = preprocessing(original_data)
    set_list = generate_set_list(pre_data)
    feature_matrix, label_vector = get_feature_matrix(pre_data, set_list)
    train_model(feature_matrix, label_vector)  # 训练模型
    # test_model(feature_matrix, label_vector)  # 测试模型
