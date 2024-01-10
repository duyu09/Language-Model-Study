# _*_ coding:utf-8 _*_
"""
@Version  : 1.0.0
@Time     : 2024/01/10
@Author   : DuYu (@Duyu09, qluduyu09@163.com)
@File     : skip_gram.py
@Describe : 自然语言处理 基于Skip-Gram模型(跳字模型)的词向量训练练手代码，配套test_sg_LM.py测试模型。Practicing code of training word vector based on skip-gram model of NLP.
@Copyright: Copyright © 2024 DuYu (No.202103180009), Faculty of Computer Science & Technology, Qilu University of Technology (Shandong Academy of Sciences).
@Note     : Last Modify: 2024/01/10 备注：数据集格式：文本文件，每行一个句子。语料可以是任何以空格为单词间隔的自然语言 (汉语就不符合条件，除非用JIEBA分词)。
"""

import sys
import torch
import random
import pickle
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm


# 分词器，按序返回词语列表
def tokenizer(sentence):
    return sentence.split(' ')  # 英文以空格分词，中文考虑用JIEBA分词库。


# 读入并处理数据集
def load_dataset(dataset_path=r"./dataset.txt", piece=1250):
    with open(dataset_path, 'r') as file:
        lines = file.readlines()
    list_of_sentence = [line.strip() for line in lines]
    # random.shuffle(list_of_sentence)  # 给数据集打乱顺序
    list_of_sentence.sort(key=len, reverse=True)  # 按句子由长到短的顺序排序
    list_of_sentence = list_of_sentence[:int(len(list_of_sentence) / piece)]  # 为了测试，仅取前piece(=1250)分之一的数据进行训练
    return list_of_sentence


# 处理数据函数，返回词汇表列表和分组元组(邻居词, 目标词)列表，取前后各n(=2)个词为邻居词。
def preprocess_data(list_of_sentence, n=2):
    vocabulary_set = set()
    neighbor_tuple_list = []
    for sentence in list_of_sentence:
        word_list = tokenizer(sentence)
        for i in range(len(word_list)):
            neighbor_list = word_list[max(0, i - n):i] + word_list[i + 1:min(len(word_list), i + n + 1)]
            for neighbor_word in neighbor_list:
                neighbor_tuple_list.append((neighbor_word, word_list[i]))
            vocabulary_set.add(word_list[i])
    vocabulary_dict = dict()
    idx = 0
    for word in vocabulary_set:
        vocabulary_dict[word] = idx
        idx = idx + 1
    return vocabulary_dict, neighbor_tuple_list


# 定义Skip-Gram模型结构：一个嵌入层+一个线性层，嵌入层免去了以独热编码输入NN的步骤
class SkipGram(nn.Module):
    def __init__(self, vocabulary_size, embedding_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_to_hidden = nn.Embedding(vocabulary_size, embedding_size)  # 嵌入层的参数就是词向量矩阵
        self.hidden_to_output = nn.Linear(embedding_size, vocabulary_size)

    def forward(self, X):
        hidden_layer = self.input_to_hidden(X)
        output_layer = self.hidden_to_output(hidden_layer)
        return output_layer


# 模型训练
def train(vocabulary_dict, neighbor_tuple_list, embedding_size=45, learning_rate=0.002, epochs=420,
          model_path=r'./word_vectors.pt', dictionary_path=r'./vocabulary_dict.pkl'):
    vocabulary_size = len(vocabulary_dict)
    skip_gram_model = SkipGram(vocabulary_size, embedding_size)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(skip_gram_model.parameters(), lr=learning_rate)  # 随机梯度下降算法
    loss_values = []
    for epoch in range(epochs):  # 迭代训练，整个程序在这里最消耗性能和时间
        loss_sum = 0
        for item in tqdm(neighbor_tuple_list, file=sys.stdout):
            neighbor, target = item[0], item[1]
            X = torch.tensor([vocabulary_dict[target]], dtype=torch.long)
            y_pred = skip_gram_model(X)
            y_true = torch.tensor([vocabulary_dict[neighbor]], dtype=torch.long)
            loss = criterion(y_pred, y_true)
            loss_sum = loss_sum + loss
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播损失值
            optimizer.step()  # 更新NN里的参数
        print("Epoch: ", epoch + 1, "Loss: ", float(loss_sum / len(neighbor_tuple_list)))
        loss_values.append(float(loss_sum / len(neighbor_tuple_list)))

    torch.save(skip_gram_model.input_to_hidden.weight, model_path)  # 保存词向量矩阵
    with open(dictionary_path, 'wb') as dict_file:  # 保存词典
        pickle.dump(vocabulary_dict, dict_file)
    return loss_values


# 绘制损失值曲线
def draw_loss_figure(loss_values):
    plt.plot(range(len(loss_values)), loss_values)
    plt.title('Loss Values')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


if __name__ == '__main__':
    list_of_sentence = load_dataset(dataset_path=r"./dataset.txt", piece=1250)
    vocabulary_dict, neighbor_tuple_list = preprocess_data(list_of_sentence, n=2)
    loss_values = train(vocabulary_dict, neighbor_tuple_list, embedding_size=45, learning_rate=0.002, epochs=450,
                        model_path=r'./word_vectors.pt', dictionary_path=r'./vocabulary_dict.pkl')
    draw_loss_figure(loss_values)
