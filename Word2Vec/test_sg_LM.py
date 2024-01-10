# _*_ coding:utf-8 _*_
"""
@Version  : 1.0.0
@Time     : 2024/01/10
@Author   : DuYu (@Duyu09, qluduyu09@163.com)
@File     : test_sg_LM.py
@Describe : 自然语言处理 基于Skip-Gram模型(跳字模型)的词向量测试练手代码，配套skip_gram.py使用。Practicing code of testing word vector based on skip-gram model of NLP.
@Copyright: Copyright © 2024 DuYu (No.202103180009), Faculty of Computer Science & Technology, Qilu University of Technology (Shandong Academy of Sciences).
@Note     : Last Modify: 2024/01/10
"""
import torch
import pickle


# 计算两个向量的余弦相似度
def cosine_similarity(vector1, vector2):
    magnitude1 = torch.sqrt(torch.sum(vector1 ** 2))
    magnitude2 = torch.sqrt(torch.sum(vector2 ** 2))
    dot_product = torch.sum(vector1 * vector2)
    similarity = dot_product / (magnitude1 * magnitude2)
    return float(similarity)


# 给定两个词，计算它们的余弦相似度
def get_word_similarity(word01, word02, vocabulary_dict, word_vectors):
    word_vector01 = word_vectors[vocabulary_dict[word01]]
    word_vector02 = word_vectors[vocabulary_dict[word02]]
    return cosine_similarity(word_vector01, word_vector02)


# 加载模型以及字典
def load_model_and_dictionary(model_path=r'./word_vectors.pt', dictionary_path=r'./vocabulary_dict.pkl'):
    with open(dictionary_path, 'rb') as file:
        vocabulary_dict = pickle.load(file)
    word_vectors = torch.load(model_path)
    return word_vectors, vocabulary_dict


# 打印与词语target_word最相近的前count(=10)个词语。
def print_most_similar_word(target_word, vocabulary_dict, word_vectors, count=10):
    arr = [(word, get_word_similarity(target_word, word, vocabulary_dict, word_vectors)) for word in vocabulary_dict]
    arr.sort(key=lambda x: x[1], reverse=True)
    for w, s in arr[:count]:
        print("{:<15}".format(w), "{:<20}".format(s))


if __name__ == '__main__':
    word_vectors, vocabulary_dict = load_model_and_dictionary(model_path=r'./word_vectors.pt', dictionary_path=r'./vocabulary_dict.pkl')
    print(list(vocabulary_dict.keys()))  # 打印词典
    print_most_similar_word('word', vocabulary_dict, word_vectors)
