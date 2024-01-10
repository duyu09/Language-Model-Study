## Language Model Study

**初学NLP: 学习自己搭建简单的语言模型。欢迎大家一起交流，共同进步！** 

- **著作权声明：** Copyright © 2024 DuYu (@Duyu09), Faculty of Computer Science and Technology, Qilu University of Technology (Shandong Academy of Sciences). 

- **参考书籍：** 《GPT图解 大模型是怎样构建的》（黄佳）、《自然语言处理 基于预训练模型的方法》（车万翔 等）、《大规模语言模型》（张奇、桂韬、郑锐、黄萱菁）
  
- **数据集推荐：**

  | 语料库名称 | 网址 |
  | ----- | ----- |
  | 复旦大学NLP团队中文语料库 | http://www.nlpir.org/wordpress/download/tc-corpus-answer.rar |
  | 清华大学NLP实验室新闻文本数据集 | http://thuctc.thunlp.org/#中文文本分类数据集THUCNews |
  | ChineseNlpCorpus汇总的NLP相关数据集 | https://github.com/SophonPlus/ChineseNlpCorpus |
  | 中文语言理解测评基准(CLUE) | https://github.com/CLUEbenchmark/CLUE |
  | 本人整理的针对情感分析的语料库 | https://github.com/duyu09/NLP-DataSet-of-Emotion-Analysing |

**更新日志**

- Update on Jan. 4th, 2024
  
  - 创建代码仓库，开发词袋模型v1.0版本。
    
  - “词袋模型”是NLP中常用的一种文本表示方法。它将文本看作是一个“袋子”，忽略了文本中单词出现的顺序，只关注每个单词在文本中出现的频率或者存在的情况。在这个模型中，一个文本可以被表示为一个由单词构成的集合，每个单词都有一个对应的计数(或者布尔值)，表示它在文本中的出现情况。词袋模型常用于文本分类，情感分析、信息检索等场景。
 
- Update on Jan. 10th, 2024

  - 学习并尝试编写了基于Skip-Gram模型算法的英文词向量训练代码，使用PyTorch构建了具有一个嵌入层和一个线性隐藏层的神经网络，用了交叉熵损失作为损失函数，SGD作为参数优化算法。
  - Skip-Gram是一个用于词嵌入的模型，它是Word2Vec模型的一部分，它的目标是给定一个中心词，预测在该词的上下文窗口内可能出现的其他词。当神经网络的参数稳定下来时，Embedding层的参数就是词嵌入矩阵。
  - 词嵌入向量反映一个词语在高维空间中与其他词语间的关系，通常两个词的词向量越接近，两个词语的语义(或其他某些属性)就越相似。利用词向量可以完成分类、情感分析等非常多的NLP下游任务。
 
  
### 访问次数统计

<div><b>Number of Total Visits (All of Duyu09's GitHub Projects): </b><br><img src="https://profile-counter.glitch.me/duyu09/count.svg" /></div> 

<div><b>Number of Total Visits (Language-Model-Study): </b><br><img src="https://profile-counter.glitch.me/duyu09-Language-Model-Study/count.svg" /></div> 
