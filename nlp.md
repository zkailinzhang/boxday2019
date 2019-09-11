BiLSTM 序列标注
BiLSTM+ CRF  NER 命名实体识别 log


阿里的 用到transformer的
ATRank  推荐 用户建模 self-attention  
BST self-attention
DSIN  transformer 


Speech2Vid is applicable to unseen
images and speech.


语言模型 是 生成式的，就会一个字 一个字 输出，  能不能一个词输出 ，一个句子输出


attention  --transformer 
有encoder 和 decoder 
因为有了自注意力， 比如翻译任务中，
encoder 源语句 整体并行 嵌入  输入，多级，循环
然后 decoder中：先是 目标语句self-atte 模块 + encoder最后一轮输出的 作为query 目标作为 key和value   这两个模型连续堆叠  6次，，输出  

x

bert  
仅仅 encoder


语言模型 怎么用bert生成的呢

MASS   bert+LM

gensim.model 封装了  idfmodel  fasttext  word2vec doc2vec
hdpmodel,lsimodel,ldamodel 
word2vec对词向量的训练有两种方式，一种是CBOW模型，即通过上下文来预测中心词；另一种skip-Gram模型，即通过中心词来预测上下文。其中CBOW对小型数据比较适合，而skip-Gram模型在大型的训练语料中表现更好

word2vec  训练方式 cbow skipgram
word2vec 损失函数Hsoftmax
word2vec 损失函数nce
word2vec neg
rnntext  
fasttext  ngram

CBOW和skip-gram模型的核心是训练一个二元分类器，将目标词从k个构造的noise words区分出来
根据分布式假说表示词的方法分为两类：

count-based methods (e.g. Latent Semantic Analysis)
predictive methods (e.g. neural probabilistic language models)


从一个 “noise” 分布 q(y) 中采样出 k 个噪声，即对于一个context，其本来应该是一个真实样本w，现在我们采样k个噪声样本，对于真实样本我们记其标签 d=1,噪声样本d=0, 于是我们有联合分布




# fasttext 
bow词袋表征句子，把所有句子收集起来，构建一个字典，同时赋予 index 值为 出现的个数
然后每个句子， 用value 表示，，
忽略了句子词序，
使用n-gram的词向量使得Fast-text模型可以很好的解决未登录词（OOV——out-of-vocabulary）的问题
而fasttext可以计算未登录词n-gram词（subword）的词向量的平均值，从而得到未登录词的词向量，最终求得和未登录词比较相似的词。

word2vec  
- cbow、skipgram
- 负样本采样
- 层级softmax
- nce-loss
- TF-IDF技术来调整词与的权重，或者训练学习每个词的权重。

无监督句子语义表征方法：

１、一种最经典的方法是在One-hot词语语义表征的基础上使用Bag-of-Words技术。缺点：一是它丢失了词语在句子中的顺序信息；二是它忽略了词语的语义信息，每个词的One-hot表征都是等距离的。

2、类似的还有用word2vec来替换One-hot词向量，使用Bag-of-Words技术，构成句子向量。还可以结合TF-IDF技术来调整词与的权重，或者训练学习每个词的权重。详细见Cedric De Booms的相关论文。

3、基于自编码器，严格说这不是无监督方法，而是一种自监督方法，标签产生自输入数据。输入--》编码--》解码--》输出，输入和输出相同。语义编码C即为句子编码。


N-gram  条件概率  后验概率，


seq2seq 
 test阶段 有用到 beam search 集束搜索 或  greed search   
 维特比算法  利用动态规划方法，找到生成的最大可能性句子

 xlnet  自回归



recurrent NN  就是常见的循环nn
re'cu'r'si've




recursive NN  递归nn  表达  parsing tree
#bert

Tensor2Tensor提供了出色的工具对注意力进行可视化，我结合PyTorch对BERT进行了可视化
BERT采用WordPiece tokenization对原始句子进行解析  
单文本分类任务：对于文本分类任务，BERT模型在文本前插入一个[CLS]符号，并将该符号对应的输出向量作为整篇文本的语义表示，用于文本分类，如下图所示。可以理解为：与文本中已有的其它字/词相比，这个无明显语义信息的符号会更“公平”地融合文本中各个字/词的语义信息。


Transformer-XL由两种技术组成：片段级递归机制(segment-level recurrence mechanism)和相对位置编码方案(relative positional encoding scheme)


RNN, LSTM, GRU and Transformer
Human Inspired Memory Patterns https://medium.com/intuitionmachine/human-inspired-memory-patterns-in-deep-learning-25ab5a254887

RNN:

The Unreasonable Effectiveness of Recurrent Neural Networks

Intro http://eric-yuan.me/rnn1/

RNN with Dinosaurs https://towardsdatascience.com/introduction-to-recurrent-neural-networks-rnn-with-dinosaurs-790e74e3e6f6

Intuitive Guide https://towardsdatascience.com/the-most-intuitive-and-easiest-guide-for-recurrent-neural-network-873c29da73c7

Attention in RNN https://medium.com/datadriveninvestor/attention-in-rnns-321fbcd64f05

Making RNN in Colab https://medium.com/dair-ai/building-rnns-is-fun-with-pytorch-and-google-colab-3903ea9a3a79?_referrer=twitter

LSTM:

Understanding LSTM Networks

Visualizing A Neural Machine Translation Model

Intro http://eric-yuan.me/rnn2-lstm/

Read between the Layers (LSTM network) https://towardsdatascience.com/sentiment-analysis-using-lstm-step-by-step-50d074f09948  –> https://towardsdatascience.com/reading-between-the-layers-lstm-network-7956ad192e58

Amazing power of LSTMs https://medium.com/machinelearningadvantage/the-amazing-power-of-long-short-term-memory-networks-lstms-b6f2c80d50ee

GRU:

Transformer:

What is Transformer https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04

in computational creativity https://medium.com/sfu-big-data/computational-creativity-the-role-of-the-transformer-c3fa20da9c5f

The Illustrated Transformer

The Transformer — Attention is all you need

The Annotated Transformer

Attention is all you need attentional neural network models

Self-Attention For Generative Models

OpenAI GPT-2: Understanding Language Generation through Visualization

WaveNet: A Generative Model for Raw Audio

How Transformers Work

 

Attention:

Attention in PyTorch https://medium.com/intel-student-ambassadors/implementing-attention-models-in-pytorch-f947034b3e66



r = re.compile("[\s+\.\!\/_,$%^*(+\"\']+|[+——！；「」》:：“”·‘’《，。？、~@#￥%……&*（）()]+")
 #以下两行过滤出中文及字符串以外的其他符号
sentence = r.sub('',str(content[i][0]))
seg_list = jieba.cut(sentence)

In [221]: a                                                                                                                                            
Out[221]: ['H', 'e', 'llo']

In [222]: '_'.join(a)                                                                                                                                  
Out[222]: 'H_e_llo'
a[0].join("___")                                                                                                                             
Out[227]: '_H_H_'


aaaa.join("xx")  吧aaa加入到 xx  aaaXaax

ImageNet数据集非常的大，类别也是非常的多，训练出来的东西也更加普世
在很多的反卷积可视化实验当中，都得到了证明，前面层提取出来的特征更加贴近原始图片信息（如纹理，轮廓这些），越往后，越接近特定任务的抽象特征；而网络经过BP，经常是后面的层更加容易被正确地更新。所以正好，预训练把前面不易训练且比较共性地特征给训练出来了，一举两得！




什么叫fine-tuning pre-train， GPT  bert  MAss
什么叫feature-based pre-train   ELMo  


自然语言推理是NLP高级别的任务之一，不过自然语言推理包含的内容比较多，机器阅读，问答系统和对话等本质上都属于自然语言推理。最近在看AllenNLP包的时候，里面有个模块：文本蕴含任务(text entailment)，它的任务形式是：给定一个前提文本（premise），根据这个前提去推断假说文本（hypothesis）与premise的关系，一般分为蕴含关系（entailment）和矛盾关系（contradiction），蕴含关系（entailment）表示从premise中可以推断出hypothesis；矛盾关系（contradiction）即hypothesis与premise矛盾。文本蕴含的结果就是这几个概率值。


(1)词嵌入最常用的模型为word2vec和GloVe，基于分布假设的无监督学习方法(在相同的上下文中单词往往具有相似含义)。

(2)有些方法通过结合语义或句法的知识的有监督学习方法来增强这些无监督方法，纯粹的无监督方法为FastText和EMLo(最先进的上下文词向量)。

(3)在EMLo中，每个单词被赋予一个表示，它是它们所属的整个语料库句子的函数，其来自于一个两层的双向LSTM语言模型的内部状态。




4. 生成式的任务（翻译/摘要）
我觉得沿用seq2seq，类似attention is all you need一样，把两个bert模型，一个当encoder，一个是decoder，也可以做。

文本生成 bert 雷区
不用bert做decoder，只是用来做encoder耶，相当于特征提取


bert不适应 生成类的任务，而 transformerxl  xlnet 可以做 文本生成
bert 11项任务
xlnet 20项任务

综合考虑自编码（autoencoding）模型BERT与自回归（autoregressive）模型transformer-XL的优缺点，作者提出了XLNet，它具有以下优势：

能够学习双向的文本信息，得到最大化期望似然（对所有随机排列，解决mask导致的问题）





改进算法1：基于最大熵的CTC正则
改进算法2：基于等间距的CTC变形



Generally, 
NLTK is used primarily for general NLP tasks (tokenization, POS tagging, parsing, etc.)
Sklearn is used primarily for machine learning (classification, clustering, etc.)
Gensim is used primarily for topic modeling and document similarity.
Having said that, NLTK provides a nice wrapper for Sklearn's classifiers - 
nltk.classify package
Combining Scikit-Learn and NTLK
Python NLP - NLTK and scikit-learn

And, to confuse you further, there also exist TextBlob: Simplified Text Processing

and spaCy.io | Build Tomorrow's Language Technologies - 
aiming to give industry-ready NLP modules instead of NLTK,
including a single quick algorithm for each of tokenization, POS tagging and parsing and word vectors for similarity calculation.

I suggest that you mix and match, according to your needs.


noun phrase extraction 名字提取

# Load English tokenizer, tagger, parser, NER and word vectors
nlp = spacy.load("en_core_web_sm")

Part-of-speech tagging   词性标注

spaCy的语言模型
Gensim - 矢量化文本和转换以及n-gram 
POS标记及其应用程序
NER标记及其应用程序
依赖性解析
顶级模型
高级主题建模
聚类和分类文本
相似性查询和摘要
Word2Vec，Doc2Vec和Gensim 
深度学习文本
Keras和spaCy用于深度学习
情感分析和ChatBots 



word2vec ->  sense2vec   


如何用 gensim 建立语言模型；
如何把词嵌入预训练模型读入；
如何根据语义，查找某单词近似词汇列表；
如何利用语义计算，进行查询；
如何用字符串替换与结巴分词对中文文本做预处理；
如何用 tsne 将高维词向量压缩到低维；
如何可视化压缩到低维的词汇集合；