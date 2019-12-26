BiLSTM 序列标注
BiLSTM+ CRF  NER 命名实体识别 log

https://github.com/jantic/DeOldify 上色

https://github.com/chineseGLUE/chineseGLUE
https://github.com/huggingface/transformers

BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet, CTRL..


https://github.com/huggingface/pytorch-pretrained-BERT

Huggingface近日发布了一个新版本的NLP预训练变压器模型开源库：PyTorch-Transformers 1.0。该开源库包含了一下体系结构: 谷歌的BERT，OpenAI的GPT和GPT-2，谷歌以及CMU的Transformer-XL和XLNet，以及Facebook的XLM。这些架构共有27个预训练模型权重

bpe bytes-pair-encoding   解决未登陆词，  训练 #UNK  ，测试
zero-shot 是在下游任务 微调时，
zero-shot domain transfer


openai
任何智力任务的能力，并具有“广泛分布
Gym(用于测试和比较通过反复试验来实现目标的强化学习算法的工具包)和Neural MMO(可以极大地削弱其性能的“多主体”虚拟培训基地)像RPG一样的世界中间的特工。最近的其他公共工作包括CoinRun，它测试强化学习代理的适应性。Spinning Up，一个旨在教任何人深度学习的程序;稀疏变压器，可以预测冗长的文本，图像和音频序列中的下一个内容;和MuseNet，可以使用10种不同乐器在各种流派和风格下生成新颖的四分钟歌曲
Gym 和 Unverse


用增强学习实验环境 I (MuJoCo, OpenAI Gym, rllab, DeepMind Lab, TORCS, PySC2)

伯克利、斯坦福、CMU、OpenAI、Deepmind、Google X

'GPT2-Chinese - 中文的GPT2模型训练代码，基于Pytorch-Transformers，可以写诗，写新闻，写小说，或是训练通用语言模型等


“深度模仿文字制造”（deepfakes for text)

虽然GPT-2经常会产生令人信服的文字，往往给人一种很智能的感觉。然而，如果系统运行的时间足够长，那么它的局限性就会变得很明显。例如：在一个故事中始终如一地使用角色的名字和属性，或者在一篇新闻文章中坚持一个主题
尤其是其面临长期一致性的挑战。例如，在故事中始终如一地使用同样的人名或坚持某个角色独有的个性。

连贯的文本
该模型不仅可以根据给定的文本流畅地续写句子，甚至可以形成成篇的文章，写作水平堪比人类，被外界成为新一代“编故事”神器。
除了能用于语言建模，GPT-2在问答、阅读理解、摘要生成、翻译等等任务上也都有非常好的成绩

两个工具 网页正文提取 dragnet
1https://github.com/codelucas/newspaper
https://github.com/stanfordnlp/stanfordnlp


Megatron 80亿参数  英伟达
gpt2 脑洞过于天马行空  openai  15亿参数

而ctrl   16亿参数  基于条件的  conditional  transformer language 
定向化编故事，
有情节，有逻辑，有细节 有故事性
命题作为，指哪打哪，  买家评论，站主评论

一模比一模更优秀  这厢
CTRL在训练过程中，就会学习这些URL的结构和文本之间的关系。在推理过程中，URL可以指定各种功能，包括域，子域，实体，实体关系，乃至日期

ACL、NeurIPS、EMNLP等AI顶会  ICLR

PPLM  a plug and play method for controlled language generation 
使用预训练好的gpt2 模型，不需要 fine-tuning, 就可以实现 可控的文本生成。 还提供了源代码。


狗屁不通的那个  重复性太严重 车轱辘话来回说   叙事性的故事就容易露馅



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




在skip-gram里面，每个词在作为中心词的时候，实际上是 1个学生 VS K个老师，K个老师（周围词）都会对学生（中心词）进行“专业”的训练，这样学生（中心词）的“能力”（向量结果）相对就会扎实（准确）一些，但是这样肯定会使用更长的时间；

cbow是 1个老师 VS K个学生，K个学生（周围词）都会从老师（中心词）那里学习知识，但是老师（中心词）是一视同仁的，教给大家的一样的知识。至于你学到了多少，还要看下一轮（假如还在窗口内），或者以后的某一轮，你还有机会加入老师的课堂当中（再次出现作为周围词），跟着大家一起学习，然后进步一点。因此相对skip-gram，你的业务能力肯定没有人家强，但是对于整个训练营（训练过程）来说，这样肯定效率高，速度更快。




GPT-2 是使用「transformer 解码器模块」构建的，而 BERT 则是通过「transformer 编码器」模块构建的。我们将在下一节中详述二者的区别，但这里需要指出的是，二者一个很关键的不同之处在于：GPT-2 就像传统的语言模型一样，一次只输出一个单词（token
这种模型之所以效果好是因为在每个新单词产生后，该单词就被添加在之前生成的单词序列后面，这个序列会成为模型下一步的新输入。这种机制叫做自回归（auto-regression），同时也是令 RNN 模型效果拔群的重要思想。
GPT-2，以及一些诸如 TransformerXL 和 XLNet 等后续出现的模型，本质上都是自回归模型，而 BERT 则不然。这就是一个权衡的问题了。虽然没有使用自回归机制，但 BERT 获得了结合单词前后的上下文信息的能力，从而取得了更好的效果。XLNet 使用了自回归，并且引入了一种能够同时兼顾前后的上下文信息的方法。

空白单词 <pad>  <eos>

解码器  带掩模的自注意力层

GPT-2 可以处理最长 1024 个单词的序列。每个单词都会和它的前续路径一起「流过」所有的解码器模块。

生成无条件样本
即生成交互式条件样本

请注意，第二个单词的路径是当前唯一活跃的路径了。GPT-2 的每一层都保留了它们对第一个单词的解释，并且将运用这些信息处理第二个单词（具体将在下面一节对自注意力机制的讲解中详述），GPT-2 不会根据第二个单词重新解释第一个单词


这就是自注意力机制所做的工作，它在处理每个单词（将其传入神经网络）之前，融入了模型对于用来解释某个单词的上下文的相关单词的理解。具体做法是，给序列中每一个单词都赋予一个相关度得分，之后对他们的向量表征求和。

混用了「单词」（word）和「词」（token）这两个概念。但事实上，GPT-2 使用字节对编码（Byte Pair Encoding）方式来创建词汇表中的词（token），也就是说词（token）其实通常只是单词的一部分。
举的例子其实是 GPT-2 在「推断/评价」（inference / evaluation）模式下运行的流程，所以一次只处理一个单词。在训练过程中，模型会在更长的文本序列上进行训练，并且一次处理多个词（token）。训练过程的批处理大小（batch size）也更大（512），而评价时的批处理大小只有 1



在本文中“words”和“token”是可以互换使用的。但实际上，GPT2在词汇表中创建token是使用的字节对编码（Byte Pair Encoding）。这意味着token通常是words的一部分。


我们展示的示例在其推理/评估模式下运行GPT2。这就是为什么它一次只处理一个单词。在训练时，模型将针对较长的文本序列进行训练并一次处理多个tokens。此外，在训练时，模型将处理较大批量（512）并评估使用的批量大小。



CTRL: A CONDITIONAL TRANSFORMER LANGUAGE MODEL FOR CONTROLLABLE GENERATION
推荐里的 是 
Follow The Regularized Leader (FTRL) The



#12月计划
换种方式学习，任务导向，先去玩各种任务，再去看代码，再去看各个模型
gpt-2  写诗 写小说，写作文，生成式对话，qa生成，  可以做nmt吗，感觉都是偏生成的任务
后续的条件式的生成

bert qa判断，ner，seq tag，文本分类，文本蕴含，句向量 词向量，nmt  感觉都是偏任务的

chatbot 
nmt
后续的各种模型，对比，细节，
eml0 xlnet gpt
以及传统的
 word2vec 的改进路线  huffman  层softmax 负样本


推荐学习
在线学习 ftrl
learn2rank
强化学习
那几篇经典论文重读，google的
以及传统的
gbdt+lr
rf

刷题




任务型对话   

IJCNLP 2017（我能说血亏吗）。就是大名鼎鼎的TC-Bot，之前总结的任务型对话中的开源系统就有它。本文使用SL（监督学习）来监督每个模型部件的学习，同时RL（强化学习）做end-to-end的训练。

成功率，平均回报，平均收益

评估
slu :slot-match-rate
dpl : task-success-rate
nlg: bleu
response seletion: acc


任务型机器人核心模块主要包括三部分：

自然语言理解模块 —— Language Understanding  
领域识别，用户意图识别以及槽位提取三个子模块
对话管理模块 —— Dialog Management
自然语言生成模块 —— Natural Language Generation
hun
了
l1 和l2 正则 混合 作为正则项  l1 

特征的每一维度  一个学习率


# bert 项项目

是的，cls是一个，sep是大于等于一个。

'[CLS]'必须出现在样本段落的开头，一个段落可以有一句话也可以有多句话，每句话的结尾必须是'[SEP]'。

例如：

['[CLS]', 'this', 'is', 'blue', '[SEP]', 'that', 'is', 'red', '[SEP]']
"<PAD>", "<UNK>", "<S>", "</S>"
文本结束符</s> 用以表示句子末尾

@bert 的输出 是什么 后面怎么加 分类器 
 熟练   还有 gpt-2 的 有两个网站 熟练

bert 分类 
input-id
input_mask
segment_id 指什么

WordPiece tokenization

gleu 基准
 BERT模型和ELMo有大不同，在之前的预训练模型（包括word2vec，ELMo等）都会生成词向量，这种类别的预训练模型属于domain transfer。而近一两年提出的ULMFiT，GPT，BERT等都属于模型迁移
 cased是意味着输入的词会保存其大写（在命名实体识别等项目上需要）


 guid = "train-%d" % (i)
 text_a = tokenization.convert_to_unicode(line[0])
 label = tokenization.convert_to_unicode(line[1])

 bert 也是很大篇幅 的  unicode   gpt  bpe算呢
 停用词（stopword）。称它们为停用词是因为在文本处理过程中如果遇到它们，则立即停止处理，将其扔掉。将这些词扔掉减少了索引量，增加了检索效率，并且通常都会提高检索的效果。停用词主要包括英文字符、数字、数学字符、标点符号及使用频率特高的单汉字等

 bert 字  不是词   wordpieceWordpieceTokenizer

入门基础
 https://www.cnblogs.com/luchenyao/p/10223209.html 



 sentence-level (e.g., SST-2), sentence-pair-level (e.g., MultiNLI), word-level (e.g., NER), and span-level (e.g., SQuAD)

 预训练表示可以是上下文无关的，也可以是上下文相关的，而且，上下文相关的表示可以是单向的或双向的。上下文无关模型例如word2vec或GloVe可以为词表中的每一个词生成一个单独的“词向量”表示，所以“bank”这个词在“bank deposit”（银行）和“river bank”（岸边）的表示是一样的。上下文相关的模型会基于句子中的其他词生成每一个词的表示。

sklearn 的 word2vec bin 怎么加载
BPE，（byte pair encoder）字节对编码，也可以叫做digram coding双字母组合编码，主要目的是为了数据压缩，算法描述为字符串里频率最常见的一对字符被一个没有在这个字符中出现的字符代替的层层迭代过程


vocab.txt  怎么统计


模型git上的 训练时 添加的 随机数

layernorm  batchnorm 

gelu  relu  

labelsmoothing

str split() 默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等

get_variable  
tf.trainable_variables(), tf.all_variables(), tf.global_variables()


    with tf.name_scope('gru'),tf.variable_scope("gru", reuse=tf.AUTO_REUSE):


1*1 卷积核 
包括1）跨通道的特征整合2）特征通道的升维和降维  3）减少卷积核参数（简化模型）

1维卷积 2维卷积


roi pooling 

faster rcnn 复述
首先vgg16，然后两个分支，
一个分支，rpn网络，3*3 256d 256个通道
1*1 w*h*2k  w*h*4k k=9 
相对坐标 回归， 有个初始的anchor 坐标，  有尺度系数，直接投到 特征图上，
然后 首先是分数 初步刷选，，然后nms 得到topN  

损失函数 分类有正样本负样本 回归 正样本，这里类别识别，主要给坐标 提供的， 坐标初训练

然后另一个分支
roi 和特征图 一起送入 roi pooling，  
N个候选框 固定， 每个候选框 14*14 这是关键，插值 拟合   14*14的网格盖上去，依宽平分，依高平分
   再池化 7*7*N 固定向量， 实现不同大小图片 
   坐标 精调 ，类别再识别

整体训练两轮 即可，精度就不在提升了
首先vgg预训练权重，训练rpn   
然后整体 rpn  联调
在训练rpn
再整体



maskrcnn 复述
一般检测， 网络最后 池化 全连接 cls regloss
池化 对于小目标不友好， fpn feature p





死看 
py-faster-rcnn
mask-rcnn

池化 
rcnn ss svm bbox 回归
roi pooling 提出于 fast rcnn 但候选生成还是ss
而    faster rcnn 候选生成 提出rpn网络

roi align  mask rcnn

必备
物体检测 
人检测
换脸，
舞蹈
关键点那个人很多



这三个死看
Bert 官方源代码  特征提取.py 不是很明白
bert as service 
keras4bert 


bert 提取 词向量  句向量
bert ner 
bert 分类，
bert 相似性
fasttext 意图提取 也是分类

序列标注  开始结束  

序列标注貌似这个词是 范围很广的 分词 后处理 

还有个句法分析  词法分析 还没去搜索



transformer 的多头注意力 就像 多种卷积核


bert 更像rcnn  常见的 都能做 分类  分割 识别  监督任务

gpt 更像 gan  主要是 生成的任务   无监督任务

看似
gpt-chinese 要求  json[ 一段落，一段落]  doupo  正则 检索 所有第几章行，一章 一个段落，     
transformer-xl-chinese 也是好多生成 小说 文章 古诗，先从这个看吧


必备
中英翻译，文本分类
生成对话，
写诗 写文章

但是这样看 nlp的模型结构  远没有 cv的复杂 框架  仅仅transformer 




词性标注 ，pos    
词性标注（part-of-speech tagging） 动词 名词，单纯选取最高频词性，

可以想象 猫狗识别，标注一部分数据  然后 也能识别  未见过的图片  有猫的 
标注 也是一样啊 ，标 名词 动词，一部分  然后 去未见过的词  句子 也能正确标识对的啊 

送给bert 若 是word 则 onehot 太多了
所以 一般 都是 character 字，  常见的 字  大约5000 吧，做onehot 就很容易了，token
所以 一般 bert gpt 先分词 在喂入





improt thulac

THULAC（THU Lexical Analyzer for Chinese）由清华大学自然语言处理与社会人文计算实验室研制推出的一套中文词法分析工具包，具有中文分词和词性标注功能


 BertViz 


通过查看该模型在实际应用（如拼写检查、机器翻译）中的表现来评价
 迷惑度/困惑度/混乱度（preplexity）
 迷惑度越小，句子概率越大，语言模型越好。


词素 byte pair encoding
https://github.com/rsennrich/subword-nmt 
 如果需要把bpe使用到神经网络中，很简单 使用subword-nmt apply-bpe 对输入语料进行解码得到decode.txt，然后在程序载入subword-nmt生成的字典voc.txt。然后按照机器翻译正常处理语料的套路来做即可：读入decode.txt的每个单词，查找它在voc.txt的字典中的编号。
————————————————
https://blog.csdn.net/jmh1996/article/details/89286898

中文有bpe吗




用1*1卷积层代替全连接层的好处：
1、不改变图像空间结构
全连接层会破坏图像的空间结构，而1*1卷积层不会破坏图像的空间结构。
2、输入可以是任意尺寸
全连接层的输入尺寸是固定的，因为全连接层的参数个数取决于图像大小。而卷积层的输入尺寸是任意的，因为卷积核的参数个数与图像大小无关。



这一类讨论者主要在研究 GPT-2 的实用性，一些开发者也附上了自己的做的测试模型，感兴趣的读者可以前去体验：

http://textsynth.org/

https://talktotransformer.com/

pip install gpt2

AllenNLP 
spacy 英语文本处理工具库

Enhanced Representation through Knowledge Integration来看补充更多的先验知识供预训练语言模型学习能够使模型泛化能力更高。ERNIE相当于融入了知识图谱，清华的ERNIE在BERT的MLM以及Next Sentence Prediction任务的基础上增加了denoising entity auto-encoder (dEA)任务，这是自然而然应该想到了，MLM相当于在字上的降噪，增加了实体信息，自然应该在实体层次进行降噪。

cvat视频标注

spacy nlp所有的任务都有 
spaCy 带有预先训练的统计模型和单词向量，目前支持 34+语言的标记（暂不支持中文）。它具有世界上速度最快的句法分析器，用于标签的卷积神经网络模型，解析和命名实体识别以及与深度学习整合。
依存分析，[限定词， 形容词修饰, 名词主语， 根节点, 限定词, 形容词修饰, 形容词修饰, 属性, 标点]

名词短语 介词短语


albert
就有对词向量的投射做一个因式分解和对隐层的参数做共享两个方法分别来减少这两个不同模块的参数量。
他们训练了一个 ALBERT-Tiny 模型。我在谷歌的同事将它转成 tensorflow-lite 之后在手机端上做了一些测试。在 4 线程的 CPU 上的延时是 120 毫秒左右

Transformer中共享参数有多种方案，只共享全连接层，只共享attention层，ALBERT结合了上述两种方案，全连接层与attention层都进行参数共享，也就是说共享encoder内的所有参数

roberta
动态mask



pip install pytorch-pretrained-bert
pytorch-pretrained-bert 内 BERT，GPT，Transformer-XL，GPT-2。

from bert_serving.client import BertClient 

Gaussian error linear units GELU

误差函数，也称高斯误差函数(Error Function or Gauss Error Function)

3) 以往的非线性和随机正则化这两部分基本都是互不相关的，因为辅助非线性变换的那些随机正则化器是与输入无关的。

4) GELU将非线性与随机正则化结合，是Adaptive Dropout的修改

sigmoid -》relu -》 GELU 在cv nlp 语音上都不错

在激活函数领域，大家公式的鄙视链应该是：Elus > Relu > Sigmoid ，这些激活函数都有自身的缺陷， sigmoid容易饱和，Elus与Relu缺乏随机因素

在神经网络的建模过程中，模型很重要的性质就是非线性，同时为了模型泛化能力，需要加入随机正则，例如dropout(随机置一些输出为0,其实也是一种变相的随机非线性激活)， 而随机正则与非线性激活是分开的两个事情， 而其实模型的输入是由非线性激活与随机正则两者共同决定的。