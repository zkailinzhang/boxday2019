(https://github.com/facebookresearch/ReAgent)

Algorithms Supported
Discrete-Action DQN
Parametric-Action DQN
Double DQN, Dueling DQN, Dueling Double DQN
Distributional RL: C51 and QR-DQN
Twin Delayed DDPG (TD3)
Soft Actor-Critic (SAC)



Deep Reinforcement Learning for Recommender Systems
Papers
#DQN:
​ WWW 18 DRN: A Deep Reinforcement Learning Framework for News Recommendation paper ⭐️[Microsoft]

​ KDD 18 Stabilizing Reinforcement Learning in Dynamic Environment with Application to Online Recommendation paper ⭐️[Alibaba]

​ IJCAI 19 Reinforcement Learning for Slate-based Recommender Systems: A Tractable Decomposition and Practical Methodology paper arxiv ⭐️[Google]

​ ICML 19 Off-Policy Deep Reinforcement Learning without Exploration paper

#Policy Gradient:
​ WSDM 19 Top-K Off-Policy Correction for a REINFORCE Recommender System paper ⭐️[Google]

​ NIPS 17 Off-policy evaluation for slate recommendation paper

​ ICML 19 Safe Policy Improvement with Baseline Bootstrapping paper

​ WWW 19 Policy Gradients for Contextual Recommendations paper

​ AAAI 19 Large-scale Interactive Recommendation with Tree-structured Policy Gradient paper

#Actor-Critic:
​ Arxiv 15 Deep Reinforcement Learning in Large Discrete Action Spaces paper code

​ Arxiv 18 Deep Reinforcement Learning based Recommendation with Explicit User-Item Interactions Modeling paper

​ KDD 18 Supervised Reinforcement Learning with Recurrent Neural Network for Dynamic Treatment Recommendation paper

#Post Rank:
​ WWW 19 Value-aware Recommendation based on Reinforcement Profit Maximization paper code Dataset ⭐️[Alibaba]

#Top K:
​ KDD 19 Exact-K Recommendation via Maximal Clique Optimization paper ⭐️[Alibaba]

#Bandit:
​ WWW 10 A Contextual-Bandit Approach to Personalized News Article Recommendation paper

​ KDD 16 Online Context-Aware Recommendation with Time Varying Multi-Armed Bandit paper

​ CIKM 17 Returning is Believing Optimizing Long-term User Engagement in Recommender Systems

​ ICLR 18 Deep Learning with Logged Bandit Feedback paper

​ Recsys 18 Explore, Exploit, and Explain Personalizing Explainable Recommendations with Bandits paper

#Multi-agent:
​ WWW 18 Learning to Collaborate Multi-Scenario Ranking via Multi-Agent Reinforcement Learning paper

⭐️[Alibaba]

#Hierarchical RL
​ AAAI19 Hierarchical Reinforcement Learning for Course Recommendation in MOOCs paper

​ WWW 19 Aggregating E-commerce Search Results from Heterogeneous Sources via Hierarchical Reinforcement Learning paper ⭐️[Alibaba]

#Offline:
​ WSDM 19 Offline Evaluation to Make Decisions About Playlist Recommendation Algorithms paper

​ KDD 19 Off-policy Learning for Multiple Loggers paper

#Explainable:
​ ICDM 18 A Reinforcement Learning Framework for Explainable Recommendation paper

​ SIGIR 19 Reinforcement Knowledge Graph Reasoning for Explainable Recommendation paper

#Search Engine:
​ KDD 18 Reinforcement Learning to Rank in E-Commerce Search Engine Formalization, Analysis, and Application paper ⭐️[Alibaba]

Simulation:
​ ICML 19 Generative Adversarial User Model for Reinforcement Learning Based Recommendation System paper

JD.com:
​ JD Data Science Lab / Dawei Yin / Xiangyu Zhao

KDD 19 Reinforcement Learning to Optimize Long-term User Engagement in Recommender Systems paper ⭐️[JD]

DSFAA 19 Reinforcement Learning to Diversify Top-N Recommendation paper code ⭐️[JD]

​ KDD 18 Recommendations with Negative Feedback via Pairwise Deep Reinforcement Learning paper ⭐️[JD]

​ RecSys 18 Deep Reinforcement Learning for Page-wise Recommendations paper ⭐️[JD]

​ DRL4KDD Deep Reinforcement Learning for List-wise Recommendations paper ⭐️[JD]

​ Sigweb 19 Deep Reinforcement Learning for Search, Recommendation, and Online Advertising: A Survey paper ⭐️[JD]

​ Arxiv 19 Model-Based Reinforcement Learning for Whole-Chain Recommendations paper ⭐️[JD]

​ Arxiv 19 Simulating User Feedback for Reinforcement Learning Based Recommendations paper ⭐️[JD]

​ Arxiv 19 Deep Reinforcement Learning for Online Advertising in Recommender Systems paper




对大多数机器学习初学者来说，较为熟悉的是监督学习（Supervised Learning，SL），但是对强化学习（Reinforcement Learning，RL）比较陌生。2016年初AlphaGo火了以后，作为AlphaGo背后核心技术的Deep Q-Network（DQN）就是一种强化学习算法的一种。

网上关于强化学习的科普文章、介绍资料很多，有些水平差强人意，尤其是一些中文博客简直没法看。下面给出强化学习的一种学习路线图，帮助初学者少走一些弯路。

基础
推荐David Silver关于RL的公开课：
http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html

（David Silver是DeepMind的研究员，也是AlphaGo、DQN背后的大牛之一。）

结合Sutton的经典教材《Reinforcement Learning: An Introduction》
https://webdocs.cs.ualberta.ca/~sutton/book/bookdraft2016sep.pdf
搞定RL的基础不是问题。

进阶
传统RL的主要困难之一在于对复杂的环境进行建模，需要对高维的传感器输入如图像、语音等，抽取特征来表征环境。近年来RL的巨大进展是由于和深度学习（Deep Learning）结合，直接实现了end-to-end的学习和规划。可以看下面几篇paper。

DeepMind用Deep Q-Network来玩Atari系列游戏，达到接近甚至超越人类高手玩家的水平：

Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602 (2013). [pdf])

Human-level control through deep reinforcement learning. Nature 518.7540 (2015): 529-533. [pdf]

大名鼎鼎的AlphaGo，用的是policy gradient算法：
Mastering the game of Go with deep neural networks and tree search. Nature 529.7587 (2016): 484-489. [pdf]

A3C算法，当前的state-of-the-art方法，其中实验结果表明从效果来看A3C > policy gradient > DQN:
Asynchronous methods for deep reinforcement learning." arXiv preprint arXiv:1602.01783 (2016).

前沿
强化学习有很多好的应用，如robotics、route planning等，去看各个顶级会议的paper吧