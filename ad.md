dien 用到了 neg-samplting
#feature representation
- user profile :fileds包括 性别，年龄等
- user behavior：用户浏览的物品id，
- ad： ad_id,shop_id 
- context: 时间

对于Amazon数据 吧用户评论简单看做 是 点击，对哪个商品评论 就是对哪个商品点击，   而不是把评论看做正负 来看做点击

阿里的数据
UV：店铺各页面的访问人数，一个IP在24小时内多次访问店铺被只算一次
PV：24小时内店铺内所有页面的浏览总量，可累加。
IPV：指买家找到您店铺的宝贝后，点击进入宝贝详情页的次数，可累加。
IPV_UV是浏览过商品详情的独立访问者，注意：IPV_UV也是不能累加的。


连续型数据 怎么放入网络呢

#

learning to rank
pagerank，From RankNet to LambdaRank to LambdaMART
个性化特征，
排序学习的目的就是通过一些自动化的方法完成模型参数的训练。根据不同类型的训练数据可以将排序学习方法分为以下三类：a）单点标注（point wise）；b）两两标注（pair wise）；c）列表标注（list wise）


linecache里面最常用到的就是getline方法，简单实用可以直接从内容中读到指定的行，日常编程中如果涉及读取大文件，一定要使用首选linecache模块，相比open()那种方法要快N倍，它是你读取文件的效率之源。



检索，展现，点击，收入，CTR3


目标衡量已完成的操作或转化，转化的示例包括购买或提交联系表单。
所需的转化次数
参与度
A / B测试:目标是确定哪些元素导致访问者更高的参与度和行动，然后隔离此变量。

因为把新的广告产品放在已经在进行广告投放的营销者触手可及的地方永远会是最能把盘子做大的方法

Google Analytics 网站分析工具

营销 渠道 流量

帮助广告主在一个场所购买多个出版商的广告位，并使用位置或时间信息定向广告，可自动将广告插入网页，获取报告


第三方监测工具如doubleclick、秒针等都利用了用户跟踪技术，网站 分析工具如GoogleAnalytics、百度统计、CNZZ等 也利用了用户跟踪 技术。

协议，域名，端口有任何一个的不同，就被当作是跨域

重定向 
反向代理


select poll 轮询

QPS query per second

#TODO
blstm
attenlstm  是对广告，是对点击历史


python高级进阶
 scrapy 
 flask

git进阶
分支 
合并
tag

tensorflow进阶
bert
文本分类， word2vec 分层softmax ngram
tfserving

hash idmb5 

search query tokens

