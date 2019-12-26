Context   是用户api
session    是点击序列


组合学习  

Xdeepfm  所有特征  嵌入空间一样    CIN 输入 

word2vec 未知词


天数 从log  怎么取  前天的 昨天的 大前天的  上一周的  mysql  


第一任务   
indices[107,1] = 364117 is not in [0, 100000)
Section       
chapter     index error
Reflect  index error   

三个基本上 有序，，

分别提前建立 三个json 文件，，不连续的  ，   映射成连续的int  然后嵌入
嵌入空间  10000
Reflect  嵌入空间15wan

近似连续的   但整体 是分段连续的，有缺失值
chapter  54377-59515     86791-92538    324451- 605659  不怎么连续了

Sec   54378-59513     86792-92537    324453 - 605659   不怎么连续了

54000-60000    0  6000    
86000 94000     6000   14000
320000  610000    14000  -  304000


100 0000  128    
40 0000  64 

同时把 128改为  64  避免内存oom

[abs(i) if i < AD_BOUND else i - 90000 for i in x]

# 422144 54377
# 957
# 605659 54378
# 3982 
Chapeter  embed  2000
Section   embed  5000


# 605651 54377
# 11886
Relect    embed  15000     


从候选集 base_chap  base_sec  的数据多，
从history作业集 的数据少 

chapter_id section_id  唯一

base_chap  ,multihot  


第二任务

7天到14的历史作业   缺失的呢    9.1  11111  9.2 没有 填充 11111  9.3 没有 11111   
For 循环呢
还是 都

两个方法，
1。在feature 取历史天数的作业时，就判断，若为0，则，往前7天 看有值吗，有值给填充上，
2. 在取历史天数时，没有 就填充0，，在 handle哪里  prepare_data 那里  倒排 填充，，，可以复习，可以继续学习   可以填充后面的，可以填冲前面的


发现隔天 有都相同的，，这个月复习上个月的，

14天 
cha sec
cha sec
cha sec 


‘12,23’ ’343’   
’34’     ’45,7’
’34,56’   ’45’

注意 multi 不会相差太大
‘12,13’ ’343’   
’34’     ’45,47’
’34,36’   ’45’

第一种 Multihot sum  

’35’  ‘343’
第二种 Multihot mean  更有实际意义

第三种 
1： 12  343      13  343      嵌入做 mean
12 13 嵌入后 mean，，再和343 拼接

注意结果  
先14天，在13天   。。。昨天
        history_all_embedded.append(tf.concat(
        [tf.reduce_mean(getattr(self, key_c),axis=-2),
        tf.reduce_mean(getattr(self, key_s),axis=-2 )],axis=-1))


01001 
00001.  
00011
10010 ok 
01110
0000  这是什么情况  


Chapter id  section id  做repeat 
那reflect  是不是也要做 repeat, 不用吧，就今天的映射，但好多0    reflect 到底是个啥，为啥 today homeid 都有值，reflect为啥还有好多为零？？？


样本\one  two  three 
[][][][]
[][][][]
[][][][]

首先 列，补齐，等长，不够补 那个元素的最后一位  所有样本的 第一列，即第一天
然后，每行，对所有天数，若 某天为零，则补前面一天的 
注意 
[[2],[2,2],[0,0],[0,0,0]]   不等长





第三个任务  
Top5

候选作业集，让然是 offline 表，  依据 chapter id 索取  前500，，三个特征，

从csv读数据， groupby，  每一组遍历，    item[0],,item[1]

然后拼接 

但是 剩余的 数据ph呢，
填充零？？

测试时 ，模型结构喂入的dict 和之前一样啊，
若 

喂入train 两层字典   外层 list  字典


第四个任务
三个跑起来  
设置  
迭代次数 
保存次数
打印次数

第四个 
gbdtlr 跑起来


两个style  ok 其他 o   
连续值得默认值  
离散值得默认值



第五个  
style   3*4  填充值    


候选集有问题，
chapter_id 有问题，是从csv   train_base文件夹，基础章节 基础sction
那训练集也有问题啊 
训练集 没有用 csv 中的 chapter_id 

候选集 有用到，用 csv 中的 chapter_id ，去一个表 检索 候选集

修改
在生成csv那里 handle_train.py  添加 代码 替换chapter_id 

#遗留问题：
-1   替换时 section_id  好多-1 ，，
-1 时，condicate len 为0 continue

{'today_chapters_ph': '86915,86889,86922,86901,86904,86909', 'today_sections_ph': '505110,505111,505112,505113,505114,505115'}]
单独一天的   应该没有次序吧

base_chapter    sec_chapter 
55372,92463    55373,55374,55375,55376,55377,55378,92464,92465
相差比较大，，不是异常值吧，第一章  最后一章


训练

但是准确度 仍然很低啊
2019-10-01 1 1.3062211 0.578125
2019-10-01 2 7.625468 0.171875
2019-10-01 3 6.33211 0.3125
2019-10-01 4 6.619931 0.28125
2019-10-01 5 6.906337 0.25
2019-10-01 6 5.032886 0.2421875


2019-10-01 1 5.252773 0.4296875
2019-10-01 2 7.6273146 0.171875
2019-10-01 3 6.33211 0.3125
2019-10-01 4 6.619932 0.28125
2019-10-01 5 6.317716 0.25
2019-10-01 6 1.5682448 0.21875
INFO:tensorflow:No assets to save.
INFO:tensorflow:No assets to write.
INFO:tensorflow:SavedModel written to: /Users/zhangkailin/zklcode/turing_new/save_sum_3/serving/9/saved_model.pb
2019-10-01 7 0.32165152 0.6875
2019-10-01 8 0.29414144 0.7910448

这精度提上了 

轮数



问题：
1. 预测时  有的天数 没有   len(condicate) == 0:
2. reflect 为啥也嵌入
3. 还有好多序列特征没利用
4. 有个bug   history_chapter  为空，，明明 娶不到值  给领了啊    9.1 还是9.2  47 
5. 训练准确度 0.75 ？？    0.4  0.2  
6. 调参
7. 原始的 gbdt+lr  特征 不一致，，，  在现在这个框架上 重写，，
8. 候选集  有的全证  有的全负
9. 好多特征 挖掘，问科菲
10.  好多Todo  
11. 有的base_chapter 多个值，都比较接近，有的 相差大  


 
model_sum_4.   每一天 的 cha sec 分别 mean  concat        n*14*256  handle.sum.4
                           (Cha1 sec1)  (cha2 sec2 ) ….
                            Sum  后  n*256 
model_sum_1    每一天 的 cha sec 分别 sum  concat       n*14*256  handle.sum.3


model_sum_2    14天 每天 的  cha sum   concat    然后 sec sum concat 
 handle.sum.2      Cha1 cha2 cha3..   sec1 sec2 sec3…
                           不sum 直接喂入   
    

model_sum_22    14天 每天 的  cha mean   concat    然后 sec mean concat 
 handle.sum.22     Cha1 cha2 cha3..   sec1 sec2 sec3…
                           不sum 直接喂入 
 

Ok 有正确结果
model_sum_23    14天 每天 的  cha mean   concat    然后 sec mean concat 
 handle.sum.23     Cha1 cha2 cha3..   sec1 sec2 sec3…
                         分别  mean   mean
                          最终 inp     再分别乘以 today的 cha  sec
                          参考之前的 midas
                           另外  reflect 也是 mean，today也是mean
  

model_sum_24    14天 每天 的  cha sum   concat    然后 sec mean concat 
 handle.sum.24     Cha1 cha2 cha3..   sec1 sec2 sec3…
                         分别 sum sum
                          最终 inp     再分别乘以 today的 cha  sec
                          参考之前的 midas
                           另外  reflect 也是 sum，today也是sum 


0001 两个方向都看    11111
1002 1112   中间有空 拿之前的补，
1000   11111

上面两个

0001  前面没值的 不补了 0001
1000   11111
model_sum_25  sum

model_sum_26  mean

这四个做 base_line




文件锁
多个文件 两个log  两个txt  一个csv  结果  也是csv

先 跑一个，把两个txt  一个csv  都先生成了，
其他，看文件已有 直接读





Handle.bgru.3    model_bgru_1
每一天 的 cha sec 分别 mean  concat
(Cha1 sec1)  (cha2 sec2 )

Handle.gru.3    model_gru_1
Handle.din.3    model_din_1

加dien

双向
Handle.gru.4    model_gru_4
每一天 的 cha  mean   14天  从第14天开始  rnn1  
每一天 的 sec  mean   14天  从第14天开始  rnn2 
然后 concat 
注意；rnn输入 是 mean，但输出 多对多，sum


Handle.bgru.4    model_bgru_4
同上
Handle.din.4    model_din_4

加dien


双向
Handle.gru.5    model_bgru_5
每一天 的 cha  sum  14天  从第14天开始  rnn1  
每一天 的 sec  sum   14天  从第14天开始  rnn2 
然后 concat 
Handle.din.5    model_din_5











Handle.gru.5    model_gru_5     不科学  不可行
每一天 的 cha  rnn    14天  从第14天开始  rnn1     
每一天 的 sec  rnn   14天  从第14天开始  rnn2 
然后 concat 





未做
Handle.gru.6    model_gru_6     不补等长也可以吧 ，给个最大长度即可
14天 所有cha 补等长，14*10    喂rnn 
14天 所有sec 补等长，14*10    喂rnn 


dien


Handle.gru.6

每一天的cha  rnn  多对多，mean     
每一天的sec  rnn  多对多，mean 









Todo  
其他序列特征
学情
deepfm 


问题
UnicodeDecodeError: 'ascii' codec can't decode byte 0xe7 in position 419: ordinal not in range(128)
1. import sys 
2. sys.getdefaultencoding() 
sys.setdefaultencoding('utf-8′)
解决
with open('test.py', encoding = 'UTF-8') as f:



问题
Supervisor FATAL     Exited too quickly (process log may have details)
 BACKOFF   Exited too quickly (process log may have details)
 BACKOFF   Exited too quickly (process log may have details)
FATAL     Exited too quickly (process log may have details)
看log  都是代码问题

1 2 3 4 5
-1 -1 -1 -1 -1-1
-1 -1 -1 -1 -1-1
-1 -1 -1 -1 -1-1
……

Aa是正样本啊，应该【0，1】 
为啥【1，0】也有相同的啊

********************
[array([1.e+00, 1.e-08], dtype=float32), {'today_sections_ph': '86793', 'today_chapters_ph': '86791'}]
{'base_chapters': '86791', 'base_sections': '86793'}
********************

********************
[array([0.43250018, 0.56749976], dtype=float32), {'today_sections_ph': '56147', 'today_chapters_ph': '56143'}]
{'base_chapters': '56143', 'base_sections': '56147'}
********************

********************
[array([0.91495115, 0.08504883], dtype=float32), {'today_sections_ph': '86895', 'today_chapters_ph': '86889'}]
{'base_chapters': '86889', 'base_sections': '86895'}
********************

********************
[array([1.0000000e+00, 1.0017072e-08], dtype=float32), {'today_sections_ph': '54382', 'today_chapters_ph': '54380'}]
{'base_chapters': '54380', 'base_sections': '54382'}
********************

********************
[array([1.e+00, 1.e-08], dtype=float32), {'today_sections_ph': '55386,55387,55388', 'today_chapters_ph': '55385'}]
{'base_chapters': '55385', 'base_sections': '55386,55387,55388'}
********************


********************
[array([1.e+00, 1.e-08], dtype=float32), {'today_sections_ph': '54457,54458,54459,505298', 'today_chapters_ph': '54456'}]
{'base_chapters': '54456', 'base_sections': '54457,54458,54459,505298'}
********************



最终各个算法 对比 csv

横坐标  每天  ，第一张图，每个算法的 top1   在每一天的变化

每一天 ，各个算法 测试 的样本一样多吗  是的

还有个隐形的事实   模型每天都在训练，后期的模型  应该比前期的模型好

机器学习 

数据预处理，怎么处理啊 ，提前知道total_num ？ 
连续值 
类别值
序列

进程队列    作用，用哪里

Gbdt  数据1  增量训练   一下子全部喂入吗，还是分批次   多线程用在什么地方

Onehot  或者 变换矩阵， 有嵌入吗
Lr    还是数据1 还是数据2   增量训练    还可以 keras实现 


训练
测试

训练一天 测试一天  深度学习那边  是 训练一天 测试一天  要不要改成 训练20天的测试一天的，，感觉这个效果好啊

训练20的  测试一天   20天的数据 一下子读出来

先把都择出来  先跑起来   再设置14天，，
再建个项目   应该不是一个体系


#提前生成的   需要 重新生成吗   chapter_id 坏点了

reflect/ encode_v2.pkl    
Table/class.csv

Config.ini  
Generate  还需要 model501 的  可以生成吗

多个文件夹的loggging日志



基准 
gbdt+lr     top1 3 5 10 15   最高的  33 56  68  80

38  67   77  85 



每天的候选样本数 一样的   


Result_sum_23.cp   数据待确定

十分钟一天，，两个月 的数据   10个小时



主讲老师，辅导老师，

核心 金牌主讲老师，权重，权重加哪里，像正负样本那样，参考之前貌似有这样的写法，
现在嵌入，  
直接二分类，做连续值，不嵌入

历史作业 二次嵌入

几个特征 交互 下 

作业推荐  知网  google


先把能跑的跑上，


多看数据，top1 top3 top5 的 从结果发现规律，发现问题，发现改进方向



Midas 的 把数据保存下来，csv更新，，全kill



填充 填三天  填五天，填七天  
0001   0111  而不是 1111
1000  1110   而不是 1111

这个时候就出现 0索引， 在embedding_lookup 
这个0 是从哪里蹦出来的，是原始数据就是0，还是  那天没布置作业，给0，
原始数据没0，

Vocab_size  从1 开始， +1 
other呢，

占位符，  嵌入矩阵，  查找表， 嵌入向量 调整，为0 的为00000，sum rnn gru
而在输入端，原始数据过来，批次同一个占位符的所有数据，保证等长，
然后，二次插值，一个样本内 所有天数的数据，有空的，填充，
然后 sess.run  feed_dict  去lookup，为0的话，不学习，嵌入向量为0000



是否核心， 权重，不是可以学到吗
0嵌入   不学习   从1开始   嵌入矩阵

短补长，[1,2]   [1,2,2]
长补短，[1,2,3]   [1,2]   修改下[1,2,3]  [2,3]
但是有的跨度很大啊，

取最后一个 ，
1223333
1232323 



slot  巣  


先把dien 上 ，mean   双向

gbdt+lr  上 ，   


第二次不插值的，原始喂入，看下效果，掩码  保证原始分布，
但是mask，  是尾部为零，这个这里，mask，中间丢失，中间夹渣着0  mask
啥也不做，让网络去学吧
sum
gru
din
dien


再尝试个
历史作业 按天 走，  不按天走呢，直接拉过来
100200300003      换成1233         淘宝的历史浏览记录也没有按天，广告也没有按天啊
sum 
gru
din
dien


base_mean  填充1天，填充3天  填充5五
所有都上，
dien 1 3 5 
din 1 3 5
gru 1 3 5




先用base_mean  跑 到最新一天，然后在跑其他 




Din mean 双向  继续训练，从10.26 到 
Gru mean 双向 继续训练，从10.26 到最新一天，


强相关  特征点积  

强化学习 ，DQN

考虑 index 0 不学习  cha sec reflect

Handle.sum.30   考虑 index 0 不学习   s gram 1  只补1个
Handle.sum.31   考虑 index 0 不学习   slot 3  补三个
Handle.sum.32   考虑 index 0 不学习   s 5  补五个
Handle.sum.33   考虑 index 0 不学习   不补 原始喂入      
Handle.sum.34   考虑 index 0 不学习   全补    保存按天保存  可以和之前的23 比较 ，判断 0学习否
 Gru din dien  同上 

Handle.sum.40    考虑 index 0 不学习   核心老师权重 1.2  保存按天保存  repeat那 history逆序   
Handle.sum.41    考虑 index 0 学习   核心老师权重   保存按天保存  history逆序
Gru din dien  同上   din40 cp moxing lele
注意 Din 若补5个最好 则跑 补5个的
Sum 补1    全补   和补1  补5
Gru 补3    不补  补1 
bgru 补1    全补
Din 补5    不补
dien 补1     补1

然后综合 模型对比 看看


是不是在来一组 没有核心老师权重，的     看 是否加 核心老四权重 
Handle.sum.42  考虑0不学习，不加老师权重，保存按天保存  repeat那 history逆序 

40和42  对比试验看 老师权重
40和41  看0学习
34 和 4 23  看 0学习否

要不 handle.sum.30 31 32 33 重跑一边   repeat那 history逆序   再判断 补几个    
feature.py 代码 修改

注意 din 的 两个mask  ，model 要改  dien 也有吧   

Handle-sum.43  保存起来
Sum 去看实际 14天 有多少0



Json 三个 从 2019-01-01  跑 
prepare_data函数 和乐乐 对接
核心老师 权重，
离线测试 
1. 简单的restore
2. tf.saved_model.loader.load

tf.tile(


Handle.sum.40   核心老师   损失函数权重，

教龄

既然dien 没有din好  还是重心放在模型结构，其他特征上


Gru32 

代码 restore 有问题，
每天迭代几次 不固定，有的一次
改成一天保存一个

现在训练到10.26   全部跑到最新一天

学习率 

Top1 0.5   +top3 0.3  top5  0.2  top10  0.1 

考虑 mask0 的是不是要跟不 mask0 的 对比下


实际历史天数的作业id ，有多少个连续零，统计一下，5天不到

重要特征，，分支，单独一个分支    太多的话 淹没
交叉熵 权重，

全部重新跑 从9.1 到

Redis 

Gru模型  上线测试

重整midas
dsin_4.19 dsin_4.20  dsin_4.11 dsin_4.14 

06-11-2019 16:15:49 urllib3.connectionpool:DEBUG:https://pypi.python.org:443 "GET /pypi/deepctr/json HTTP/1.1" 301 122
06-11-2019 16:15:49 urllib3.connectionpool:DEBUG:Starting new HTTPS connection (1): pypi.org:443
06-11-2019 16:15:51 urllib3.connectionpool:DEBUG:https://pypi.org:443 "GET /pypi/deepctr/json HTTP/1.1" 200 8162

dsin_4.3 run ok


检查 三个json文件没问题， npy的问题，网络 然后npy一直增加，超过了restore的 model_version

dsin_4.19  9.29开始跑，10.27 开始bug， 生成 7.26到8.9 的

dsin_4.11   9.25开始跑，10.27 开始bug，，生成7.26到8.12的

dsin_4.20   10.2开始跑，10.27 开始bug，，生成7.26到8.8的
Traceback (most recent call last):
  File "/data/zhangkl/Midas/Midas_Engine/update_model_4.20.dsin.py", line 1224, in <module>
    break_dic = np.load(FLAGS.train_cnt_file,allow_pickle=True).item()
  File "/usr/local/lib/python3.5/dist-packages/numpy/lib/npyio.py", line 457, in load
    "Failed to interpret file %s as a pickle" % repr(file))
OSError: Failed to interpret file '/data/zhangkl/Midas/Midas_Engine/train_cnt_dsin_4.20.npy' as a pickle
the JSON object must be str, not 'bytes'

Ipython 读npy没问题的，，吧已有模型删除，npy cp  重零跑

dsin_4.14   9.24开始跑，10.27开始bug，，生成7.26到8.26的


dsin_4.3  9.25开始跑，10.27开始bug，， 生成7.26到8.8的
                10.30又开始跑，但数据开始有 0.5  


34  40 重新跑， 
先跑34 sum的  然后在跑其他，生成csv  注意 测试部分屏蔽

 

0学习和0不学习的 对比
34 和 其他是4  sum是22吧   比较  都是 mean  都是全补，一个考虑0 一个不考虑0


发现两个问题，
一个 补1 补3 补5的， 而din  dien 的mask  仍然是 np.ones()  确定下
一个 14天的  是否按  第14天开始排，，逆序  repeat   确认下

注意  model那边  gru din  是 逆序了 [::-1]

还有个  
feature study()   chapter_id  写错了，


三个json  更新了后，max_num  要更新的
Chapter nums  900 个 给2000      now  2315个 给5000  
Section nums 4000个 给5000   now  9585个  给12000


确认 train_base 下的csv是更新过的吗，
是的，不然 不出结果呢

{
  "K" : "NAVHT4PXWT8WQBL5",
  "P" : "Mac 10.13",
  "DI" : "ODY3M2U0MDZkOWUwNDBi"
}


10.27的log 为0   可以把28的cp，过去，但得到的 train_base  csv 还是不一样，因为从两张表取10.27的 应该也没有数据
最好，10.27的不参与训练，测试无所谓
26训练 27测试 
27训练 continue
28训练  29           27  28  数据无效 。。那拿历史14数据 没有问题吧

26训练 27测试 没有 28测试，28训练 29测试


20分钟一天，9.2 11.6  30 31 6  66*20  1320  60   22     18 个小时  20个小时

Redis   
chapter   key:value  1: map(“id”:1,)   分组，一组1千
Section  
Reflect  


=refine2!$A$1465:$A$1530
=refine2!$J$1539:$J$1604


Dien  30 -34  ok

Sum 40-42 ok

Bgru 40-42 ok 
Gru 40-42 ok 
Din 40-42 ok



Dien40-42 run

考虑0不学习，保存按天保存  repeat那 history逆序   


三个json更新
ones mask  len   
study  




Model  都乘以其他 run  
跨到最后  run    
权重0.8 run
机器学习 重要性内部实现原理
Sum43  看历史天的 run   9.1  9.5  9.10   9.20  9.30   10.1   10.10  10.20  10.30   11.1 

Model_sum 48  classid teacherid #     还有个feature style

model_sum.45 跨连  200 80 20 2      80 core  concat   concat(80 128 )      
model_sum.46   128 -> 20      concat(80 20 )    
model_sum.47  嵌入之后 都乘以其他    原始输入，在添加 core*fea1 core*f2    最后concat 

其他 45 46  47  


45 
46 
42  
45 46 对比 都没老师权重，o学习  和30-34中某个比



线上和线下 有几个补一样
时间，day
study vector  float int
history_mask  [] [] [] 

跟谁做对比试验


Bn   一个特征不加bn呢   一个批次的所有这个特征 
Log 正样本  负样本从表里拿  调试看下    click且 cnt为1  为正样本，点击 cnt为0 为负    仍然是log为主，中间通过一个表，取教师 class cha 
log (423,10)有tag   表 （1282，5 有cnt）merge    cnt 有1 有0     若view 或者cnt 0 则 为负样本


重点学习 style？？


='11.12online'!$A$4:$A$67

='11.12online'!$J$1417:$J$1480


='11.12online'!$J$1488:$J$1551

='11.12online'!$J$1558:$J$1621


='11.12online'!$A$4:$A$67

='11.12online'!$L$1488:$L$1551


后续style 重点学  today的 style 

sum 


表格上 的 top10  补全，  总结，ppt 阶段性实验结果 放到ppt 记录，  结果要详细
服务器上的  代码 全改了， 
int 改成  float
 self.study_vector_ph = tf.placeholder(tf.float32, [None, 20], name="study_vector_ph")


表格上突变的 看下  是不是报错了  看了  没错

特征重要性  

真实数据   14天历史天数  好好统计，表

实验跑到今天

git 上传

注释掉 teacher——id  再跑个  out of rank

feature.py style  
    his = day + datetime.timedelta(days=-30)   改成14


文本分类，ner   gpt2.0
Bert再看  论文代码

把bgru 训练到今天     
补1  补3  
study float 
style 14 


Nlp 下游任务 各个 举例
Gpt2.0   
seq2seq+atte  
生成作文，生成段落，
要一个作文，一直生成句子，怎么不跑偏

占位符 start end  界定符  结束符  
停用符，的  了  吗  

syntactic ambiguity
句法歧义是歧义的一种  

BiLSTM + ELMo + Attn

自动纠错


作业推荐部署线上 turing_
Redis 的代码 维护
监控lishuang那个log文件夹，
若已存在 9.1  8.30 ，新增 9.2 ，就训练9.2 保存模型，不测试，保存serve模型，上传模型，
部署的那个写个程序，，超过5个模型了，删除之前的，  

两个监控程序，
一个是 


Bert 做词向量  文本分类  问答

多来几个为什么

#在作业推荐里
为什么din dien不好  没有 bgru好 ，是打开方式不对吗
为什么 bgru 补零 5 个 比 3 1  不补，全补  好      补1个 因为有大量的 超过1个零，的 所以补1 个 于事无补，，而全补，又感觉不符合事实，

不是不做数据csv，而是从数据中发现问题，

头脑风暴   思想发散 



[array([1.e-08, 1.e+00], dtype=float32), {'base_chapters': '87097', 'base_sections': '87098'}]
{'base_chapters': '87097', 'base_sections': '87098'}
********************
********************
[array([1.016279e-08, 1.000000e+00], dtype=float32), {'base_chapters': '86794', 'base_sections': '424714'}]
{'base_chapters': '86794', 'base_sections': '86795'}
********************
********************
[array([1.0000001e-08, 1.0000000e+00], dtype=float32), {'base_chapters': '86791', 'base_sections': '86793'}]
{'base_chapters': '86791', 'base_sections': '86792'}
********************
********************
[array([1.0014076e-08, 1.0000000e+00], dtype=float32), {'base_chapters': '86889', 'base_sections': '505110'}]
{'base_chapters': '86889', 'base_sections': '86890'}
********************
********************
[array([1.e-08, 1.e+00], dtype=float32), {'base_chapters': '87347', 'base_sections': '87348,87350'}]
{'base_chapters': '87347', 'base_sections': '87350'}
********************
[array([1.0000001e-08, 1.0000000e+00], dtype=float32), {'base_chapters': '54377', 'base_sections': '54379'}]
{'base_chapters': '54377', 'base_sections': '54379'}

********************
[array([0.51389563, 0.48610437], dtype=float32), {'base_chapters': '55379', 'base_sections': '55381'}]
{'base_chapters': '55379', 'base_sections': '55380'}
********************
********************
[array([0.6488945 , 0.35110548], dtype=float32), {'base_chapters': '87018,90170,89308,87013', 'base_sections': '89314,87015,87023,87024,87029,90171'}]
{'base_chapters': '89308', 'base_sections': '89313'}

********************
[array([1.0010749e-08, 1.0000000e+00], dtype=float32), {'base_chapters': '54456', 'base_sections': '54457,54458,54459,505298'}]
{'base_chapters': '54456', 'base_sections': '54459'}



11.29  
#问题1 
 din，dien没有gru好，看喂入的数据，是章id，节id，是多值的，而不是单值的，这种，做分类或回归，不是太好做，拟合目标呢，拟合几条支线呢
并且发现，基础题还好，他的章id 一个值或者两三个值，并且比较连续，并且相差不大，;    若综合题，则不仅包括基础章节，还包括后面的章节，
这样就发现，章id 或者节id 就会有多个值，并且跨度就有点大了，
现在做的  是 mean  若多值比较连续，ok，若不连续，  数据失真，丢失数据价值，丢失有用信息，
那怎么做的，一个是 
常规做法 是 多值特征 
sum  
mean
Rnn  去学习里面的蕴含的信息， 这道题包含哪些章节，
引入外部注意力  怎么处理呢 
gpt
bert 

重要特征作为外部注意力？？

#问题2 
补3 补5  
统计发现，14天全零 一起绝尘   占比最大的， 并且其他都比较均匀，   012345 要比 678910111213    多，
14天全零 代表是 新老师，新老师在这个班级  肯定是没有历史布置作业的，
14天全零，既然代表新老师，那就可以分别处理啊，
14天全零的 
现在做的处理 是   不填充，不做处理
那可以去查表 看这个class  这个教材版本，  去看看 之前的历史


='11.12online'!$A$4:$A$67

Style
indices[28] = 250 is not in [0, 50)
indices[28] = 250 is not in [0, 100)

新增实验 
sum.35  补7 
Sum.36  补9  
Sum.37   补11       
其他模型类似
综合 bgru 补7好 




这里面的rnn_sum 不共享
Sum.50   
   1. 每天的 cha sec  原mean改成  rnn 双向的，，， 再确定下     是双向
2. today style 1000 改成 30    
3. style_his  = 300
sum.51   mean->brnn, today style 1000 改成 30 
Sum.52  mean->brnn,  today style 1000 改成 30 ,  today_style 嵌入维度128 -》256
再跑个和之前对比的
sum.53   mean->brnn,  today style 依然1000 
sum.54 ，mean，  today style 1000 改成 30 

数据量不大，可以增量训练，60天的 也足够稳定了， 
Shuffle  每天的打撒，， 看下log 是否是按 时间排序的，打散有影响吗，，60天的数据放在一起打散，还是一天的数据打散，

会影响数据 ，，我现在跑第一天的，然后在跑第60天的，  记得madis 的日志 本身散的 ，

原始日志是 严格时间的，最后生成的csv 是打散的


实验
home_dien_50 home_dien_51 home_dien_52 
以及 各个 53 54      super start  晚上


Sum50 52    12.3号 所有的已经跑完 
Sum53 54  dien50-54   开始跑12.3 
Sum53 54  dien50-54    跑完 未统计  12.5 
Din302 跑完未统计  12.5   和din30 相比稍好    12.10 跑完 未整理
Din3022 在跑下 看看补7的 怎么样，12.9跑     12.10 跑完 未整理
Din3023  补7  

新增实验
home_sum_38  补6   12.9跑完 未整理        
home_sum_39  补8   12.9跑完 未整理
home_gru_38    home_gru_38 39     12.9跑完 未整理
home_din 38 39  就不添加了
总结： 12.17整理完毕 ， 其实都差不多，暂定 仍然选择补7吧



新增实验
home_sum_60  数据增加，两个添加，
每天的训练时 需要 去年当天的数据拿过来，
历史当 14天都是0的时候，去年当天的14天历史 拿过来填充

这里的rnn_sum 共享
新增实验    
两个维护，所有共享
cha sec   跨连，   brnn sum ， 然后相加还是cancat

home_sum_61  add 
home_sum_62  concat   后 reduce_sum  结果一样   

画图


home_sum_62   只做 
style* today   
style*hist


Dsin len？？

限制到 乡镇，没有到学校
把去年今天的数据 也添加进去训练 ok
今天历史14天全为0，把去年今天的历史拿过来，ok
去年今天的数据，cha sep sty 有重复的  要去重


home_sum_60   维护两个 全局共享     
Home_Sum_61   维护两个 全局共享   bgru 那里    resnet         12 .11 跑完 ok 未整理        和sum_60 差不多，为啥都这么波动呢
home_sum_63 维护两个 全局共享    bgru 那里    resnet    再加layernorm   12.13 跑完 未整理   整理 效果不好呢

Layenorm 没有加对呢 
home_sum_64   layernorms  name=scope 不共享
Hom_sum_65  layernorm 不在with下面，和return 对齐

home_sum_66     上面的with 去掉 reuse 
home_sum_67  最终的 sum->mean 

home_sum_68  先训练去年今天，在训练今年今天

再加实验
home_sum_66  reduce_sum 不在with下面，和return 对齐   不加resent 
home_sum_67  reduce_sum 不在with下面，和return 对齐  改成 reduce_mean  不加 resent 
home_sum_  reduce_sum 不在with下面，和return 对齐   不加resent    加roi池化


home_din_62  style*today 
home_dien_62  style *today  12.11 开跑

Din
Dien  的  尺度归一化 添加

Excel 看出 sum_63 并没有表现出优势，
但是 din63 dien63  在最后11.6 5 7 直接上去了，所以 再加时间天数，
9.2-11.6  
9.2-11.24  

还没跑
home_din_60 61 63   12.15 跑完未整理  
home_dine_60 61 63   12.15 跑完未整理 

添加时间天数的实验 
12.17 整理  效果见好    


更改layernorm 位置的实验

为啥layernorm 不好呢
Bwhc  沿着 b
BTN  沿着  b  14*128 

两个with 第一个with 不用 reuse


震荡原因分析
1. 数据脏数据，，sql  limit 1    或者字典去重  ok去重
2. 学习率  改变了吗     不存在这个问题
3.  数据脏， 现在是 今天的数据训练 去年今天的数据训练，接着 今年明天的数据测试 topN，更改 先训练去年今天的数据，再今年今天的数据
4. 

增加实验
1. grade 修正  ?    不存在这个问题
2. Dense 分布 归一化    把有量纲表达式变成无量纲表达式。
3. 增加特征  学情和  问答 
4. Sum->  pool  
5. [sty,cha ,sec]  把历史 4 8 =32 的style 个数统计 拿掉

Madis 


还有问题，
怎么没有以前稳定了，以前的 最后阶段 都比较稳定了啊
现在，怎么这么波动？
Bgru 后阶段 比其他 比较稳，






teacher_id 
edition_id 
Grade 
Term 

class_id 
班级id 不会随年级而变化，

新班级，  
新老师，  看注册时间吗  r_teacher_info   有个register_day  看新老师的 多少 剩下的就是新班级
新老师的话，
即使新老师，我也没法 填充啊，原始代码里 本身就是 不考虑老师的

拿去年的填充啊啊

去年今天的 数据，   章节  重复的   限制条件，同地区桐城市， 年级     降一级   班级 取相近的


同一城市  同一年级，同一学期，同一教材版本，布置的作业
city_id   
school_id 
County_id 
province_id 
edition_id   
Grade     
Term  

去年同年级的呀， 不要减1 

喂入 以上 以及日期改为去年今日  输出  cha  


从 r_teacher_info 根据teacher_id 取出 province_id  county_id  city_id school_id   
然后 这个表 根据( 以上地理信息)  取出所有老师id 

然后到 daily_homework_info  根据 上面取到的 所有老师id    还有时间  year-1  日期， edition_id  grade  term   
找出 cha

然后 字符串匹配   两个字符串 有交集，

mysql where 条件是个列表

去年今天  一周的 数据，
章节和style  重复的  
   
 先看去年今天  和今天  比
再看去年今天所在周的   和今天的比
 前年的

SELECT * FROM daily_homework_info WHERE `day`='2019-02-01' and teacher_id in (SELECT teacher_id FROM r_teacher_info WHERE province_id=2 AND city_id=2 AND county_id=501 AND school_id=852634 ) ;

SELECT * FROM daily_homework_info WHERE `day`='2018-11-01' and teacher_id in (SELECT teacher_id FROM r_teacher_info WHERE province_id=2 AND city_id=2 AND county_id=501 ) ;

还有midas 表格



Dense 特征 做归一化，
预处理  都转浮点数  0-1   -1  1   
还有 考虑分布 


数据扩增，
去年的数据  能匹配上 肯定都是正样本，  feature那边 除了基本的去年 特征，还有 chapter_id  用今年的
还有个 就是    去年的数据  可以加个 权重，

多值 笛卡尔  huffman
Sql  有序，，section_rank
汇报  乱  没有主线，

多值特征-》 单值特征

1. 单独训练 目标函数呢，
2. Resnet    跨连，防止梯度消失，防止没学习到吗，，这里的跨连 是 相加呢  还是拼接呢  两个实验



cha_se_style  今天的 以及 历史的   历史的也可以 把style 拼起来

Style 也会发生变化，
怎么考虑呢

style* todya，，
style* his  

Dsin 加 attegru  这里量身定制
三孪生网络，多源注意力   三个din 交互   attentionscore  权重处理 
分开学习，先 style 再考虑 cha sec


='11.12online'!$A$4:$A$67
='60.61.63 l'!$A$259:$A$339

40111 —— 1专练 2 单元练习 3 小综合 4 大综合 111基础口算应用都有 章节数以基础为准 9开头的数据节id同章id，可视为无效

Dsin 在 作业推荐不适合，没有明显的session
不像madis


尝试这么多，实验结果 哪个好  理论上哪个好，  好的原因，不好的原因
一些想法 理论感觉可以提升结果的，为啥失效

池化 
roi pooling    faster rcnn 
roi align  mask rcnn

Bert 源代码 



Dense 特征 归一化，两种， ，最大值 最小值，，标准正太归一化，  不同的适用情况
分布而定

归一化，roi pooling
学情表，section_id  多个基础章节，sql表里 只取一个 ，其实不对的

句向量  三层lstm  怎么是 每一层 最后的输出  拼接
不是最后一层 最后的输出吗



11.30 

1. 看线上ok 否，两个定时任务，模型有生成没，确定总的时间，两个txt生成，模生成，传送ok，以及 异常捕获，邮件，
2. 并坐下记录，docker 命令，以及 log
3. 整理线上 预测完整py代码，，而不是命令行，
4. 这周例会 ，几个实验，跑起来，    代码改好   并且改进 要想好理由    都有哪些实验，
5.  每天的 cha sec 加个rnn  包括历史的,,双向的，因为 并不是按大小排的
6.  style  today的  是 个数，然后28个 style   
7.  补几个，14天的额外处理 mysql 再看，这个花时间，并且具体思路还没有
8. 同时测试几个  补7天的  补9天的  
9. 还有个 增加轮数
10.  
11. 
12. 在想下，训练集的acc ，，可能也是更多是 看  top10 内 包括 这个  不会太追求top1，，但是话说回来，训练集的acc 越高， 测试上 更好啊，   并且它和madis 的 acc 一般都acc 0.98      而这里最高的 只有 0.89 普遍0.70 


madis
1. 表格统计 csv 对比
2. 看项目改进的地方


ppt
述职ppt




# #####


9.6号 bug一个
训练有的不能连上，900 -》 1000  而有个bug   900 -》100 导致 8 20 失效

Supervisor  2>&1  把 错误 也重定向到日志文件
同时  每个程序添加一个本地文件， 记录训练个数

16:30 所有才stop start 


模型结构的
din
xdeepfm
deepffm

工程导向的 
ESSM deepMatch


Auc
gauc  可以
dice
ccauc

兴趣

autoInt


#嵌入特征维度  embedding_size
din  16 
dien  18
dsin  4   即使广告sum很多， 没问题啊，，，tf.float32 


兴趣  短暂的 
用户画像 哪门课程弱  哪门课程强

 see the target item

# 对行为序列建模
1. BST 
结构 ：  
输入   常规的 还包括 交叉特征
侧   
历史序列 oncatented (item id , category id)  还包括tgt ad，相关性   比时序性，但长远有依赖性，局部有随机

以上线性

三层 ffnn 非线性  1024 512 256

Dsin的 最后的ffnn
200 80  2 

dsin 中的 transformer的ffnn
嵌入大小 *4     嵌入大小

而程序中 
transfomer中的  ffnn
num_units=[2048, 512],     嵌入大小 *4     嵌入大小 *3


实验：
1 个block 优于2，3
8个头
嵌入向量  4-64 


2。bert4rec，，SASrec
Modeling users’ dynamic and evolving preferences from their historical behaviors
首先MCs 仅与前一个有关，条件独立，
rnn
attention

仅仅喂入输入序列，预测行为，没有常规稀疏特征，，？？

Mask训练，双向，增加样本，
但gap 在结尾添加特定token

在召回侧

4。强化学习
百万量级 item 用户 亿级

但从打点日志，会学到偏见，固定，从历史抽取，所以 rl

在召回侧

3。R-transfomer





#打开方式  买了又买 看了又看  探索 开发
历史点击ID   用bert
历史行为  强化学习


历史行为：历史点击，历史购买，浏览


#历史行为整合
最近十个广告id  10个广告位，10个行为


把历史点击ID  和 tgt ad   做 bert

是不是应该把 行为，定义  btag   点击 浏览 购物
而不是 userapi  

价钱 dense feature

注意力图生成



sequence-aware recommendation
item   context  session -base


在跑两个
1. 用户adlist  transformer ，，，userapi sum  参考dsin 但未有 外部注意    dsin_4  running  随机   nan  30 20 8    有值
2. 用户adlist  transformer ，，，userapi sum  参考dsin 但未有 外部注意    dsin_4.3    正余弦   30 20 8     todo  OOM  9.20 add  有值
3. 之前0.99的   userapi  concat  -2           dsin_s40    runing  logs30.csv
4. Userapi  sum;   adlist +tgtad  trasformer    参考 BST
5. 用户adlist  transformer ，再加外部注意，，userapi sum ，  dsin_4.2   随机  30 20 8     atte  先有值 后nan  s41
6. 用户adlist  transformer ，再加外部注意，，userapi sum ，  dsin_4.11  正余弦  30 20 8      todo   OOM    9.20 add.  一直有值
7. 用户adlist  transformer ，再加外部注意，，userapi sum ，  dsin_4.12  随机  16 10  4     todo  OOM         9.20 add  较多有值 后nan.  9.29 kill
8. 用户adlist  transformer ，再加外部注意，，userapi sum ，  dsin_4.13  不加  30 20 8   todo                      9.20 add 先有值后nan.    9.29 kill

9. 用户adlist  transformer ，再加外部注意，，userapi sum ，  dsin_4.14  正余弦  16 10  4     不错啊

iter: 600,loss_average:nan, accuracy_average:0.9728255208333333,loss:nan,acc:0.0

1. 不相等的嵌入特征  4  dsin_4.4  , adlist  dsin   userapi sum,     基础是 adlist dsin  userpai sum   30 20 8  随机     10.12 kill
2. 嵌入特征  8   dsin_4.5   , adlist  dsin   userapi sum     10.12 kill
3. 嵌入特征  16    dsin_4.6   , adlist  dsin   userapi sum,    10.12 kill
4. 嵌入特征  32   dsin_4.7   , adlist  dsin   userapi sum,       10.12 kill
5. 嵌入特征  64   dsin_4.8   , adlist  dsin   userapi sum,             都比较差     10.12 kill

1. dsin_4.15    adlist dsin  userpai sum   30 20 8  pe   4      gethttp
2. dsin_4.16    adlist dsin  userpai sum   30 20 8  pe   8      gethttp
3. dsin_4.17    adlist dsin  userpai sum   30 20 8  pe   16     gethttp
4. dsin_4.18    adlist dsin  userpai sum   30 20 8  pe   32
5. dsin_4.19   adlist dsin  userpai sum   30 20 8  pe   64    这个比较好 11.27
6. dsin_4.21   adlist dsin  userpai sum   30 20 8  pe   256

dsin_4.20  adlist dsin  userpai sum   30 20 8  pe   64    sum   删除 concat里的  final3    9.29 add


Userid  nums ,embedding【1000，  128 】   4 16 32  64

Mobile   [3,5]
Provice   [40,128]         4 16 32  64
City     [5000,128]        4 16 32  64
English   [6,5]   ABCDE
Math  [6,5]
Chinese   [6,5]
Grade  [102,128]             4 16 32  64   
Purchase [6,5]   A1 A4  B
Activaty  [6,5]   
Freshness 新鲜度  [8,5]   A-G
Hour  [25,5]

Mid item  [100k,128]     4 16 32  64

Adlist      dsin  attention  hidden_units 128   ->  4 16 32  64   

adlable    [20,40 ]  ->128         4 16 32  64
advalue    

Userapi 
ABC    [100,200,800],->128      4 16 32  64


实验：
嵌入特征向量长度：
1. dsin_4.4  -》dsin_4.8    和 dsin_4.2   128 
看正余弦和随机
1. dsin_4和 dsin_4.3       
2.  4.2 和4.11   4.13
看训练参数 全量
1. 4.2  和 4.12    4.4and  4.14
看之前最好的是否随机  concat -1  -2
  1 . 之前0.99 的再跑一边   s40
 




Ss
兴趣抽取

####DSIN

你好，请问一下，论文中只是使用了用户的点击序列，那么用户的收藏序列或者购买序列是否也能加进去？如果多个序列的话应该是不能共享参数的？

我们尝试过其他序列的建模，都是负向效果。我们不确定如果一个用户买过XX,是不是应该给他再推类似的东西? 收藏的话过于稀疏了，也不好用。
PS 你可以看看ATRANK 是专门做 heterogenous sequence rank 的。



todolist：
1. 广告id  点击广告idlist ，  sku  类别    拼接一起
2. 历史序列  sku  类别
3. Ppt  各种模型  时间线   最好表现，是论文中的 数据中，，， 还是在自己数据上跑出来的 scor




Multihead 中屏蔽了 drop，但不是因为这个 导致的  auc0.99的与 auc0.5  在train的数据中，就表现差的一个量级   

---train--- day:2019-09-15, iter: 6500,loss_average:0.011299126702702531, accuracy_average:0.9923416877526503,loss:0.0014480622485280037,acc:1.0

---train--- day:2019-08-16, iter: 7100,loss_average:0.12443515291186949, accuracy_average:0.9863646121176196,loss:-0.0,acc:1.0



oom：
batch_size,  gpu个数设置


三个bug
1. 写外部注意力，调deepctr api 折腾两天
2. TensorFlow升级1.14   logging 不输出日志了      TensorFlow1.12  dropout失效
3. Oom    ，保存两三天的 down掉



两个excel
1. 参数打满的与参数未满的对比，
2. 嵌入特征向量长度对比


Session的现实意义 实际意义      为什么

作业推荐  历史作业 session   序列特征

两个bug
1. 网络问题
2. 好多出现nan  刚开始 有值，后面 nan了




26-09-2019 12:25:14 urllib3.connectionpool:DEBUG:https://pypi.python.org:443 "GET /pypi/deepctr/json HTTP/1.1" 301 122
26-09-2019 12:25:14 urllib3.connectionpool:DEBUG:Starting new HTTPS connection (1): pypi.org:443
26-09-2019 12:25:15 urllib3.connectionpool:DEBUG:https://pypi.org:443 "GET /pypi/deepctr/json HTTP/1.1" 200 7556
26-09-2019 12:25:24 urllib3.connectionpool:DEBUG:Starting new HTTPS connection (1): pypi.python.org:443
26-09-2019 12:25:26 urllib3.connectionpool:DEBUG:https://pypi.python.org:443 "GET /pypi/deepctr/json HTTP/1.1" 301 122
26-09-2019 12:25:26 urllib3.connectionpool:DEBUG:Starting new HTTPS connection (1): pypi.org:443
26-09-2019 12:25:27 urllib3.connectionpool:DEBUG:https://pypi.org:443 "GET /pypi/deepctr/json HTTP/1.1" 200 7556

代码问题，不一致，

72c72
< EMBEDDING_DIM = 16
---
> EMBEDDING_DIM = 32
394c394
<             hidden_units = 16 #嵌入向量长度  原为128  
---
>             hidden_units = 32 #嵌入向量长度  原为128  











# #######

现在的rnn bgru 都是多对多，，输入 b*10*200   ，unit=200，输出也是 b*10*200 
所以才 reduce_sum 
双向的  是吧最后一层的所有输出 正向  反向的 拼接起来  
 b*10*200      unit=256    b*10*512


Block   1  3  6
Head   2  4   8
Unit   128  

9种

其他停到  superviosr 文件删除    s2-s7


base_sum    
 1.  466  concat  -2  》 -1      rm  模型 

gru 
1. 486   concat  -1    rm  模型      
2. B*10*384   384  384（其他bug，可以其他方式）    B*None*384   
3. gru 那块    HIDDEN_DIM*3  

bgru
 1.  486   concat -1  rm 模型  
2. B*10*384   128*2   128*2（可以不用一致）     输出  B*None*512


dsin.s   
1. 451     concat   -1  
   2. 多头    num_units = hidden_units*3,
  3.   feed   [4 * hidden_units,hidden_units]
[6 * hidden_units,hidden_units*3]

ds_sum  30 20 8
ds_gru  30 20 8
ds_bgru  30 20 8

复制  改参数     s2-s9
Block   1  3  6
Head   2  4   8
Unit   128  

S1-s9  训练超参数 都是 30，20，8
128   1  2    s   pe   9.20 keep.  9.29 kill
 128   1  4  s2       9.20 kill
128   1  8              9.20 kill

128   3 2    s4       9.20 keep   9.29 kill 
 128   3 4              9.20 kill
128   3  8              9.20 kill

128   6 2    s7        9.20 kill
 128   6 4               9.20 kill
128   6  8   s9        9.20 kill

128 1 2 s10  be  未做  从 s20 开始

512   1,2   s10  xxxxx    太大了
512   6,8   s11   xxxxxxxx

256   1,2   s12   xxxxxxxx
256   6,8   s13   xxxxxxxx

S10  - s  训练超参数 是  16，10，4  加了5个
128  1  2   16 10 4     s10                     9.20 kill
128  1  4   16 10 4     s11                     9.20 kill
128  3  2     。。。    S12                    9.20 kill
128  3  4     。。。    S13                    9.20 kill
128  6  2     。。。    S14   xxxxxxx      9.20 kill


S20  pe-》be.     9.29 kill
128 1 2  16 10 4  s20 

S21  pe-》不加   9.20 add    9.29 kill
128 1 2  16 10 4     

看全量训练参数
1. S 和s10 对比  重点
2. S2和s11  对比 
3. S4 和 s12
4. S5 和  s13
5. S7 和  s14 
看transformer 参数
1.  s s2  -》  s9  对比 
2. S10  -》 s14 对比
看位置编码
1. S20 和s10   s21


、
Dsin.ns

Ns4 stop   512  6 8  30,20,8      为啥没有test 值，，cal_auc 0   0 
Ns2  stop   512  1,2   16,10,6
Ns stop  512  1,2   10,6 4     也在跑
怎么rm啊


Ns5 有的有数据  128  6 8  30,20,8   刚开始有数据  后来没数据了     9.19 数据0.5  Kill.    有些0.5 有些 0.98
Ns3 一直有数据  128  1,2  10,6,4     一直有数据.   9.19 数据还行 0.98 
在跑个 
Ns.sum    30 20 8  running   有缺失天数？看log     9.19  数据还行 0.98 0.99

Ns   512  1,2   10,6 4   running     9,19  数据弱  0.5    kill
Ns6 128  1  2  30 20 8  running    9.19  数据还行 0.98 
:
看全量训练参数
1. NS3 和 NS6 对比  看训练参数打满  全量
看128  512
  1.  ns和ns3
2. Ns4 和ns5
3. ns2和ns3
看6 8 1  2 
1. Ns5 和ns6




应该在跑个四个
和ns5对比
Ns8   128  6 8  10 6 4
和sum对比
Ns.sum2   16 10 4
和ns3对比
Ns7 128  6 8  10,6,4
Ns6 128  1  2  30 20 8 最想跑的   



Ns  128  1,2 30,20,8  加个模型  未做
Ns  128  30 20 8  sum     
Ns  128  30 20 8  gru   未做
Ns  128  30 20 8  bgru   未做

Bert  
1. drop  
2. mask
3. 正余弦     随机  好处 

大的注意力分类，
注意力函数汇总
位置编码 汇总   
多头后面怎么处理 concat
文件读写锁  锁 程序中断， 写的时候 程序中断   会不会造成  

读写锁，读写分离，高并发 高可用

读锁  
写锁 互斥类的


总结：：

9.2下午3点，分开三个json文件，ok 一直ok
当时10+2  个程序     dsin.s   22min 100轮     22*160/24/60  = 2.4
4500 17h  16000   =2.5

9.3  中午12点 加了  6+2 个程序     原来的变成 35min  100，  = 3.8   
新加的 每100轮   20min   =  2.2天

原始的参数都打满了，现在serving  4-6    8的话 就开始test
新加的    2-4   4的话开始testig




dsin 
Bert4rec
用户点击广告id 历史
用户历史行为  api

dice  loss


用户历史 api  那点击的广告 是不是也需要知道他的  api  同一个嵌入空间下，才能做外部注意力吧


Supervisor  stop  在 start  从头开始了 没有 从4500开始
stop  update  start all

restart 应该是重头啊

stop start  感觉像重启啊



超过了当天，，一直8.30

145 -1 //4  = 36        7.25 +6  = 7.31     8.30
把 model 和serving 都改成  145    先stop 在start



cbow 的word2vec 代码 

dsin 的代码  deepctr  0.4.1   的库使用


Deepctr 0.6.1  源代码修改
1. layers文件夹内的  sequence.py   line176
            print("xxx")
            #key_masks = tf.expand_dims(mask[-1], axis=1)
            key_masks = tf.expand_dims(mask, axis=1)
            print("yyx")


2. layers下的  core.py  line 10
        #zkl9.16
        #keys_len = keys.get_shape()[1]
        #keys_len = keys.get_shape().as_list()[1]
        keys_len = tf.shape(keys)[1]
        #queries = K.repeat_elements(query, keys_len, 1)

        queries = tf.tile(query, [1, tf.shape(keys)[1],1])
           
        queries = tf.reshape(queries, tf.shape(keys))




# ######
张介  蒙特卡洛模拟 数学科学组  
组长tms
扩科   p值
三门问题  封闭解
chaosblade

玉林 consul  微服务  数据科学组
服务发现工具   类似 zookeeper   
landscape. cncf.io  
云原生


李爽 实时推荐   算法策略组
大数据： 量级，内容（年龄，等）
存储： 打点 kafka消息队列 +Hadoop  hive hbase
处理 flutter  flink  spark  批处理 流处理
模型更新：分布式训练


实时特征
最近行为
流处理 flink  三个job   用户/物品 上下文
存储，kv + 列存储  hbase  随时添加删减 重要特征

实时更新
增量训练集 
增量更新  复杂 
tf.serving 服务

推荐
多层embedding   三维  多级

训练 
批处理
Flink ： 在线 离线   在线表 ，，，离线表 增量更新
inference 10ms  还不需要分布式训练

服务：
grace restart   没有断点的 服务重启


用户行为 序列行为  每天的历史行为 偏向性

doubleaim  rpm + ctr  
Ntm npi  图灵  序列  上千种行为 
自动编码  npi    神经程序解释器
可解释模型


每时每刻都知道，成交量，而不是每天固定时间 计算


张望 AI   AI研发组  ocr nlp
AI变脸
人脸对齐 换脸 人脸融合

x先 
A-1  —2— A
B —1—3—B  

U型  表情 结构  共享   -》  纹理 细节   不共享

然后 
A—1  —3—B


AI写诗 
写宋词
写作文
transformer   4个预测1个  滑动窗口   25个 4s

Inference 时间  出来一个字 多长时间

keynote

作文判分



直播 java 


阿里 混沌工程   开源  








num_blocks = 6 
            num_heads = 8

适当减小,  区别  减少哪个  可以吧训练速度提上去


Ppt 
参数调整 ，跟上

git rm

两个模型 位置编码不用正余弦

还有个bug  可能训练的不够，col  预测的正例为0  
然后就保存不了  csv，
也可能测试的不多
是不是没有加dropout
， 多头注意力的 dropout用到  is_training，早就删除了


保存不了csv中，
1. 预测的正例个数为0 ，没法除以  所以跳出
2. 批次 不等于 128     训练和测试 从hbase 从取不是一个批次，有的 128  80 ，，默认 128
tf.shape(x)[0]

自注意力那里 
Heads  要被512 整除，，，8 512  64
6 512   error     4 2 可以

block  6 1 

两种位置编码的区别
正余弦

	
	
修改：
1. label_all  Padding 0 
2. save iiter
3. except logging.debug("test cnt {}\n,error {}".format(cnt,e))
- [ ] logging.debug("train iiter{}\n,error {}".format(iiter,e))

	
	
gru_dien 都是 阿里的dynamic_rnn

base_dien   tf.nn.dynamic_rnn

tf.name_scope(  重复的

Restore

Csv中的时间有些乱的，是 中间有段，又重新开始的原因？


现在做的仅仅是 广告点击预测，还没有做到 广告推荐哪一步，

Auc 仅仅是分类   把 点击为正的 排在 不会点击的广告 上面，

都点击的样本 之间 怎么排序，，精排序

千人前面

最终  排序损失 A/B test 线上测试



todolist
1. 训练有时中断，
法1：进程监控 重启，supervisor   
法2：查询log 最后一条 时间，等待半个小时没有变化，就重新 开启
2. 添加日志   context 下的  最近十分钟内的最近的十个行为，key：rencent_behavior     apiurl
3. 正确率： 昨天的 最近7的  最近28的 
4. 对孩子的关心程度，

特征权重，怎么加 ，嵌入向量那地方，，但是加多少呢
  


# #########
import datetime

Restore self

From  now day


Iieter-0

Version   


7.1 表 midas_offline  ok
7.25 7.26  midas_offline error


7.25 7.26  midas_offline_v1  ok

Super 研究 同时多个 ，听一个  其他不停


gru_dien 

原来  训练 8*2000    
测试  30000

训练 640       20*600    12000    16000
测试 1200     2000轮挂掉

保存模型 每2000  不用改

改为  每  200 

表 midas_offline
midas_offline_v1


对异常退出的，用supervisor可以实现无限次的 进程重启
对正常退出的，startretries=10  可以10次重启，

一个问题：多个进程同时跑的话，若某个代码想修改，怎么关掉某个进程  单独关掉  其他正常运行，
还有一个问题supervisorctl 命令不能用
 supervisorctl status
unix:///tmp/supervisor.sock refused connection


引入 suopervisor后，保存的测试结果，csv里面的天 能连接起来了了
但又出来一个问题，保存模型文件夹下的 serving  是正常的，仍然只保存一个文件夹， 但是model  文件夹 不正常的，模型都有保存，之前的没有删除，

问题：
多个文件一个logging日志文件 
logging日志 append  不覆盖  “w”->“a”

os.path.exists(“ /data//12.* ”)  *怎么不起作用
os.system(“rm /data/12.*”) 这个起作用


每种模型 到底 每天 可以训练几天的数据量


写一个配置文件，所有参数，超参数配置文件 json

用supervisor 后 cpu的利用率 20  原来200，有的时候，



Userapi  特征种类总共多少个， 字符串索引
历史上最近的十个行为，这10个行为有重复的

 b'context:rencent_behavior': b'[/user/notice/list, /parent/device/push-register-code, /homework/exam/get-detail-popwindow, /collection/homework/get-urge-homework-number, /parent/homework/check-match-report, /html5/weekly-report/math, /parent/homework/unread-new, /parent/homework/get-single-report, /parent/homework/get-single-report, /parent/homework/unread-new]',

库 map kv


参考selfatten   和dsin，，

怎么匹配  import re

10个值是保证的，不会有丢失值吧
每个网址 加引号，
Userapi 有空值 ，应该是程序上的吧， 把空值 赋值0
总共多少个
三个斜杠  每个大约多少个  相乘
训练两天看看，保存的userapi 
检索 第一个斜杠，保存 去重 len
检索第二个斜杠   去重 
检索第三个  去重   



为啥年龄等级 城市等级 都从1 开始 ，  从嵌入矩阵索引 可以从0开始啊， 索引0 转为one-hot 10000，
难道索引0 是给缺失值准备的， 是的

-1 转换为one-hot 才为 00000


tmp.append(getattr(row, I，-1))  
if k in self.field:
 if len(v) == 0:break


                         
Din dien dsin 除了模型框架大的改变 ，还有一些实用的工程技术手段  
比如 dice  什么正则等



上线的   多进程 多线程？
线下 单进程拿数据 ？
 


A/B/C
Userapi 两种处理方式：
1. 看做一个整体，大约500个种类  5000    B*10*128
2. 分开，A多少类，B多少类，C多少类  三个嵌入矩阵，然后计算结果在拼接起来，B*30*128
3. 
Userapi的 从2 开始，空值填1 ，其他0

userapi 接的模型结构
1. bilstm
2. self-attention  transformer


Model 下的模型保持删除，


Csv数据存在跳变，，为啥，下降很快
即使重新开始训练，也是restore啊，也不会突变那么大啊，
学习率？

Csv数据 有restore，但有的可以接上 天数断了话，，有的接不上，，有的天数丢失
为啥不一致呢

Userapi A/B/C  不包括“”，499，     split  35/75/358    468 
200 200 1000 


文件赶紧固定下来，其他城测试就笃定了，，

sum  1024  

超过了当前  加个判断

none none  tf.data 


Restore 没有怎么办


Dataday 到 当前日期  ，超过了当前日期
天一样，秒不一样，
天不一样，秒肯定不一样

先判断天，天一样，过，秒不更新
天不一样  秒才更新


1. 时间修改，超过当天
2. Model 删除以前的   类似serving
3. 加traceback logg
4. userapi 整体  500 

5. Userapi 三段，保存到同一个文件，json ，第一次 读，第二次读写更新，
6. Json 添加  ’‘ 空值 保证  考虑进来，  空值 为1  其他从2开始
7. 三段 也有可能不是 10个的，，不定长，，处理bug
8. supervisor restore
9. Json 保留一个 就固定下来吧，其他的  按这个索引？？？
10. 多个模型文件  py文件， ，共享一个josn文件  多进程 操作文本

并发：
互斥锁
文件锁
文件读写锁 自己实现
读写锁
文件锁，

1. Userapi  最后 sum   B*128-》  B*512    嵌入长度增加
2. 在最后 inp concat 哪里，不同的特征，不同的长度，不同的权重，引入权重？？？？


9. 35、75、358   -》 100  200  800 嵌入空间

4个模型，都是在 广告 sum  用户点击历史  dien 上 基础上添加的
1. userapi 三个 拼接， sum  128->512
2. userpai 三个split  拼接   bgru   128->512
3. userapi  三个split concat  gru  128->512
4. userapi 整体  transformer的encoder  
5. userapi 三个  transformer的 encoder     128->512
6. 把广告的那个 改成64 ？？
7. 加权重
8. 或者单独 加两层dense 128->512->128     或者  512-》128

添加 self-attention有个问题，，占位符，，[None,None]
位置编码需要 知道，第一个None我可以 提前定义，，
第二个呢  动态获取，每个批次的最大长度，时间维度的  
有补零，10  10  10   30     可以提前定义吗？？？

Import traceback 
Save
 Logg
Whie


一个py文件 保存 json  ok
几个py文件 共享  加锁  出现  @字符  数据ok

Git版本冲突，两个分支   导致 一个 gitadd  覆盖另一个的git add


Userapi 的十个最近的行为，self-attention  单向gru
Userapi  可以向用户点击idlist  做 常规注意力机制， 引入用户点击id

 
还有一个bug 到当前天的时候，连续到当天，过一天 ，训练，跳出
过一天，训练，跳出，
没有测试了，







# ######


Lr0.6 lr0.99  6个模型20天都在跑 
Lr0.8 gru一个模型在跑20天的

一般默认的 0.99

Lr0.5 虽有上扬，但应该偶然，第一次的全下降，好几次都是下降，只是有上扬的 赶紧保存了


Drop有bug，训练一天20测试一天21，然后就出bug,  

rnn.lr0.8中断, 31号凌晨4点断掉 终止
Blstm.lr0.8 最后一天降，那再跑一天
bgru.lr0.8 缺一天


感觉6.26那天的数据有脏数据，base.lr0.6 blstm.lr.0.8 再那一天都降，跳过6.26

：
粗排  初步验证  找到 大致哪些模型 
精排

有些 测试kill的早或者 凌晨断掉  天数不够的 ，后面有时间再补上，restore


修改
1. logging名字
2. Csv名字
3. 学习率decay
4. Test: cnt>=30000
5. train: break_sum 12  8 
6. Day: Jul  1    25
7. 只保存一个模型，
8. Save_serving  添加广告图片特征，build_tensor_info

很有必要上gpu了

模型达到最好的 就停止训练了，分数阈值   早停
若达到最好的，若在训练 显然会过拟合，
一个月的天数来训练一个模型，一个月热更新

若要一直训练，真的要上强化学习，

会发现 lr0.8 20 天 后面也会下降，其实这个时候 就可以 分数超过阈值了，学习率不变了，固定住，或者不再训练了，
类似tf.train.piecewise_constant
当低于某个  重新设置


Ctr中正负样本不均衡，数据分布，预测分布 
过采样，权重，
预测提高阈值，
Dice损失函数
不同指标


自注意力  图片  用户点击历史
用户行为历史，
session 

并发量 连接池
 
TODO：
作业推荐-》模型上深度学习
召回 + 过滤      模型预测
4点重启 加载模型，tf接口    重构 迁移

交互特征  特征少，  flink
历史的作业   序列化   
老师的偏好   大于客观事实   规则

kafka 

学情 作业 在库里 



8月  
1. 模型迭代： 图片+attetion 其他特征
2. 清晰数据 random  shuffle
3. 实时更新

   
  
下周 
三个模型  上线    代码 review

base 
gruatte


Model   base_din   base(reduce_sum 广告图片 )+ din(历史点击)
Modelgru gru_din
Modeldien  base_dien
Modelgrudien  gru_dien

四种学习率 0.5 0.6 0.8 0.99

base_din 0.8 keep_max =1
base_din 0.6 keep_max = None 
base_din 0.5 keep_max =0
 
update_model_2.base_din.py
update_model_2.base_dien.py
update_model_2.gru.din.py
update_model_2.gru.dien.py



三个bug
1. 原始代码我以为是模型只保留最新的， 不是，，把 tf.train.Saver() 设置全局 或者 self 类变量
2. 改好后，model文件夹 竟然没有保存，best-model可以保存最新的，serving仍然是每个都保存，，，关闭best-model，serving 删除以前的
3. 训练过程中，除了reshapebug，还有个bug，lable_ad  映射  嵌入特征不够，20 40 label value 
4.  indices[72,0] = -1 is not in [0, 20)    把max_label  总共7个label啊，，怎么会出现72 71 70
5. 
6. 3万次 test 有时中间有bug，要跳出来，继续train
7.  


TensorArray has size zero, but element shape [?,128] is not fully defined. Currently only static shapes are supported when packing zero-size TensorArrays.
         [[node rnn_1/gru1/TensorArrayStack/TensorArrayGatherV3 (defined at /home/zhangkl/zhangkailin/midas/Midas_Engine/rnn.py:787)  = TensorArrayGatherV3[dtype=DT_FLOAT, element_shape=[?,128], _device="/job:localhost/replica:0/task:0/device:CPU:0"](rnn_1/gru1/TensorArray, rnn_1/gru1/TensorArrayStack/range, rnn_1/gru1/while/Exit_1)]]

Caused by op 'rnn_1/gru1/TensorArrayStack/TensorArrayGatherV3', defined at:
  File "update_model_2.gru_dien.py", line 835, in <module>
    os.makedirs(paths)
  File "update_model_2.gru_dien.py", line 376, in __init__
    rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_his_eb, 
这个错误貌似 跳不过去呢






# #####


(b'10000022_101809', {b'context:dexposure_alocation': b'5', b'context:exe_time': 
b'2019-06-19 18:50:03', b'context:rclick_ad': b'[]', b'context:dexposure_clocation': b'1', 
b'context:hclick_similarad': b'0', b'context:dclick_otherad': b'0', b'context:count_click': b'0', 
b'context:hexposure_alocation': b'5', b'context:window_otherad': b'0', 
b'context:hexposure_clocation': b'1', b'context:exposure_duration': b'0', b'context:is_click': b'0', 
b'context:yeaterday_accuracy': b'0.0', b'context:log_week': b'2019/06/17',
 b'context:log_hourtime': b'18', b'context:log_day': b'2019/06/19', 
 b'context:location_ad': b'10', b'context:week_accuracy': b'0.0', 
 b'context:rclick_category': b'[]', b'context:hexposure_similarad': b'1', 
 b'context:duplicate_tag': b'0', b'context:log_month': b'2019/06/01', 
 b'context:log_time': b'2019-06-19 18:49:52', b'context:log_weektime': b'4', 
 'context:month_accuracy': b'0.0'})
(b'10000022_101925', {b'context:dexposure_alocation': b'17', b'context:exe_time': b'2019-06-19 19:06:58', b'context:rclick_ad': b'[]', b'context:dexposure_clocation': b'17', b'context:hclick_similarad': b'0', b'context:dclick_otherad': b'0', b'context:count_click': b'0', b'context:hexposure_alocation': b'17', b'context:window_otherad': b'0', b'context:hexposure_clocation': b'17', b'context:exposure_duration': b'0', b'context:is_click': b'0', b'context:yeaterday_accuracy': b'0.0', b'context:log_week': b'2019/06/17', b'context:log_hourtime': b'19', b'context:log_day': b'2019/06/19', b'context:location_ad': b'6', b'context:week_accuracy': b'0.0', b'context:rclick_category': b'[]', b'context:hexposure_similarad': b'40', b'context:duplicate_tag': b'1', b'context:log_month': b'2019/06/01', b'context:log_time': b'2019-06-19 19:06:48', b'context:log_weektime': b'4', b'context:month_accuracy': b'0.0'})





# ########

import happybase
import contextlib


@contextlib.contextmanager
def hbase(**kwargs):
    conn = happybase.Connection(**kwargs)
    conn.open()
    yield conn
    conn.close


mine = "10.9.75.202"
filter_str = """RowFilter (=, 'substring:{}')"""
"MIDAS_RECENT_CLICK_PRO"
"midas_ctr_pro"

"midas_online_user"
"midas_online_context"
"midas_online_ad"

# happybase.Table().regions()

if __name__ == "__main__":
    cnt = 0
    with hbase(host=mine) as conn:
        table = conn.table("midas_online_context")
        for i in table.scan():
            print(i)
            cnt += 1
            if cnt >= 100:
                break

(b'10002337_103465', {b'context:duplicate_tag': b'0', 
b'context:count_click': b'0', b'context:hexposure_alocation': b'7', 
b'context:window_otherad': b'0', b'context:hexposure_similarad': b'53', 
b'context:month_accuracy': b'0.0', b'context:dexposure_clocation': b'3', 
b'context:week_accuracy': b'0.0', b'context:dclick_otherad': b'0', 
b'context:log_week': b'2019/07/01', b'context:log_weektime': b'7', 
b'context:exposure_duration': b'0', b'context:is_click': b'0', 
b'context:hclick_similarad': b'1', b'context:exe_time': b'2019-07-06 21:09:28', 
b'context:log_month': b'2019/07/01', b'context:log_hourtime': b'21', 
b'context:rclick_ad': b'[101590, 102169]', b'context:location_ad': 
b'7', b'context:hexposure_clocation': b'6', b'context:dexposure_alocation': 
b'3', b'context:log_time': b'2019-07-06 21:09:17', b'context:rclick_category':
 b'[7, 5]', b'context:log_day': b'2019/07/06', b'context:yeaterday_accuracy': b'0.0'})

(b'10002337_103529', {b'context:duplicate_tag': b'0', b'context:count_click': b'0', 
b'context:hexposure_alocation': b'2', b'context:window_otherad': b'0', 
b'context:hexposure_similarad': b'78', b'context:month_accuracy': b'0.0', 
b'context:dexposure_clocation': b'2', b'context:week_accuracy': b'0.0', 
b'context:dclick_otherad': b'0', b'context:log_week': b'2019/07/01', 
b'context:log_weektime': b'6', b'context:exposure_duration': b'0', 
b'context:is_click': b'0', b'context:hclick_similarad': b'0', b'context:exe_time':
 b'2019-07-05 06:46:28', b'context:log_month': b'2019/07/01', b'context:log_hourtime': 
 b'6', b'context:rclick_ad': b'[101590, 102169]', b'context:location_ad': b'6', 
 b'context:hexposure_clocation': b'2', b'context:dexposure_alocation': b'2', 
 b'context:log_time': b'2019-07-05 06:46:13', b'context:rclick_category': b'[7, 5]', 
 b'context:log_day': b'2019/07/05', b'context:yeaterday_accuracy': b'0.0'})


userlog
(b'10000022', {b'user:app_freshness': b'G', b'user:county_id': b'1558', 
 b'user:city_id': b'181', b'user:app_type': b'3', b'user:chinese_ability': b'0', 
 b'user:english_ability': b'0', b'user:test_timestamp': b'1560940904729', 
 b'user:province_id': b'13', b'user:school_id': b'54085', b'user:purchase_power': b'B', 
 b'user:mobile_type': b'OPPO_R11s;7.1.1', b'user:mobile_os': b'1', b'user:activity_degree': b'E',
  b'user:grade_id': b'4', b'user:math_ability': b'E', b'user:user_id': b'10000022'})
(b'10000048', {b'user:app_freshness': b'G', b'user:county_id': b'1856', b'user:city_id': b'221', b'user:app_type': b'3', b'user:chinese_ability': b'0', b'user:english_ability': b'0', b'user:test_timestamp': b'1562471120571', b'user:province_id': b'16', b'user:school_id': b'861332', b'user:purchase_power': b'B', b'user:mobile_type': b'PAAT00;8.1.0', b'user:mobile_os': b'1', b'user:activity_degree': b'C', b'user:grade_id': b'4', b'user:math_ability': b'B', b'user:user_id': b'10000048'})
(b'10000237', {b'user:app_freshness': b'G', b'user:county_id': b'1086', b'user:city_id': b'138', b'user:app_type': b'3', b'user:chinese_ability': b'0', b'user:english_ability': b'0', b'user:test_timestamp': b'1562408881967', b'user:province_id': b'10', b'user:school_id': b'132854', b'user:purchase_power': b'B', b'user:mobile_type': b'OPPO_A77;7.1.1', b'user:mobile_os': b'1', b'user:activity_degree': b'D', b'user:grade_id': b'5', b'user:math_ability': b'B', b'user:user_id': b'10000237'})
(b'10000369', {b'user:app_freshness': b'G', b'user:county_id': b'1266', b'user:city_id': b'149', b'user:app_type': b'3', b'user:chinese_ability': b'0', b'user:english_ability': b'0', b'user:test_timestamp': b'1562114308748', b'user:province_id': b'11', b'user:school_id': b'929064', b'user:purchase_power': b'B', b'user:mobile_type': b'MI_5X;7.1.2', b'user:mobile_os': b'1', b'user:activity_degree': b'E', b'user:grade_id': b'2', b'user:math_ability': b'D', b'user:user_id': b'10000369'})

contextlog
(b'10000022_101809', {b'context:dexposure_alocation': b'5', b'context:exe_time': 
b'2019-06-19 18:50:03', b'context:rclick_ad': b'[]', b'context:dexposure_clocation': b'1', 
b'context:hclick_similarad': b'0', b'context:dclick_otherad': b'0', b'context:count_click': b'0', 
b'context:hexposure_alocation': b'5', b'context:window_otherad': b'0', 
b'context:hexposure_clocation': b'1', b'context:exposure_duration': b'0', b'context:is_click': b'0', 
b'context:yeaterday_accuracy': b'0.0', b'context:log_week': b'2019/06/17',
 b'context:log_hourtime': b'18', b'context:log_day': b'2019/06/19', 
 b'context:location_ad': b'10', b'context:week_accuracy': b'0.0', 
 b'context:rclick_category': b'[]', b'context:hexposure_similarad': b'1', 
 b'context:duplicate_tag': b'0', b'context:log_month': b'2019/06/01', 
 b'context:log_time': b'2019-06-19 18:49:52', b'context:log_weektime': b'4', 
 'context:month_accuracy': b'0.0'})
(b'10000022_101925', {b'context:dexposure_alocation': b'17', b'context:exe_time': b'2019-06-19 19:06:58', b'context:rclick_ad': b'[]', b'context:dexposure_clocation': b'17', b'context:hclick_similarad': b'0', b'context:dclick_otherad': b'0', b'context:count_click': b'0', b'context:hexposure_alocation': b'17', b'context:window_otherad': b'0', b'context:hexposure_clocation': b'17', b'context:exposure_duration': b'0', b'context:is_click': b'0', b'context:yeaterday_accuracy': b'0.0', b'context:log_week': b'2019/06/17', b'context:log_hourtime': b'19', b'context:log_day': b'2019/06/19', b'context:location_ad': b'6', b'context:week_accuracy': b'0.0', b'context:rclick_category': b'[]', b'context:hexposure_similarad': b'40', b'context:duplicate_tag': b'1', b'context:log_month': b'2019/06/01', b'context:log_time': b'2019-06-19 19:06:48', b'context:log_weektime': b'4', b'context:month_accuracy': b'0.0'})

adlog

(b'100198', {b'ad:alldexposure_alocation': b'1', b'ad:alldexposure_clocation': b'1',
 b'ad:location_ad': b'3', b'ad:label_3': b'4', b'ad:allhexposure_alocation': b'2', 
 b'ad:exposure_duration': b'97', b'ad:label_2': b'4', b'ad:window_otherad': b'0', 
 b'ad:label_1': b'1', b'ad:count_click': b'0', b'ad:label_7': b'-1', 
 b'ad:test_timestamp': b'1561433406087', b'ad:ad_id': b'100198', 
 b'ad:allhexposure_clocation': b'17', b'ad:label_5': b'-1', b'ad:label_4': 
 b'-1', b'ad:label_6': b'-1'})
(b'100199', {b'ad:alldexposure_alocation': b'0', b'ad:alldexposure_clocation': b'0', b'ad:location_ad': b'3', b'ad:label_3': b'1', b'ad:allhexposure_alocation': b'6', b'ad:exposure_duration': b'109', b'ad:label_2': b'3', b'ad:window_otherad': b'0', b'ad:label_1': b'3', b'ad:count_click': b'1', b'ad:label_7': b'-1', b'ad:test_timestamp': b'1562428833586', b'ad:ad_id': b'100199', b'ad:allhexposure_clocation': b'6', b'ad:label_5': b'-1', b'ad:label_4': b'-1', b'ad:label_6': b'-1'})



