在ipython中执行 python文件
In [103]: %run ~/zklcode/knowbox/grid.py

随机的选择、固定间隔值或手动选择。三者分别称为随机搜索、网格搜索和手动搜索
贝叶斯优化也可以用来调整超参数。其中，用高斯过程定义了一个采集函数。高斯过程使用一组先前评估的参数和得出的精度来假定未观察到的参数。采集函数使用这一信息来推测下一组参数

#2019.7.24
测试集每天都在下降，感觉是过拟合了，   过拟合 训练精度一直降，测试精度反而由下降转上升了，
训练集上不好：新的激活函数，自适应学习率
测试集不好： 早听，正则  dropout

学习率 指数衰减
初始化 凯明初始化，高斯初始化

tensorboard --logdir=run1:"/home/.../summary",run2:"/home/.../summary" 

样本不平衡，smote
负样本不够，随机采样
正样本不够，过采样， 权重

BN层，
定长不定长
、
要赶紧掌握的项目
zklcode/dien 数据是amazon
zklcode/DSIN 数据是阿里的 


eval函数就是实现list、dict、tuple与str之间的转化
str函数把list，dict，tuple转为为字符串


tf.train.exponential_decay 在测试时怎么标识啊
decayed_lr = lr* dacay_rate^(global_step/decay_steps)


2019-06-21,0.04038206225106948,0.5065926439972241,0.07480146870463668,0.005485298,0.9765625

本地断上传所有代码，常规即可，，，服务器只更新代码文件，数据 日志保留 git pull 即可
#本地主分支，本地分支  远程分支 远程主分支
git checkout -b test 新建分支
git branch --all
git checkout test  切换分支
git push origin test 
git clone -b 分支名v2.8.1 xxxx  


git add fff
git commit -m ""  能连写吗

git commit -a ""   mean 

git只更新单个文件 测试有效
git fetch
git checkout origin/zhangkl -- xx/xx.py 

git submodule init
git clone --recursive

only test
git remote add test https://gitee.com/zhangkailin/din_demo.git
git fetch
git pull https://gitee.com/zhangkailin/din_demo.git test:master


python3 update_model_2.cp.py 2>&1 | tee test_metric2.log
nohup python -u myscript.py params1 > nohup.out 2>&1 & 

nohup python3 -u update_model_2.cp.py > test_metric8.log 2>&1 &
/usr/bin/python3 -u xx.py  >> out.log 2>&1

nohup sh **.sh > /dev/null 2>&1 &

& ： 指在后台运行
nohup ： 不挂断的运行，注意并没有后台运行的功能  no hang up 的缩写，就是不挂断

cpu_config = tf.ConfigProto(intra_op_parallelism_threads = 8, inter_op_parallelism_threads = 8, device_count = {'CPU': 8})
with tf.Session(config = cpu_config) as sess:


config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.4  #占用40%显存
sess = tf.Session(config=config)


saver = tf.train.Saver(max_to_keep=3)
        saver.save(sess, save_path=path)

run_config = tf.estimator.RunConfig(save_checkpoints_secs=1e9,keep_checkpoint_max = 10)
  model = tf.estimator.Estimator(


逻辑cpu=物理CPU个数×每颗核数
     1.物理cpu数：[XXXX@server ~]# grep ‘physical id’ /proc/cpuinfo|sort|uniq|wc -l

2.cpu核数：[XXXX@server ~]# grep ‘cpu cores’ /proc/cpuinfo|uniq|awk -F ‘:’ ‘{print $2}’

3.逻辑cpu：[XXXX@server ~]# cat /proc/cpuinfo| grep “processor”|wc -l


find ./ -name '*.log'  | xargs grep 'ResourceExhaustedError'

df -h
du -h --max-depth=1
ls -lt  文件按时间最新排序
ls -ltr 时间最新的再下面
tail  -f log 实时查看log文件内容  文件在更新
tail -f -n 100 catalina.out 实时查看后100行

vim 
第一行 :0 :1 g
最后一行 :$  G
跳转行   :n
显示文件名字 :f
vim撤销操作：u
vim恢复操作：ctrl+r
复制： yy  nyy
粘贴 p
删除 dd ndd
替换：
- :{作用范围}s/{目标}/{替换}/{替换标志}
- 例 :%s/foo/bar/g 会在全局范围(%)查找foo并替换为bar，所有出现都会被替换（g）

显示行号： set nu

rm update-model-1/model/ckpt_-[0-9]500*

iter_saver = tf.train.Saver(max_to_keep=3)  # keep 3 last iterations
best_saver = tf.train.Saver(max_to_keep=5)  # keep 5 last best models

with tf.Session() as sess:
    for epoch in range(20):
        # Train model [...]

        # and save a checkpoint 
        iter_saver.save(sess, "iter/model", global_step=epoch)

        if best_validiation_acc < last_validation_acc:
            best_saver.save(sess, "best/model")



model_file=tf.train.latest_checkpoint('ckpt/')
saver.restore(sess,model_file)



#vsc remotessh 
主机访问不了hbase，
两层ssh vscode还能用吗

1. 本地code 开发和调试远程机器上的代码

2. secureCRT 
 - linux和Mac文件互传，sftp，ALT+P 这个再细究
 - rz sz 两者只能文件传，

secureFX 可以文件夹传递
Scp为啥用不了

3. Vim python youcomlepteme


#Python 
中的 （None,） (None,None) 有时逗号还不能省略
继承类调用基类的函数


#ctr
CTR预估相关模型汇总，其中包括DeepFM、AFM、DIN、DIEN、FM、FFM、FNN、NFM、Wide&Deep;、PNN等模型。xdeepfmß
还有youtube的，，哪个是鼻祖呢
【5/5】Multi-Interest Network with Dynamic Routing for Recommendation at Tmall
【4/5】BERT4Rec- Sequential Recommendation with Bidirectional Encoder Representations from Transformer
【3/5】Behavior Sequence Transformer for E-commerce Recommendation in Alibaba


并行结构和串行结构两种深度CTR模型结构，目前常见的模型，比如Wide & Deep/DeepFM/DeepCross模型／AFM/FNN等模型基本都可以归到上述两种结构

CTR任务的应用：计算广告，推荐系统，信息流排序；

CTR特点：离散 高维 稀疏 大量

#FM 
FM主要目标是：解决数据稀疏的情况下，特征怎样组合的问题。
FM低阶特征组合，dense高阶特征组合
在做点击率预估时，我们的特征往往来自于用户(user)、广告(item)和上下文环境(context)，

   因子分解机FM算法可以处理如下三类问题：
回归问题(Regression)
二分类问题(Binary Classification)
排序(Ranking)

#deepFM

#xdeepFM

#fieldFM


#DIN
Sparse Features -> Embedding Vector -> MLPs -> Sigmoid -> Output.


#两次ssh到服务器zhangkl@10-9-24-174

secureCRT 123

ssh zhangkl@106.75.22.248 zkl123

zhangkailindeMacBook-Pro:~ zhangkailin$ ssh zhangkl@106.75.22.248
zhangkl@10-10-123-101:~$ ssh zhangkl@10.9.24.174 zkl123
zhangkl@10-9-24-174:~$

#服务器上多用户，python安装的库，只安装到本用户上，其他用户不影响
 pip3 install --user -r requirements.txt 

# Mac使用快捷键
网页刷新 command+r
control+command+q  锁屏
重命名 return 
文件删除 command +del
vscode 
 - fn +f12 定义跳转
 - fn+ control + -  定义返回

终端不同窗口切换  command + 1,2,3  

保存 复制 赞贴 command+ s c v
文件剪贴 ：你只需选中目标文件，然后使用Command+C复制，然后用Command +Option+V将其移动到目标目录。

ADID  ABID  区分banner  ABtest

排序 广告位 banner

uuid 百位膜5  5类
sku  

redis 轮播只减少 


模型3 4 轮询


轮询
- 目标 是  保证都样性， 就用户1 上线，推荐几个广告，他没有点击，，用户再次上线，就不推荐这几个了，
- 实现 缓存

模型改进
- 数据 增加 地域信息，和 年级信息，  重要的信息，但要注意，特征重要性是不一样的，地域信息加几层网络，作为一个子模块，再加入到大的框架中
- din 引入 transformer 自注意力，


#7.4-7.8 第一周  两个曲线
rpm 更适合softmax   ctr 差别很大，  rpm 差别不大

加入地域和年级， 可以考虑fieldFM  

单元测试，
图像的tag，颜色  离线表 在线表，HBase   三个数据源  三个表

在线每天更新，模型训练好了，今天数据 晚上测试，有结果，然后明天吧今天的数据 放入当做训练数据，  画线





In [46]: MAP = {"purchase":{"A1":-1,"Ae":2,"A3":3},\ 
    ...: "math_abie":{"A":1,"B":2,"C":4}}                                                                                                      

In [47]: MAP["purchase"].get('A1')                                                                                                             
Out[47]: -1

In [48]: MAP["purchase"].get('A1',0)                                                                                                           
Out[48]: -1


In [59]: ??dict.get                                                                                                                            
Docstring: D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.
Type:      method_descriptor

In [60]: MAP["purchase"].get('A4',3)                                                                                                           
Out[60]: 3


In [11]: datetime.datetime.today()+ datetime.timedelta()                        
Out[11]: datetime.datetime(2019, 7, 8, 17, 5, 29, 312297)

In [12]: datetime.datetime.today()+ datetime.timedelta(days=-1)                 
Out[12]: datetime.datetime(2019, 7, 7, 17, 5, 33, 894690)

In [13]: datetime.datetime.today()+ datetime.timedelta(hours=-1)                
Out[13]: datetime.datetime(2019, 7, 8, 16, 5, 43, 30622)
In [15]: (datetime.datetime.today()+ datetime.timedelta(hours=-1)).strftime("%Y-
    ...: %m-%d")                                                                
Out[15]: '2019-07-08'


电脑多个用户，
sudo pip3 install packagename

代表进行全局安装，安装后全局可用。如果是信任的安装包可用使用该命令进行安装。

pip3 install --user packagename

代表仅该用户的安装，安装后仅该用户可用。处于安全考虑，尽量使用该命令进行安装。

sudo -h pip install xx



{

"图片标签":
{ "风格":
  {"key":"label_3","value":{"插画":3,"干净":4,"科技":1,"萌系":2}},
  "色系":
  {"key":"label_1","value":{"红色系":2,"蓝色系":3,"白色系":7,"紫色系":5,"橙色系":1,"其他":8,"粉色系":6,"绿色系":4}},
  "主视觉":
  {"key":"label_2","value":{"礼物":4,"名师":2,"真人":1,"卡通":3}}
 },

"落地页标签":
{"落地页动效":
 {"key":"label_4","value":{"有":1,"无":2}},
 "落地页屏数":{"key":"label_5","value":{"6":6,"1":1,"4":4,"3":3,"10":10,"9":9,"5":5,"8":8,"2":2,"7":7,"11":11,"12":12}}
 },

 "广告标签":
 {"业务线":{"key":"label_6","value":{"小象编程":3,"香蕉学堂":7,"赫尔墨斯":4,"小象科学":5,"流量化运营":6,"直播课":1,"小象英语":2}},
 "主标题卖点":{"key":"label_7","value":{"节假日":6,"实体礼品":4,"游戏":7,"趣味测试":8,"电子礼品":3,"价格":1,"其他":9,"金币":5,"课程":2}}
 }
 }

#multihot 太常见了， 一直不理解
 mulithot是一个样本的 一个维度同时拥有多个类别
 还是多个特征，每个特征one-hot

sklearn.preprocessing.MultiLabelBinarizer
tf.nn.embedding_lookup_sparse



hbase  flask mahout

账号：
Kael 135 !zkl2205309
邮箱 zhangkl@knowbox.cn Zkl22
Wiki xconflunce zhangkl  zkl22
Mac: zkl123
secureCRT 123



asas = adimg.numpy()                                                   

In [72]: np.shape(adimg)                                                        
Out[72]: TensorShape([Dimension(20), Dimension(128)])

In [73]: np.shape(adimg)[1]                                                     
Out[73]: Dimension(128)

In [74]: np.shape(asas)[1]                                                      
Out[74]: 128


订单状态
订单号：	2019070993147 [发送/查看商家留言]
订单状态：	已确认    
付款状态：	已付款    
配送状态：	已发货    发货于 2019-07-09 21:39:53
激活码：	激活码序号 89D9K-W2CR3-K8YKK-4JJR9-QD84G 
商品列表放回购物车
商品名称	商品价格	购买数量	小计
一年订阅含赠送最长13个月 蓝灯专业版（LanternPro）激活码	￥233.00	1	￥233.00
商品总价: ￥233.00
费用总计
商品总价: ￥233.00
应付款金额: ￥233.00
联系人信息
收货人姓名：	zkl99999	邮箱地址：	zhangkailin@163.com
支付方式
所选支付方式：微信支付应付款金额：￥233.00



  "purchase_power": {"A1": 1, "A3": 2, "A2": 3, "A4": 4, "B": 5},
    "math_ability": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, },
    "english_ability": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, },
    "chinese_ability": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, },
    "activity_degree": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, },
    "app_freshness": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, }


对于array对象，*和np.multiply函数代表的是数量积，如果希望使用矩阵的乘法规则，则应该调用np.dot和np.matmul函数。
对于matrix对象，*直接代表了原生的矩阵乘法，而如果特殊情况下需要使用数量积，则应该使用np.multiply函数。

tf.pad

Tf.shape 
tf.tile 
tf.scan
tf.ones,shape,rank,zeros，fill
tf.batch_matmul -> tf.batch_matmul

Tf.squeeze()  去掉维度
 tf.expend_dims()

tf.sequence_mask

Tf.tensordot 
np.full

Np.tensordot
Np.pad
repeat和tile  复制扩充

from pyspark.sql.functions import lit
from sparkdl.image import imageIO

img_dir = "/PATH/TO/personalities/"

jobs_df = imageIO.readImagesWithCustomFn(img_dir + "/jobs",decode_f=imageIO.PIL_decode).withColumn("label", lit(1))
zuckerberg_df = imageIO.readImagesWithCustomFn(img_dir + "/zuckerberg", decode_f=imageIO.PIL_decode).withColumn("label", lit(0))



import numpy as np
arr = np.ones((2,2))
help(arr.dtype)


In [146]: import pandas as pd                                                                                

In [147]: inp = [{'c1':10,'c2':100},{'c1':11,'c2':120},{'c1':12,'c2':130}]                                   

In [148]: inp                                                                                                
Out[148]: [{'c1': 10, 'c2': 100}, {'c1': 11, 'c2': 120}, {'c1': 12, 'c2': 130}]

In [149]: type(inp)                                                                                          
Out[149]: list

In [150]: df = pd.DataFrame(inp)                                                                             

In [151]: df                                                                                                 
Out[151]: 
   c1   c2
0  10  100
1  11  120
2  12  130

In [152]: df2 = pd.DataFrame.from_dict(inp)                                                                  

In [153]: df2                                                                                                
Out[153]: 
   c1   c2
0  10  100
1  11  120
2  12  130

In [157]: for row in df.itertuples(): 
     ...:     print(getattr(row,'c1'),getattr(row,'c2')) 
     ...:                                                                                                    
10 100
11 120
12 130

In [159]: for row in df.itertuples(): 
     ...:     print(getattr(row,'c3',-1),getattr(row,'c2')) 
     ...:                                                                                                    
-1 100
-1 120
-1 130

In [185]: tf.one_hot([0,1,-1],3)                                                                             
Out[185]: 
<tf.Tensor: id=132, shape=(3, 3), dtype=float32, numpy=
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 0.]], dtype=float32)>


In [200]: list_1 = [[2,3],[2,5,7],[6,1,2,5]]                                                                       

In [201]: list_1                                                                                                   
Out[201]: [[2, 3], [2, 5, 7], [6, 1, 2, 5]]

In [205]: keras.preprocessing.sequence.pad_sequences(list_1,maxlen=4,padding='post',value=-1)                      
Out[205]: 
array([[ 2,  3, -1, -1],
       [ 2,  5,  7, -1],
       [ 6,  1,  2,  5]], dtype=int32)



#补零
current_words=[1,2,3,4]
current_words = list(current_words + [0] * (10 - len(current_words)))


#keras 文本分类

# 把词转换为编号，词的编号根据词频设定，频率越大，编号越小
sequences = tokenizer.texts_to_sequences(texts) 
# 把序列设定为1000的长度，超过1000的部分舍弃，不到1000则补0
sequences = pad_sequences(sequences, maxlen=1000, padding='post')  
sequences = np.array(sequences)



queue — A synchronized queue class：https://docs.python.org/3/library/queue.html

菜鸟教程 - Python3 多线程：http://www.runoob.com/python3/python3-multithreading.html

python3 队列：https://cloud.tencent.com/developer/information/python3%20%E9%98%9F%E5%88%97

Python 多进程 multiprocessing 使用示例：https://blog.csdn.net/freeking101/article/details/52511837

Python Queue模块详解：https://www.jb51.net/article/58004.htm


In [95]: datetime.timedelta(days=-1)                                                                               
Out[95]: datetime.timedelta(-1)

In [96]: datetime.datetime.today()                                                                                 
Out[96]: datetime.datetime(2019, 7, 12, 11, 58, 32, 377064)

IN： datetime.datetime.today()+ datetime.timedelta(-1)                                                         
Out[97]: datetime.datetime(2019, 7, 11, 11, 58, 54, 276410)

In [99]: (datetime.datetime.today()+ datetime.timedelta(-1)).strftime("%Y-%m-%d")                                  
Out[99]: '2019-07-11'


break语句的调用，起到跳出循环或者分支语句作用。
也就是说，break只有两种使用环境：
1 用于循环体内，包括for, while和do-while循环，作用为跳出break所在循环体。注意，如果是循环嵌套，而break出现在内层，那么只能跳出内层循环，无法跳出外层循环。
2 用于开关语句，即switch - case语句，起到跳出开关语句作用。用于switch嵌套时，与上述循环嵌套效果相同，只可以跳出所在开关语句。
从以上可以看出，break语句对if(判断语句)是没有效果的，所以不可能起到跳出if的作用，只会是跳出whille




In [122]: [1]                                                                                                      
Out[122]: [1]

In [123]: [1,]                                                                                                     
Out[123]: [1]


资源耗尽  一个批次 一个进程后天

ResourceExhaustedError (see above for traceback): /home/zhangkl/zhangkailin/Midas_Engine/update-model-1/model/ckpt_9000.data-00000-of-00001.tempstate3964584279969151581; No space left on device
         [[node save_34/SaveV2 (defined at update_model_2.cp.py:230)  = SaveV2[dtypes=[DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, ..., DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_FLOAT], _device="/job:localhost/replica:0/task:0/device:CPU:0"](_arg_save_34/Const_0_0, save_34/SaveV2/tensor_names, save_34/SaveV2/shape_and_slices, Metrics/beta1_power, Metrics/beta2_power, activity_embedding_var, activity_embedding_var/Adam, activity_embedding_var/Adam_1, adimg_embedding_var2, adimg_embedding_var2/Adam, adimg_embedding_var2/Adam_1, bn1/beta, bn1/beta/Adam, bn1/beta/Adam_1, bn1/gamma, bn1/gamma/Adam, bn1/gamma/Adam_1, bn1/moving_mean, bn1/moving_variance, chinese_embedding_var, chinese_embedding_var/Adam, chinese_embedding_var/Adam_1, city_embedding_var, city_embedding_var/Adam, city_embedding_var/Adam_1, dice_1/alphadice_1, dice_1/alphadice_1/Adam, dice_1/alphadice_1/Adam_1, dice_2/alphadice_2, dice_2/alphadice_2/Adam, dice_2/alphadice_2/Adam_1, english_embedding_var, english_embedding_var/Adam, english_embedding_var/Adam_1, f1/bias, f1/bias/Adam, f1/bias/Adam_1, f1/kernel, f1/kernel/Adam, f1/kernel/Adam_1, f1_attnull/bias, f1_attnull/bias/Adam, f1_attnull/bias/Adam_1, f1_attnull/kernel, f1_attnull/kernel/Adam, f1_attnull/kernel/Adam_1, f2/bias, f2/bias/Adam, f2/bias/Adam_1, f2/kernel, f2/kernel/Adam, f2/kernel/Adam_1, f2_attnull/bias, f2_attnull/bias/Adam, f2_attnull/bias/Adam_1, f2_attnull/kernel, f2_attnull/kernel/Adam, f2_attnull/kernel/Adam_1, f3/bias, f3/bias/Adam, f3/bias/Adam_1, f3/kernel, f3/kernel/Adam, f3/kernel/Adam_1, f3_attnull/bias, f3_attnull/bias/Adam, f3_attnull/bias/Adam_1, f3_attnull/kernel, f3_attnull/kernel/Adam, f3_attnull/kernel/Adam_1, freshness_embedding_var, freshness_embedding_var/Adam, freshness_embedding_var/Adam_1, grade_embedding_var, grade_embedding_var/Adam, grade_embedding_var/Adam_1, hour_embedding_var, hour_embedding_var/Adam, hour_embedding_var/Adam_1, math_embedding_var, math_embedding_var/Adam, math_embedding_var/Adam_1, mid_embedding_var, mid_embedding_var/Adam, mid_embedding_var/Adam_1, mobile_embedding_var, mobile_embedding_var/Adam, mobile_embedding_var/Adam_1, province_embedding_var, province_embedding_var/Adam, province_embedding_var/Adam_1, purchase_embedding_var, purchase_embedding_var/Adam, purchase_embedding_var/Adam_1, uid_embedding_var)]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.


python help


chrome 
ctrl+tab 切换页面
ctrl+shift+tab 反向切换


python 再后台运行的文件， 再去修改源代码，应该对之前一直运行的没有影响
因为python程序运行前，会自动编译python字节码，再内存里运行



双向LSTM 
for n in range(num_layers):
        cell_fw = cell_forw[n]
        cell_bw = cell_back[n]

        state_fw = cell_fw.zero_state(batch_size, tf.float32)
        state_bw = cell_bw.zero_state(batch_size, tf.float32)

        (output_fw, output_bw), last_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, output,
                                                                             initial_state_fw=state_fw,
                                                                             initial_state_bw=state_bw,
                                                                             scope='BLSTM_'+ str(n),
                                                                             dtype=tf.float32)

        output = tf.concat([output_fw, output_bw], axis=2)



pandas sample函数，采样函数
sample(frac=0.2, random_state=1024)   设置random_state 下次你在运行时，会拿同样的部分数据


pandas 读取大体积的csv


Flume：日志数据收集
Kafka： 实时日志数据处理队列
HDFS： 分布式存储数据
Spark SQL： 离线处理
Spark ML/MLlib：模型训练
Redis： 缓存(数据集非常大使用HBase)
flink 


tgt = []                                                                                                                                        

In [27]: tgt.append([1,0])                                                                                                                               

In [28]: tgt                                                                                                                                             
Out[28]: [[1, 0]]

In [29]: tgt.append([0,1])                                                                                                                               

In [30]: tgt                                                                                                                                             
Out[30]: [[1, 0], [0, 1]]
In [45]: tgt.append([1,0])                                                                                                                               

In [46]: tgt                                                                                                                                             
Out[46]: [[1, 0], [0, 1], [1, 0]]
In [52]: tgt                                                                                                                                             
Out[52]: [[1, 0], [0, 1], [1, 0]]

In [53]: tgt[:][1]                                                                                                                                       
Out[53]: [0, 1]

In [54]: tgt[:][0]                                                                                                                                       
Out[54]: [1, 0]

In [55]: tg = np.array(tgt)                                                                                                                              

In [56]: tg                                                                                                                                              
Out[56]: 
array([[1, 0],
       [0, 1],
       [1, 0]])

In [57]: tg[:][1]                                                                                                                                        
Out[57]: array([0, 1])

In [58]: tg[:][:]                                                                                                                                        
Out[58]: 
array([[1, 0],
       [0, 1],
       [1, 0]])

In [59]: tg[:,1]                                                                                                                                         
Out[59]: array([0, 1, 0])



df = pd.DataFrame(np.arange(12).reshape(3,4),
                          columns=['A', 'B', 'C', 'D'])


is_training = True
is_training = tf.cast(True, tf.bool)
不要添加shape参数
self.is_training_mode= tf.placeholder(tf.bool,name='is_training_mode')
尽量用 tf.float32，不用tf.float64

rand_array = np.random.rand(1024, 1024)
    print(sess.run(y, feed_dict={x: rand_array}))

两种，
tf.nn.dropout(keep4)     keep=0.5  keep =1.0 
dnn2 = tf.layers.dropout(dnn2, rate=0.5, training=self.is_training_mode)  false true


学习率指数衰减
步骤：1.首先使用较大学习率(目的：为快速得到一个比较优的解);
2.然后通过迭代逐步减小学习率(目的：为使模型在训练后期更加稳定);
decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)
0.1*0.99^(variable/100)    每decay_steps *0.99
staircase=False 阶梯 贴着指数 
staircase=True 指数

通常初始学习率，衰减系数，衰减速度的设定具有主观性(即经验设置)，而损失函数下降的速度与迭代结束之后损失的大小没有必然联系，

所以神经网络的效果不能单一的通过前几轮损失函数的下降速度来比较
只要sess.run(train_step)   global_step自动增加
train_step=tf.train.GradientDescentOptimizer(lr).minimize(loss,global_step=global_step)




python中标准错误（std.err）和标准输出(std.out)的输出规则（标准输出默认需要缓存后再输出到屏幕，而标准错误则直接打印到屏幕）
sys.stdout.write("stdout1")

sys.stderr.write("stderr1")

#tf高级操作api
tf.train.batch
tf线程协调器生成并读取 batch数据
tf.train.start_queue_runners

tensorflow中协调器 tf.train.Coordinator 和入队线程启动器 tf.train.start_queue_runners
tf.app.flags
tf.app.run


random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

# parse two
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("xx",xx,"xx")
FLAGS.xx

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
FLAGS = parser.parse_args(




joblib 多个模型


tf.train.Supervisor可以简化编程,避免显示地实现restore操作  
tf.train.exponential_decay( ) & tf.train.piecewise_constant( )  区间


python try except finally
异常的传递


基类也有 继承类也有 怎么调用，python中，继承类 调用基类的函数

继承类（子类） 有一函数继承 基类函数， 
子类调用函数，父类

父类
函数a
函数b 调用函数a

子类 函数a 复写 

子类实例化后，调用函数b，子类内复写的函数a 会被调用吗

如果父类构造函数中调用了已被子类重写的方法，则会进入子类重写的方法体内执行，如果该方法体中有引用子类的成员变量，由于子类成员还未初始化，所以会取其数据类型的默认值，

getattr(self,a)  self.a  类对象的属性


os.listdir 返回是无序的
os.walk 有文件夹和文件


a = np.array([2,4,6,8,10])
In [135]: np.where(a>4)                                                                                                               
Out[135]: (array([3, 4]),)

In [136]: np.where(a>4)[0]                                                                                                            
Out[136]: array([3, 4])
In [138]: a[np.where(a>4)]                                                                                                  
Out[138]: array([5, 5])

In [139]: a[np.where(a>4)].tolist()                                                                                                   
Out[139]: [5, 5]



builder = tf.saved_model.builder.SavedModelBuilder  

tensorflow:INFO:No assets to save.
 tensorflow:INFO:No assets to write.
 tensorflow:INFO:SavedModel written to: /home/zhangkl/zhangkailin/midas/Midas_Engine/update-model-1/model0.8/serving/17/saved_model.pb


区别
 from rnn import dynamic_rnn
#from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell import GRUCell



din_fcn_attention
din_attention  区别




In [152]: data = []                                                                                                                   

In [153]: da = {"a":1,"b":-1,"c":4}                                                                                                   

In [154]: da2 = {"a":1,"b":-1,"c":3}                                                                                                  

In [155]: da1 = {"a":0,"b":-1,"c":3}                                                                                                  

In [156]: data.append(da)                                                                                                             

In [157]: data.append(da1)                                                                                                            

In [158]: data.append(da2)                                                                                                            

In [159]:                                                                                                                             

In [159]:                                                                                                                             

In [159]: data                                                                                                                        
Out[159]: 
[{'a': 1, 'b': -1, 'c': 4},
 {'a': 0, 'b': -1, 'c': 3},
 {'a': 1, 'b': -1, 'c': 3}]

In [160]: item  =  pd.DataFrame.from_dict(data)                                                                                       

In [161]: item                                                                                                                        
Out[161]: 
   a  b  c
0  1 -1  4
1  0 -1  3
2  1 -1  3

In [162]: data2 =[]                                                                                                                   

In [163]: data2.append(da)                                                                                                            

In [164]: data2                                                                                                                       
Out[164]: [{'a': 1, 'b': -1, 'c': 4}]

In [165]: item2  =  pd.DataFrame.from_dict(data2)                                                                                     

In [166]: item2                                                                                                                       
Out[166]: 
   a  b  c
0  1 -1  4

In [167]:                                                                                                                             

python 
logging.info  logging.debug  怎么捕获bug信息啊，在异常那里， logging.debug吗，不是自动捕获的 

想知道错误什么时候出的，在哪里报错额

捕获异常那里 ，loggging.只能用debug error  不用info  ， info不打印

except Exception as e:
                    print(e)
                    logging.info("error in test ")
                    logging.debug("error in test {}".format(e))
                    logging.debug(sys.exc_info())

                    logging.error("cx",exc_info = True)
                    logging.except("xxx")



In [122]: from tensorflow.python.ops.rnn_cell import GRUCell                             
In [123]: GRUCell                                                                            
Out[123]: tensorflow.python.ops.rnn_cell_impl.GRUCell
In [124]: tf.nn.rnn_cell.GRUCell                                                              
Out[124]: tensorflow.python.ops.rnn_cell_impl.GRUCell