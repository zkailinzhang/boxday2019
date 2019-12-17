
作业代码 
/data/lishuang/model_update_script     gbdt+lr
/data/lishuang/model_update/model     模型，三部分 dis dis  mdl ohe dis
dis-2019-06-30.dis
model-2019-06-30.mdl   已经更新到最近一天
ohe-2019-06-30.dis   
In [4]: import xlearn                                                                                                                                                                

In [5]: import lightgbm                                                                                                                                                              

In [6]: import xgboost                                                                                                                                                               

In [7]: import DBUtils  

gbdt+lr  中间 是onehot   
我记得中间是嵌入层啊

普通代码是onehot 直接喂入 lr    组合  应该是样本特征组合   
每个样本 维度10 ，放入每个
深度代码是嵌入层 再喂入lr 吗



特征预处理 

    连续值特征，int 或float  可以直接喂入，
    类别特征， 需要嵌入， 
    - 类别id本表示类别，不能看出任何意义，，那我定义的类别是连续的  还是不连续的，，看cv 多目标识别，是连续的 1 猫 2 狗 3 马 
    - 还有本身开始，类别就变成的离散不连续的数字，，是不是要映射成 连续id   参考 像midas点击中  各种map_int
    - 还有本身开始，类别就变成的离散不连续的数字，，直接嵌入 不好是吧

    序列特征，rnn


类别特征的最大值 ，嵌入空间用    
所有类别特征都重新跑一边，
mysql feature.py  
homework 的 chapter section     这里是找所有班级的 所有的 chapter，故吧sql 中classid剔除了   
sql_sec = """select base_sections from daily_homework_info where `day`='{}' order by homework_id """
DISTINCT关键词   
用distinct关键字， 如果结果中有完全相同的行，就去除重复行
SELECT DISTINCT province_id FROM r_teacher_info

SELECT DISTINCT base_sections FROM daily_homework_info

"""SELECT chapters,sections,style FROM chapter_homework_set
            WHERE chapter_id={} ORDER BY rank ASC""".

unique().tolist()


        info3 = f.get_all("""SELECT section_id,parent_id FROM base_section_info WHERE level=3""")
        info2 = f.get_all("""SELECT section_id,section_order FROM base_section_info WHERE level=2""")


reflect 报错，
indices[11,1] = 364117 is not in [0, 100000)


"""SELECT chapters,sections,style FROM chapter_homework_set
            WHERE chapter_id={} ORDER BY rank ASC"""


这个作业推荐的项目，所有的表名和 每个表的列名


history homework 现在是3天   7-14天
b*7*256   mask b*7 全1  注意 现在设的全1 要改，  已经提到，不是每天都有作业，故 不是全1了    
当天布置的 b*256  
喂入din ok



评价指标


top1 accuration  
top3 acc  
top5
top7

参考 /Users/zhangkailin/zklcode/turing/train_test/model2010.5/main.py   
line192  evaa = [1, 3, 5, 10, 15]

吧auc加上吧，万一要呢  recall  precision

有些疑问， 
测试集的时候  
  auc recall  都是在整个第二天的数据上 计算的   
    也不是完整的，也是分批次的， 然后汇总，一天的数据  在计算all_auc, r, p, f1
  那top1 top3 top5呢
  想不明白   在物体识别的时候，10个类别吧，若提供一张猫的图，会给10个概率，top1 top3  top5  这好理解，
  代码应该 tf.topk  tf.nn.topk
  这个场景呢，怎么用，参考gbdt+lr，，是不是 仍然在测试阶段，一批数据过来，每个样本， 。。。。
  但这边是二分类啊，    那看b站的首页，8个，  top放在候选上
  训练的时候是不是 候选集是不是 拆分成单个的了，
  测试的时候，一起喂入，8个list  每个[0.9,0.1]  
  还是不能完全理解啊 

  看gbdt吧



网路结构疑问
madis  ctr  是不是这样的构造的   喂入的有  
现在点击广告的样本和特征 ，以及没有点击的广告样本和特征
历史点击ad的样本的特征，用户画像，上下文特征 然后输出(1,0) (0,1)

其中，现在点击广告的样本和特征  是否喂入了，还是训练要喂入的 ，，测试 就随机喂入个 样本 看预测，是这样的流程吧

同样的道理，猫狗是吧，也不是全是正样本，单分类的 肯定有正负样本吧，多分类的 有other，
同样的道理，word2vec，也是要正负样本的，误差函数就很明显啊，
先是层级softmax 再 负样本采样，再nce





老师的风格也会变啊  特征空间 放在一起学 

作业推荐
一个老师给这个班级 每天推荐作业一个，不一定每天都有作业，
一个就是不会每天都布置，还有个 这个老师不是每天都上课
作业看做是  chapter section   

#老师特征
style  老师的风格，喜欢跟着大纲走，还是继续探索
core 老师的等级 金牌讲师
老师的科目，？


#班级特征
cap_avg_ph
cap_max_ph  
cap_min_ph  班级能力等级  
edition id ？

#待推荐作业
today_chap_mask_p  学生掌握程度 list
today_sec_mask_p


#班级历史作业
昨天 前天  大前天

#context
学习能力打分，
隔的天数
作业提交率


prefer_assign_time_var_ph  偏好 布置时间
prefer_assign_rank_avg_ph  布置难度

reflect_value_ph  


作业推荐的 

样本 不点击的负样本   点击的正样本，很少 很珍贵

老师分 核心老师 和不核心老师， 
金牌老师有两个 需要重点处理，
一个 是他 点击的  加重权重处理，一个 是 他 不点击的 也加重权重处理，
在损失函数那里，

　6. Subsampling Training Data
　　　　
1）在实际中，CTR远小于50%，所以正样本更加有价值。通过对训练数据集进行subsampling，可以大大减小训练数据集的大小
2）正样本全部采（至少有一个广告被点击的query数据），负样本使用一个比例r采样（完全没有广告被点击的query数据）。但是直接在这种采样上进行训练，会导致比较大的biased prediction
3）解决办法：训练的时候，对样本再乘一个权重。权重直接乘到loss上面，从而梯度也会乘以这个权重


先采样减少负样本数目，在训练的时候再用权重弥补负样本，非常不错的想法


  coe = tf.constant([0.2,0.2])
  coe_mask = tf.equal(self.core_type_ph,1)
  coe_mask2 = tf.concat([tf.expand_dims(coe_mask,-1) for i in range(2) ],-1)
  self.target_ph_coe =  tf.where(coe_mask2,self.target_ph*coe,self.target_ph)

  ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph_coe)