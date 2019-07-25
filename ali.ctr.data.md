This data set covers the shopping behavior in 22 days of all users in raw_sample(totally seven hundred million records)
https://tianchi.aliyun.com/dataset/dataDetail?dataId=56

特征主要 连续型特征和离散型特征
离散型 onehot
连续型 标准化


"raw_sample.csv"
     user  time_stamp  adgroup_id          pid  nonclk  clk
0  581738  1494137644           1  430548_1007       1    0
1  449818  1494638778           3  430548_1007       1    0
2  914836  1494650879           4  430548_1007       1    0
3  914836  1494651029           5  430548_1007       1    0
4  399907  1494302958           8  430548_1007       1    0
pid 资源位 scenario
adgroup_id：脱敏过的广告单元ID；
noclk：为1代表没有点击；为0代表点击；
clk：为0代表没有点击；为1代表点击
 time_stamp: time stamp(Bigint, 1494032110 stands for 2017-05-06 08:55:10)
我们用前面7天的做训练样本（20170506-20170512），用第8天的做测试样本（20170513）


 pd.read_csv("behavior_log.csv",skiprows=9,nrows=15) 
      user  time_stamp btag   cate   brand
0  558157  1493741625   pv   6250   91286
1  558157  1493741626   pv   6250   91286
2  558157  1493741627   pv   6250   91286
3  728690  1493776998   pv  11800   62353
4  332634  1493809895   pv   1101  365477

btag：行为类型，包括 ipv 浏览，cart加入购物车，fav喜欢，buy
cate：脱敏过的商品类目
brand：品牌词
为啥没有adgroup_id呢，


'ad_feature.csv'
   adgroup_id  cate_id  campaign_id  customer     brand   price
0       63133     6406        83237         1   95471.0  170.00
1      313401     6406        83237         1   87331.0  199.00
2      248909      392        83237         1   32233.0   38.00
3      208458      392        83237         1  174374.0  139.00
4      110847     7211       135256         2  145952.0   32.99
campaign_id：脱敏过的广告计划ID
customer_id:脱敏过的广告主ID
其中一个广告ID对应一个商品item（宝贝），一个宝贝属于一个类目，一个宝贝属于一个品牌。
brand 有null



"user_profile.csv"
     userid cms_segid cms_group_id final_gender_code  age_level  pvalue_level  shopping_level  occupation new_user_class_level 
0     234          0             5                  2          5           NaN               3           0       3.0
1     523          5             2                  2          2           1.0               3           1       2.0
2     612          0             8                  1          2           2.0               3           0       NaN
3    1670          0             4                  2          4           NaN               1           0       NaN
4    2545          0            10                  1          4           NaN               3           0       NaN

age_level 年龄层次
pvalue_level 消费档次，1 抵挡，2 中档，3 高档  有nan
shopping_level 购物深度 1 浅层 2 中度 3 深度用户
occupation 大学生  1 是 0 否
user_class 城市层级 有nan





knowbox
{"signature_name":"serving","inputs":{"uid_ph":[46793507,46793507,46793507,46793507],"chinese_ph":[1,1,1,1],"math_ph":[3,3,3,3],"hour_ph":[18,18,18,18],"mobile_ph":[1,1,1,1],"activity_ph":[2,2,2,2],"mid_his_ph":[[11640,10818,10818,10628,10575,10545,10458],[11640,10818,10818,10628,10575,10545,10458],[11640,10818,10818,10628,10575,10545,10458],[11640,10818,10818,10628,10575,10545,10458]],"mid_ph":[13687,13559,13703,13728],"mask_ph":[[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1]],"seq_len_ph":[7,7,7,7],"purchase_ph":[5,5,5,5],"freshness_ph":[7,7,7,7],"city_ph":[3457,3457,3457,3457],"english_ph":[0,0,0,0],"province_ph":[13,13,13,13],"grade_ph":[2,2,2,2]}}


(b'10000022_101475_2019-06-19 18:42:27', 
{b'context:count_click': b'0', b'context:log_time': b'2019-06-19 18:42:27', 
b'context:log_day': b'2019/06/19', b'ad:exposure_duration': b'5', b'ad:test_timestamp': 
b'1560911463778', b'user:school_id': b'54085', b'user:activity_degree': b'E',
 b'context:week_accuracy': b'0.0', b'context:exposure_duration': b'0', b'context:exe_time': 
 b'2019-06-19 18:42:40', b'ad:label_6': b'-1', b'context:hexposure_alocation': b'1',
  b'ad:count_click': b'350', b'context:dexposure_alocation': b'1', b'ad:location_ad': b'6', 
  b'context:is_click': b'0', b'user:county_id': b'1558', b'context:hclick_similarad': b'0', 
  b'context:log_month': b'2019/06/01', b'ad:ad_id': b'101475', b'ad:label_3': b'4', 
  b'context:log_week': b'2019/06/17', b'context:window_otherad': b'0', b'user:grade_id': b'4', 
  b'user:english_ability': b'0', b'context:yeaterday_accuracy': b'0.0', b'ad:label_4': b'-1',
   b'ad:alldexposure_clocation': b'407573', b'user:mobile_os': b'1', b'context:location_ad': b'6', 
   b'context:hexposure_similarad': b'9', b'user:chinese_ability': b'0', b'ad:allhexposure_alocation': 
   b'63035', b'ad:label_5': b'-1', b'ad:label_2': b'3', b'context:hexposure_clocation': b'1',
    b'ad:alldexposure_alocation': b'4324', b'context:dclick_otherad': b'0', b'user:user_id': b'10000022', 
    b'user:purchase_power': b'B', b'user:app_freshness': b'G', b'user:math_ability': b'E', 
    b'context:log_hourtime': b'18', b'user:app_type': b'3', b'ad:label_1': b'1',
     b'context:log_weektime': b'4', b'context:rclick_ad': b'[]', b'user:province_id': b'13', 
     b'ad:window_otherad': b'0', b'user:mobile_type': b'OPPO_R11s;7.1.1', b'user:test_timestamp':
      b'1560940904729', b'context:duplicate_tag': b'0', b'ad:label_7': b'-1', b'context:rclick_category': 
      b'[]', b'context:month_accuracy': b'0.0', b'user:city_id': b'181', b'ad:allhexposure_clocation': 
      b'9418410', b'context:dexposure_clocation': b'1'})