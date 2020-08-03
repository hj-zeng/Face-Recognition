# 人脸识别

## VGGFACE2数据集

> VGGFace2是一个大规模的人脸识别数据集。图片是从Google图片搜索中下载的，并且在姿势，年龄，照度，种族和职业方面都有很大差异。整个数据集分为训练（8631身份）和测试（500身份）集。http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/meta_infor.html

## 数据集预处理

选择5个人作为训练数据集，对这5个人图片进行：

1. 截取人脸：利用所给的标注信息进行截取

2. 标签生成：遵循one-hot编码，按照顺序生成以下格式

   >  [1,0,0,0,0]  [0,1,0,0,0]  [0,0,1,0,0]  [0,0,0,1,0]  [0,0,0,0,1] 

## 神经网络搭建

1. 参考网络：vgg-16

2. 网络参数：
   * 激活函数：tanh
   * 输入层维度：64×64×3
   * 损失函数：交叉熵
   * 正则化权重：le-5
   * 优化算法：MBGD，小批量梯度下降
   * batch大小：10

## 训练过程损失值的变化

![损失值](https://github.com/hj-zeng/face-recognition/blob/master/%E6%8D%9F%E5%A4%B1%E5%80%BC.jpg)

## 测试结果

测试图片中共有十张图片（从训练集分离出来，没有经过训练），每个人共两张图片，将图片以数字到0到4命名。如，第一个人的图片命名为0_0.jpg 和 0_1.jpg 。对应神经网络的输出下标。在这8张测试图片中，网络可以得到100%的识别。

## 文件说明

* [origin_face_data](https://github.com/hj-zeng/face-recognition/tree/master/origin_face_data): 原始数据集部分图片
* [face_image](https://github.com/hj-zeng/face-recognition/tree/master/face_image)   :数据集获取的人脸图片
*  [txt](https://github.com/hj-zeng/face-recognition/tree/master/txt) : 训练和验证人脸图片地址
* [loose_bb_test.csv](https://github.com/hj-zeng/face-recognition/blob/master/loose_bb_test.csv) : 人脸位置标注文件
* py文件：
  * [get_face_data.py](https://github.com/hj-zeng/face-recognition/blob/master/get_face_data.py) : 从原始图片截取人脸图片程序
  * [get_batch_data.py](https://github.com/hj-zeng/face-recognition/blob/master/get_batch_data.py) : 获取batch个人脸数据用于训练程序
  * [vgg_tf.py](https://github.com/hj-zeng/face-recognition/blob/master/vgg_tf.py) : vgg网络搭建程序
  * [vgg_train.py](https://github.com/hj-zeng/face-recognition/blob/master/vgg_train.py) : vgg网络训练程序
  * [vgg_evalute.py](https://github.com/hj-zeng/face-recognition/blob/master/vgg_evalute.py) : 测试验证训练好的模型程序
