# -*- coding: utf-8 -*-

#验证神经网络
import tensorflow as tf
from vgg_tf import vgg_16
import vgg_train
import numpy as np
import cv2
import os

BATCH_SIZE = 1
IMAGE_SIZE = 64
NUM_CHANNELS = 3
OUTPUT_NODE = 5

def evalute(imput_image):
    
    #输入重塑
    imput_image =  cv2.resize(imput_image, (IMAGE_SIZE, IMAGE_SIZE))
    shape=(1, IMAGE_SIZE, IMAGE_SIZE, 3)
    images_array = np.zeros(shape,dtype='uint8')
    images_array[0] = imput_image
    
    with tf.Graph().as_default() as g:
        #定义输入输出格式
        x = tf.compat.v1.placeholder(tf.float32,[
            BATCH_SIZE, #batch大小
            IMAGE_SIZE, #图片大小
            IMAGE_SIZE,
            NUM_CHANNELS], #图片深度（灰度为1）
            name='x-input')
    
        #计算传播结果 测试无需正则 无需dropout
        y = vgg_16(x, 5, None) #神经网络输出
 
        softmax_ys = tf.nn.softmax(y) #经过softmax输出概率值
        saver = tf.compat.v1.train.Saver()
        
        with tf.compat.v1.Session() as sess:
             #函数直接找到目录中的最新模型
             ckpt = tf.train.get_checkpoint_state(
                 vgg_train.MODEL_SAVE_PATH)
             
             if ckpt and ckpt.model_checkpoint_path:
                 #加载模型
                 saver.restore(sess, ckpt.model_checkpoint_path)
                 
                 y = sess.run(y,feed_dict={x: images_array})
                 
                 softmax_y = sess.run(softmax_ys,feed_dict={x: images_array})
                 
                 result = np.argmax(softmax_y[0])
                 
                 return result #最后结果 返回第几个类别
                 
def evalute_image(path):
    """测试图片"""
    for root,dirs,files  in os.walk(path) : 
        for file in files:
            all_path = os.path.join(root, file)
            image = cv2.imread(all_path, 1)
            result = evalute(image) #验证
            print(result) #验证结果
            print(file) #图片名称(以类别命名)
            
#主程序                                                     
def main(argv=None):
    path = r"face_image\test_data"
    evalute_image(path)

if __name__=='__main__':
    tf.compat.v1.app.run()        
