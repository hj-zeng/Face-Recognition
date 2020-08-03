# -*- coding: utf-8 -*-

import tensorflow as tf

def conv(input_, input_deep, output_deep,ksize, stride,name):
    """卷积层函数"""
    #输入参数->输入，输入维度，输出维度，卷积核大小，移动步长，变量名
    #输出参数->卷积层输出
    with tf.compat.v1.variable_scope(name):
        conv_weights = tf.compat.v1.get_variable(
            'weight', [ksize, ksize, input_deep, output_deep],
            initializer = tf.truncated_normal_initializer(stddev=0.1))
        
        conv_biases = tf.compat.v1.get_variable(
            'biases', [output_deep], initializer = tf.constant_initializer(0.0))
        
        conv = tf.nn.conv2d(
            input_, conv_weights, strides=[1,stride,stride,1], padding='SAME')
        
        tanh = tf.nn.tanh(tf.nn.bias_add(conv, conv_biases))
        
    return tanh

def maxpool(input_, ksize, stride,name):
    """最大池化层函数"""
    #输入参数->输入，池化核大小，移动步长，变量名
    #输出参数->池化层输出
    with tf.name_scope(name):
        pool = tf.nn.max_pool2d(input_, ksize=[1,ksize,ksize,1], strides=[1,stride,stride,1],
                                padding='VALID')
    return pool

def fc(input_, input_deep, out_deep, name,regularizer):
    """全连接函数"""
    #输入参数->输入，输入维度，输出维度，变量名，正则化
    #输出参数->全连接层输出
    with tf.compat.v1.variable_scope(name):
        weight = tf.Variable(
            tf.truncated_normal([input_deep, out_deep], stddev=0.05),
            name="weights")
        
        bias = tf.Variable(
            tf.constant(0.1, dtype=tf.float32, shape=[out_deep]), 
            name="bias")

        if regularizer != None:
            tf.compat.v1.add_to_collection('losses', regularizer(weight))
            
        net = tf.add(tf.matmul(input_, weight), bias)     
        
    return net
  

def vgg_16(input_, class_number, regularizer):
    """vgg16网络结构函数"""
    #输入函数->网络输入，输出维度，正则化
    #输出函数->网络输出
    conv1 = conv(input_, 3, 64, 3, 1,'layer1_conv1')
    
    conv2 = conv(conv1, 64, 64, 3, 1,'layer2_conv2')
    
    pool1 = maxpool(conv2, 3, 2,'layer3_pool1')
    
    conv3 = conv(pool1, 64, 128, 3, 1,'layer4_conv3')
    
    conv4 = conv(conv3, 128, 128, 3, 1,'layer5_conv4')
    
    pool2 = maxpool(conv4, 3, 2,'layer6_pool2')

    conv5 = conv(pool2, 128, 256, 3, 1,'layer7_conv5')

    conv6 = conv(conv5, 256, 256, 3, 1,'layer8_conv6')

    conv7 = conv(conv6, 256, 256, 1, 1,'layer9_conv7')

    pool3 = maxpool(conv7, 3, 2,'layer10_pool3')
    
    conv8 = conv(pool3, 256, 512, 3, 1,'layer11_conv8')

    conv9 = conv(conv8, 512, 512, 3, 1,'layer12_conv9')

    conv10 = conv(conv9, 512, 512, 1, 1,'layer13_conv10')

    pool4 = maxpool(conv10, 3, 2,'layer14_pool4')
    
    conv11 = conv(pool4, 512, 512, 3, 1,'layer15_conv11')

    conv12 = conv(conv11, 512, 512, 3, 1,'layer16_conv12')

    conv13 = conv(conv12, 512, 512, 1, 1,'layer17_conv13')

    pool5 = maxpool(conv13, 3, 2,'layer18_pool5')
    

    pool_shape = pool5.get_shape().as_list()
    #计算向量长度 长x宽x高(第二、三、四维度，第一维为batch个数)
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    #通过函数变成向量
    reshaped = tf.reshape(pool5, [pool_shape[0], nodes])
    
    fc1 = fc(reshaped, nodes, 4096, 'layer19_fc1', regularizer)
    
    fc2 = fc(fc1, 4096, 4096, 'layer20_fc2', regularizer)
    
    fc3 = fc(fc2, 4096, class_number, 'layer21_fc3' ,regularizer)
    
    return fc3


# import cv2
# import numpy as np
# path = r"face_image\train_data\n000078\0004_01.jpg"
# tf.compat.v1.reset_default_graph() #先清空计算图
# image = cv2.imread(path, 1)
# image_data = tf.image.convert_image_dtype(image,dtype=tf.float32)
# with tf.compat.v1.Session() as sess:
#     image = sess.run(image_data)
#     image = cv2.resize(image,(227,227))
#     reshape_xs=np.reshape(image,(1,227,227,3))
#     y = vgg_16(reshape_xs, 5, None,)
#     print(y) #Tensor("layer21_fc3/Add:0", shape=(1, 5), dtype=float32)
