# -*- coding: utf-8 -*-

#训练vgg网络
import os 
import tensorflow as tf
from get_batch_data import get_image_arrary,get_batch
from vgg_tf import vgg_16
from datetime import datetime
import matplotlib.pyplot as plt
#中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#参数设置
BATCH_SIZE = 20 #batch大小 
LEARNINNG_RATE_BASE = 0.001 #基础学习率
LEARNING_BATE_DECAY = 0.98 #学习率衰减率
REGULARATION_RATE = 0.00001 #正则化权重
TRANING_STEPS = 10001 #训练次数
IMAGE_SIZE = 784 #输入图片大小
NUM_CHANNELS = 3 #输入图片维度 
OUTPUT_NODE = 5 #神经网络输出维度=标签维度

#地址设置
MODEL_SAVE_PATH = "model" #模型保存路径
MODEL_NMAE = 'model.ckpt'  #模型名字

def train():
    """训练模型"""
    #获取图片和标签列表
    images_array, labels_array = get_image_arrary( #以数组的形式加载数据
        IMAGE_SIZE,
        IMAGE_SIZE,
        NUM_CHANNELS,
        True)
    
    images_val, labels_val = get_image_arrary(
        IMAGE_SIZE,
        IMAGE_SIZE,
        NUM_CHANNELS,
        False)

    with tf.name_scope('input'):
        #预定义输入x和标签y_
        x = tf.compat.v1.placeholder(tf.float32,[
            BATCH_SIZE, #batch大小
            IMAGE_SIZE, #图片大小
            IMAGE_SIZE,
            NUM_CHANNELS], 
            name='x-input') 
        
        y_ = tf.compat.v1.placeholder(tf.float32,
                                      [None, OUTPUT_NODE],
                                      name='y-input')
        
    regularizer = tf.contrib.layers.l2_regularizer(REGULARATION_RATE)

    y = vgg_16(x, 5, regularizer)  #   前向传播结果 
    
    global_step = tf.Variable(0, trainable=False) #训练次数 属于非优化对象
       
    #生成损失函数
    #利用函数生成交叉熵 
    with tf.name_scope('loss_function'): #命名管理
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=tf.argmax(y_,1)) #argmax对最大下标
        # 计算交叉熵平均值
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.compat.v1.get_collection('losses')) 
        
    
    #指数衰减法设置学习率
    learning_rate = tf.compat.v1.train.exponential_decay(
        LEARNINNG_RATE_BASE,
        global_step,
        int(len(images_array)/BATCH_SIZE),
        LEARNING_BATE_DECAY
        )

    
    #优化损失函数（反向优化算法）
    with tf.name_scope('train_step'):
        train_step = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)\
            .minimize(loss, global_step=global_step)
            
        #反向传播更新参数 
        with tf.control_dependencies([train_step]):
            train_op = tf.no_op(name='train')
       
    #计算正确率 比较输出结果和标签
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1)) 
        #将布尔值转为实行再计算平均值 即正确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


    #初保存器初始化
    saver = tf.compat.v1.train.Saver()
    steps = []
    accracys = []
    loss_values = []
    
    with tf.compat.v1.Session() as sess:
        
        tf.compat.v1.global_variables_initializer().run() #参数初始化
            
        for i in range(TRANING_STEPS): #开始训练
            #获取batch个数据
            xs, ys = get_batch(
                BATCH_SIZE, 
                images_array, 
                labels_array, 
                IMAGE_SIZE,
                IMAGE_SIZE,
                NUM_CHANNELS
                )
            
            _ , losses_value, step = sess.run(
                [train_op, loss, global_step],feed_dict ={x: xs, y_: ys})
            
            #获取batch个验证数据
            valxs, valys = get_batch(
                BATCH_SIZE, 
                images_val,
                labels_val, 
                IMAGE_SIZE,
                IMAGE_SIZE,
                NUM_CHANNELS
                )
            
            #正确率
            validate_acc = sess.run(accuracy,feed_dict ={x: valxs, y_: valys})

            steps.append(i)
            accracys.append(validate_acc)
            loss_values.append(losses_value)
            
            #打印训练过程的参数变化
            if i % 1000 ==0:
                print("训练 %d 轮后的损失值为 %g" %(step, losses_value))
                #验证
                print("训练 %d 轮后的正确率为 %g" %(i,validate_acc))
                #保存模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NMAE), 
                            global_step=global_step)
    
    draw_train_process(steps,loss_values, accracys)        
   
#画图函数
def draw_train_process(steps,para1, para2, name):
    """训练过程中的损失值/正确率变化"""
    title="训练过程中损失值变化"
    plt.title(title, fontsize=24)
    plt.xlabel("训练次数", fontsize=14)
    # plt.ylabel("损失值", fontsize=14)
    plt.plot(steps, para1,color='red',label='损失值') 
    plt.plot(steps, para2,color='blue',label='正确率') 
    plt.savefig(name +'损失值.jpg')
    plt.legend('损失值', '正确率')
    plt.grid()
    plt.show()      

#主程序                                                     
def main(argv=None):
    tf.compat.v1.reset_default_graph() #先清空计算图
    strat_time = datetime.now()
    train()
    end_time = datetime.now() 
    use_time = end_time - strat_time
    print('训练所用时间' + str(use_time))

if __name__=='__main__':
    tf.compat.v1.app.run()  
