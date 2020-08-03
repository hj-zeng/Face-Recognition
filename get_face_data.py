# -*- coding: utf-8 -*-

import os
import cv2
from datetime import datetime
import csv
import numpy as np

# ------------------------------------------获取人脸 划分数据集

#从csv读取图片标注信息
def get_img_loc(class_msg):
    """获取图片标注信息"""
    msgs = []
    csv_file = 'loose_bb_test.csv'
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if class_msg in row[0]:
                row[0] = row[0] + '.jpg'
                msgs.append(row)
                
    return msgs
                
# msgs = get_img_loc('n000078')
# print(msgs)
    
def is_exists(path):
    """判断存在路径 不存在则创建"""
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path) 
        
def get_face_img(msgs, root_path, rate):
    """截取人脸"""
    #输入参数:人脸位置标注， 原始图片根目录，训练集划分比列
    for msg in msgs:
        read_path = root_path + msg[0]
        image = cv2.imread(read_path,1)
        x = int(msg[1])
        y = int(msg[2])
        w = int(msg[3])
        h = int(msg[4])
        face_image = image[y:y+h, x:x+w]

        #划分
        number = np.random.randint(rate)
        if number<1:
            save_path = 'face_image/val_data/'+msg[0]
        else:
            save_path = 'face_image/train_data/'+msg[0]
        try:
            cv2.imwrite(save_path, face_image)
        except:
            print(save_path) #错误图片
 
def get_face():
    """获取人脸信息，截取人脸"""
    for root,dirs,files  in os.walk('origin_face_data') : 
     all_class = dirs #['n000078', 'n000178', 'n000410', 'n000527', 'n000596']
     break 
    
    for class_ in all_class:
        
        is_exists('face_image/train_data/' + class_)
        is_exists('face_image/val_data/' + class_)
        
        msgs = get_img_loc(class_)
        strat_time = datetime.now() 
        get_face_img(msgs,'origin_face_data/',4) #4比1的比例
        end_time = datetime.now() 
        use_time = end_time - strat_time
        print('所用时间' + str(use_time))


# get_face()

# ------------------------------------------生成图片地址文件 图片标签
def get_image_path(path, save_txt):
    """图片地址生成txt文件"""
    all_root_paths = []
    once = True
    for root,dirs,files  in os.walk(path):
        if once:
            once=False #第一个为根目录 不添加
        else:
            all_root_paths.append(root)
            
    for root,dirs,files  in os.walk(path) : 
     all_class = dirs #['n000078', 'n000178', 'n000410', 'n000527', 'n000596']
     break 
            
    for i in range(len(all_root_paths)):
        for root,dirs,files  in os.walk(all_root_paths[i]):
            for file in files:
                image_path = os.path.join(all_class[i], file)
                #写入文件
                with open(save_txt,'a') as obj:
                    obj.write(image_path+'\n')
        
        
path = r"face_image\train_data"    
get_image_path(path, r'txt\train_image.txt')
test_path = r"face_image\val_data"    
get_image_path(test_path, r'txt\val_image.txt')


