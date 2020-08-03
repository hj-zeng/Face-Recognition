# -*- coding: utf-8 -*-

import cv2
import numpy as np
import  os


def get_image_arrary(image_h, image_w, image_c, is_trian):
    """输入参数：图片的三个维度（神经网络输入维度）"""
    #初始化储存列表
    images = []
    
    if is_trian:
        root_path = r"face_image\train_data"#图片根目录
        images_txt = r'txt\train_image.txt' #图片地址文件

    else:
        root_path = r"face_image\val_data" #图片根目录
        images_txt = r'txt\val_image.txt' #图片地址文件


    with open(images_txt) as obj:
        for l in obj.readlines():
            images.append(l.strip('\n'))
    
    sum_number = len(images)
    
    for root,dirs,files  in os.walk(root_path) : 
     all_class = dirs #['n000078', 'n000178', 'n000410', 'n000527', 'n000596']
     break 
 
    #创建数组
    shape=(sum_number, image_h, image_w, image_c)
    images_array = np.zeros(shape,dtype='uint8')
    shape2 = (sum_number, 5)
    labels_array = np.zeros(shape2)
    for i in range(sum_number):
        #图片
        path = images[i]
        all_path = os.path.join(root_path, path)
        image = cv2.imread(all_path, 1)
        image = cv2.resize(image, (image_h, image_w))
        images_array[i] = image
        #标签
        label_array = np.zeros((5), dtype='uint8')
        for j in range(len(all_class)):
            if all_class[j] in path:
                label_array[j] = 1

        labels_array[i] = label_array

    return images_array, labels_array


def get_batch(btch_size, images_array, labels_array, image_h, image_w, image_c):
    """得到一部分图片和标签"""
    #输入参数 一部分(batch)的大小，一部分的图片，一部分的标签，图片维度(h,w,c)
    lim = len(images_array)
    shape = (btch_size,image_h, image_w, image_c)
    shpae1 = (btch_size,5)
    batch_img_array = np.zeros(shape,dtype='uint8') #类型保持一致
    batch_lab_array = np.zeros(shpae1,dtype='uint8')
    randoms = [] #避免重复
    for i in range(btch_size):
        flag = True
        while flag:
            random = np.random.randint(lim)
            if random not in randoms:
                flag = False
        batch_img_array[i] = images_array[random]
        batch_lab_array[i] = labels_array[random]
        randoms.append(random)
    batch_lab_array = batch_lab_array.astype(np.float32) #类型转换
    
    return batch_img_array, batch_lab_array

#测试程序-测试函数是否正确(引用的时候应该注释掉)
# images_array, labels_array = get_image_arrary(227, 227, 3, None)
# print(len(images_array))
# batch_img_array, batch_lab_array = get_batch(10, images_array, labels_array, 227, 227, 3)
# print(batch_lab_array)
# for i in range(10):
#     name = str(i)
#     cv2.imshow(name,batch_img_array[i])

