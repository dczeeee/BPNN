import os
import numpy as np
from PIL import Image


# 用图片生成train数据
def generate_train_data():
    train = []
    train_labels = []
    for root, dirs, files in os.walk("./trainimage"):
        for filename in files:
            # 打开一张图片
            file = os.path.join(root, filename)
            img = Image.open(file)
            # 图片灰度化
            img = img.convert("L")
            # 图片具体路径并打印
            print("正在用%s生成数据..." % file)
            # 将图片转换为数组形式，元素为图片像素的亮度值(0-255)，28*28->784，并归一化
            arr = np.asarray(img).reshape(-1) / 256.0
            train.append(arr)
            # 添加标签数据
            train_labels.append([int(file[18])])
    return train, train_labels


# 用图片生成test数据
def generate_test_data():
    test = []
    test_labels = []
    for root, dirs, files in os.walk("./testimage"):
        for filename in files:
            # 打开一张图片
            file = os.path.join(root, filename)
            img = Image.open(file)
            # 图片灰度化
            img = img.convert("L")
            # 图片具体路径并打印
            print("正在用%s生成数据..." % file)
            # 将图片转换为数组形式，元素为图片像素的亮度值(0-255)，28*28->784，并归一化
            arr = np.asarray(img).reshape(-1) / 256.0
            test.append(arr)
            # 添加标签数据
            test_labels.append([int(file[17])])
    return test, test_labels


train, train_labels = generate_train_data()
test, test_labels = generate_test_data()

print("正在存储训练与测试数据...")
# # 存储train数据
# traindata = './traindata'
# train_file = os.path.join(traindata, 'train.data')
# train_labels_file = os.path.join(traindata, 'train_labels.data')
# # 存储test数据
# testdata = './testdata'
# test_file = os.path.join(testdata, 'test.data')
# test_labels_file = os.path.join(testdata, 'test_labels.data')
#
# with open(train_file, 'w', encoding='utf-8') as f:
#     f.write(str(train))
#
# with open(train_labels_file, 'w', encoding='utf-8') as f:
#     f.write(str(train_labels))
#
# with open(test_file, 'w', encoding='utf-8') as f:
#     f.write(str(test))
#
# with open(test_labels_file, 'w', encoding='utf-8') as f:
#     f.write(str(test_labels))
print("存储完毕！")