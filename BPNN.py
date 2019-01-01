import numpy as np
import gen_data as gd
import random

sample_num = len(gd.train)  # 样本总数 60000
input_num = len(gd.train[0])  # 输入层节点数 784
hidden_num = 30  # 隐层节点数
output_num = 10  # 输出节点数
wij = 0.2 * np.random.random((input_num, hidden_num)) - 0.1  # 初始化输入层和隐藏间权矩阵
wjk = 0.2 * np.random.random((hidden_num, output_num)) - 0.1 # 初始化隐层和输出层间权矩阵
hidden_offset = np.zeros(hidden_num)  # 隐层偏置向量
output_offset = np.zeros(output_num)  # 输出层偏置向量
input_learn_rate = 0.3  # 输入层权值学习率
hidden_learn_rate = 0.3  # 隐层权值学习率
err_th = 1e-6  # 训练精度


# 激活函数
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# 训练过程
random_sample = random.sample(range(0, sample_num), 60000)
cnt = 0
for num in random_sample:
    cnt += 1
    print("正在训练第%s组数据..." % cnt)
    out_t = np.zeros(output_num)  # 初始化实际输出矩阵
    index = gd.train_labels[num]
    out_t[index] = 1  # 实际输出矩阵 例如[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]代表6
    # 前向过程
    hidden_value = np.dot(gd.train[num], wij) + hidden_offset  # 计算隐层值
    hidden_act = sigmoid(hidden_value)  # 隐层激活值
    output_value = np.dot(hidden_act, wjk) + output_offset  # 计算输出层值
    output_act = sigmoid(output_value)  # 输出层激活值

    # 后向过程
    e = out_t - output_act  # 计算输出误差
    output_delta = e * output_act * (1.0 - output_act)  # 计算输出层delta
    hidden_delta = hidden_act * (1.0 - hidden_act) * np.dot(wjk, output_delta)  # 计算隐层delta
    for i in range(0, output_num):
        wjk[:, i] += hidden_learn_rate * output_delta[i] * hidden_act  # 更新隐层到输出层的权向量
    for i in range(0, hidden_num):
        wij[:, i] += input_learn_rate * hidden_delta[i] * gd.train[num]  # 更新输入层到隐层的权向量
    output_offset += hidden_learn_rate * output_delta  # 输出层偏置更新
    hidden_offset += input_learn_rate * hidden_delta  # 隐层偏置更新

    if 0.5 * np.dot(e, e) < err_th:
        break

print("训练完毕！")

# 用测试集测试
print("正在测试...")
right = np.zeros(10)
numbers = np.zeros(10)

# 统计测试数据中各个数字的数目
for i in gd.test_labels:
    numbers[i] += 1

for num in range(len(gd.test)):
    hidden_value = np.dot(gd.test[num], wij) + hidden_offset  # 隐层值
    hidden_act = sigmoid(hidden_value)  # 隐层激活值
    output_value = np.dot(hidden_act, wjk) + output_offset  # 输出层值
    output_act = sigmoid(output_value)  # 输出层激活值
    if np.argmax(output_act) == gd.test_labels[num][0]:
        right[gd.test_labels[num]] += 1

# 正确的个数
print("正确的个数：")
print(right)
# 测试集总个数
print("测试集总个数：")
print(numbers)
# 每个数字正确率
print("每个数字正确率：")
print(right / numbers)
# 总正确率
print("总正确率：")
print(right.sum() / len(gd.test))

# 从测试集中随机抽取一张图片测试
i = np.random.randint(0, len(gd.test))
hidden_value = np.dot(gd.test[i], wij) + hidden_offset  # 隐层值
hidden_act = sigmoid(hidden_value)  # 隐层激活值
output_value = np.dot(hidden_act, wjk) + output_offset  # 输出层值
output_act = sigmoid(output_value)  # 输出层激活值
print("该数字正确结果为：")
print(gd.test_labels[i][0])
print("该数字预测结果为：")
print(np.argmax(output_act))
if gd.test_labels[i][0] == np.argmax(output_act):
    print("识别正确")
else:
    print("识别错误")