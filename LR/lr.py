import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset

# 导入数据_orig,并处理
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# 原始训练集shape(209,64,64,3),训练样本209,图片长64,宽64,RGB类别3原色
# 原始测试集shape(50,64,64,3),测试样本50,图片长64,宽64RGB类别3原色
m_train = train_set_x_orig.shape[0]  # 训练集大小
m_test = test_set_x_orig.shape[0]  # 测试集大小
num_px = train_set_x_orig.shape[1]  # 图片长宽度64x64

# print(m_train)
# print(num_px)

# 图片数据向量化
train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# print(train_set_x)
# print(test_set_x[:10])

# 数据标准化(3原色取值范围0-255)
train_set_x = train_set_x/255
test_set_x = test_set_x/255

# # testing data
# print(train_set_x.shape)
# print(train_set_y.shape)
# print(test_set_x.shape)
# print(test_set_y.shape)

# # (12288, 209)
# # (1, 209)
# # (12288, 50)
# # (1, 50)


# sigmoid 函数
def sigmoid(z):
    return 1/(1+np.exp(-z))


# 初始化参数权重(dim*1)=0,偏置b=0
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0

    return w, b


def propagate(w, b, X, Y):
    """
    传参:
    w -- 权重, shape： (num_px * num_px * 3, 1)
    b -- 偏置项, 一个标量
    X -- 数据集，shape： (num_px * num_px * 3, m),m为样本数
    Y -- 真实标签，shape： (1,m)

    返回值:
    cost， dw ，db，后两者放在一个字典grads里
    """

    # 获取样本数量
    m = X.shape[1]

    # 前向传播
    A = sigmoid(np.dot(w.T, X)+b)
    cost = -(np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))/m

    # 反向传播
    dZ = A - Y
    dw = (np.dot(X, dZ.T))/m
    db = np.sum(dZ)/m

    # print("dw" , dw)
    # print("db" , db)

    grads = {"dw": dw, "db": db}

    return grads, cost


def optimize(w, b, X, Y,  learning_rate, num_iterations, print_cost=False):
    # 定义costs[],存储迭代cost
    costs = []

    # 解决全局变量在局部中修改被Python认为无效 UnboundLocalError: local variable 'dw' referenced before assignment
    dw = 0
    db = 0
    grads = {'dw': dw, 'db': db}

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after interation %i: %f" % (i, cost))

    params = {"w": w, "b": b}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))

    A = sigmoid(np.dot(w.T, X) + b)

    # for i in range(m):
        # if A[0, i] > 0.5: # A[0,i] numpy赋予数组的一种写法表达
    #         Y_prediction[0, i] = 1
    #     else:
    #         Y_prediction[0, i] = 0

    for i in range(m):
        if A[0][i] > 0.5:
            Y_prediction[0][i] = 1
        else:
            Y_prediction[0][i] = 0

    return Y_prediction


def logistic_model(X_train, Y_train, X_test, Y_test, learning_rate=0.1, num_iterations=2000, print_cost=False):
    # 获得特征维度,初始化参数
    dim = X_train.shape[0]
    W, b = initialize_with_zeros(dim)

    # 梯度下降获取模型参数
    params, grads, costs = optimize(
        W, b, X_train, Y_train, learning_rate, num_iterations, print_cost)
    W = params['w']
    b = params['b']

    # 预测
    prediction_train = predict(W, b, X_train)
    prediction_test = predict(W, b, X_test)

    # 计算准确率
    acc_train = 1 - np.mean(np.abs(prediction_train - Y_train))
    acc_test = 1 - np.mean(np.abs(prediction_test - Y_test))

    print("Accuracy on train set:", acc_train)
    print("Accuracy on test set:", acc_test)

    # 为了便于分析和检查，我们把得到的所有参数、超参数都存进一个字典返回出来：
    d = {"costs": costs,
         "Y_prediction_test": prediction_test,
         "Y_prediction_train": prediction_train,
         "w": W,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations,
         "train_acy": acc_train,
         "test_acy": acc_test
         }

    return d


# 运行模型
d = logistic_model(train_set_x, train_set_y, test_set_x, test_set_y,
                   learning_rate=0.005, num_iterations=2000, print_cost=True)

# costs picture
plt.plot(d['costs'], 'o-')
plt.show()
