from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

images, targets = fetch_openml("mnist_784", return_X_y=True, as_frame=False)

# 可视化图片
images = images.reshape(-1, 28, 28)
plt.imshow(images[0], cmap='gray')
plt.show()

# 取3个类各200张图完成即可
# 示例取类2,3,4
classes = ['2', '3', '4']
data = []
label = [[i] * 200 for i in range(len(classes))]

print(label)

for l in classes:
    data.append(images[targets == l][: 200])

print(data)

data = np.concatenate(data, axis=0)
label = np.concatenate(label)
data = data.reshape(600, -1) / 255  # 压缩值0-1之间方便计算距离

# 划分数据集为训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(data, label)

# 建立映射方便表示。由于我们只取3类,映射为0,1,2就行
class_map = {'2': 0, '3': 1, '4': 2}


def predict(x_test, x_train, y_train, k):
    numSet = x_train.shape[0]

    dis = np.tile(x_test, (numSet, 1)) - x_train
    dis2 = dis ** 2
    sumdis = np.sum(dis2, axis=1)
    final_dis = np.sqrt(sumdis)

    sortDis = np.sort(final_dis)
    count = {}
    for i in range(k):
        sec = y_train[sortDis[i]]
        count[sec] = count.get(sec, 0) + 1

    maxCount = 0
    index = 0
    for v, temp in count.items():
        if v > maxCount:
            maxCount = v
            index = temp
    return index


pred = []
for k in range(2, 9):
    preds = []
    for j in range(X_test.shape[0]):
        pred_labels = predict(X_test[j], X_train, Y_train, k)
        pred.append(pred_labels)
    s = 0
    for i in range(len(preds)):
        if preds[i] == list(Y_test)[j]:
            s += 1
    preds_class = s / len(preds)
    pred.append(preds_class)

print(pred)
# uxqqlqj huuru
# 进程已结束,退出代码1

# ???????????????????????????????????