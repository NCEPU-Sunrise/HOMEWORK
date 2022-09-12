# HOMEWORK

Sunrise Machine Learning Homework

## Lecture 1

### 2022.9.3

使用以下代码实现KNN手写数字的分类，K的取值自选。提交至[GitHub仓库](https://github.com/NCEPU-Sunrise/HOMEWORK)的个人文件夹，并在文件夹下建立文件夹Lecture1，推荐在该文件夹下新建README.md文件记录任务以及完成情况。提交的内容需要有代码以及分类的结果。根据下列提示获得训练集和测试集，在2022.9.10 00:00:00之前提交完成。

```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

images, targets = fetch_openml("mnist_784", return_X_y=True, as_frame=False)

# 可视化图片
images = images.reshape(-1, 28, 28)
plt.imshow(images[0], cmap='gray')
plt.show()

# 取3个类各200张图完成即可
# 示例取类2,3,4
classes = ['2', '3', '4']
data = []
label = [[i]*200 for i in range(len(classes))]
for l in classes:
    data.append(images[targets == l][: 200])
data = np.concatenate(data, axis=0)
label = np.concatenate(label)
data = data.reshape(600, -1) / 255 # 压缩值0-1之间方便计算距离

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, label)

# 建立映射方便表示。由于我们只取3类,映射为0,1,2就行
class_map = {'2' : 0, '3': 1, '4': 2}
```

## Lecture 2

### 2022.9.11

使用Lecture1中的代码产生数据集或者自己构建分类数据集（必须是图片），使用numpy写两个一层神经网络分别实现二分类和多分类，即一个是Logistic Regression，一个是Softmax Regression。前者是后者的特例。使用梯度下降算法更新参数。在2022.9.17 00:00:00之前提交完成。**在完成代码的过程中，学会python以及numpy的使用，为学习新的框架打基础，有基础了就可以自主学习pytorch，为后面实现更深的神经网络做铺垫。**