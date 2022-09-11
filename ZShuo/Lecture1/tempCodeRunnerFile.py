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
label = [[i]*200 for i in range(len(classes))]

print(label)


for l in classes:
    data.append(images[targets == l][: 200])
    
print(data)   