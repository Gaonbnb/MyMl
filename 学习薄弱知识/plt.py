# meshgrid函数可以让两个坐标轴生成一个网格图
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
xx, yy = np.meshgrid(x, y, sparse=True)
z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
h = plt.contourf(x,y,z)
plt.show()

# enumerate(swapcase, [start=0]). 自动捆绑索引
seasons = ["spring", "summer", "fall", "winter"]
l1 = list(enumerate(seasons))
l2 = list(enumerate(seasons, start=1))
print(l1)
print(l2)
# for 使用
seq = ["one", "two", "three"]
for i, element in enumerate(seq):
    print(i, element)
    
# matplotlib.pyplot.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
# 自动调整子图参数，是指填充整个图像区域
#在 matplotlib 中，轴域（包括子图）的位置以标准化图形坐标指定。有时轴标签或标题（有时甚至是刻度标签）
#会超出图形区域，而因此被截断。为了避免这种情况，轴域的位置需要调整，
#这可以通过调整子图参数来解决（移动轴域的一条边来给刻度标签腾地方），
#而Matplotlib的命令tight_layout(）也可以自动解决这个问题；当你拥有多个子图时，
#会经常看到不同轴域的标签叠在一起，如下图所示，而tight_layout()也会调整子图之间的间隔来减少堆叠。

# numpy.ravel()
# numpy.flatten(()
# 上面两个函数的功能一致，都是将多维数组降为一维，区别在于返回拷贝（copy）还是返回视图（view）
# flatten（）将多维数组拉成一维，返回的是一份拷贝，对拷贝所作的操作不会影响原始矩阵，ravel会影响源实矩阵


#np.c_中的c是column（列）的缩写，是按列叠加两个矩阵的意思，也可以说是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()

#np.r_中的r是row（行）的缩写，是按行叠加两个矩阵的意思，也可以说是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等，类似于pandas中的concat()

# numpy中关于向量的使用
import numpy as np
a = np.random.randn(5)
# a是一个置为一的矩阵，特点只有一个方括号
np.dot(a, a.T)
#[[1,2,34,5]] 代表一个1行5列的行向量，特点两个方括号

#向量的建造 reshape（）

# z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#这里实际是将meshgrid得到的坐标矩阵的横坐标和纵坐标进行配对（拼接）得到网格点的完整坐标，然后进行分类预测

# cs = plt.contourf(xx. yy, Z,cmap=plt.cm.RdYlBu)
#等高线是三维函数在二维平面的投影，生成等高线图需要三维点（xx，yy）和对应的高度值Z（由clf.predict()生成的预测值）；这里的x，y是由meshgrid()在二维平面中将每一个x和每一个y分别对应起来编织成网格，cmap指定了等高线间的填充颜色。