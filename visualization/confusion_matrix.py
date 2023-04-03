#coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
 
save_fig = True
 
# confusion = confusion_matrix(y_test, y_pred)
confusion = np.array([[97, 2,  0,  0, 1, 0],
                     [ 4, 94,  1,  21, 0, 0],
                     [ 3,  2, 95,  0, 0, 0],
                     [ 0,  0,  0, 98, 2, 0],
                     [ 3,  1,  0,  0,96, 0],
                     [ 0,  1,  3,  0, 6,90]])
 

#  [[170   6  13   1   2   0   0   1]
#  [  5 164   3   0   0   0   0   1]
#  [  7   2 198   0   1   0   2   0]
#  [  1   0   0  83   0   0   0   3]
#  [  1   0   0   0 155   3   9   0]
#  [  0   3   0   0   6 167   3   0]
#  [  0   5   4   1   6   3 206   1]
#  [  0   0   1   3   1   0   1  96]]
# plt.figure(figsize=(5, 5))  #设置图片大小
 
 
# 1.热度图，后面是指定的颜色块，cmap可设置其他的不同颜色
plt.imshow(confusion, cmap=plt.cm.Blues)
plt.colorbar()   # 右边的colorbar
 
 
# 2.设置坐标轴显示列表
indices = range(len(confusion))    
classes = ['A', 'B', 'C', 'D', 'E', 'F']  
# 第一个是迭代对象，表示坐标的显示顺序，第二个参数是坐标轴显示列表
plt.xticks(indices, classes, rotation=45) # 设置横坐标方向，rotation=45为45度倾斜
plt.yticks(indices, classes)
 
 
# 3.设置全局字体
# 在本例中，坐标轴刻度和图例均用新罗马字体['TimesNewRoman']来表示
# ['SimSun']宋体；['SimHei']黑体，有很多自己都可以设置
# plt.rcParams['font.sans-serif'] = ['TimesNewRoman']  
plt.rcParams['axes.unicode_minus'] = False
 
# config = {
#     "font.family":'Times New Roman',  # 设置字体类型
# }
# plt.rcParams.update(config)
 
# 4.设置坐标轴标题、字体
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.title('Confusion matrix')
 
# plt.xlabel('预测值')
# plt.ylabel('真实值')
# plt.title('混淆矩阵', fontsize=12, fontfamily="SimHei")  #可设置标题大小、字体
 
 
# 5.显示数据
normalize = False
fmt = '.2f' if normalize else 'd'
thresh = confusion.max() / 2.
 
for i in range(len(confusion)):    #第几行
    for j in range(len(confusion[i])):    #第几列
        plt.text(j, i, format(confusion[i][j], fmt),
        fontsize=16,  # 矩阵字体大小
        horizontalalignment="center",  # 水平居中。
        verticalalignment="center",  # 垂直居中。
        color="white" if confusion[i, j] > thresh else "black")
 
 
#6.保存图片
if save_fig:  
    plt.savefig("visualization/figure/confusion_matrix.pdf")
 
 
# 7.显示
# plt.show()