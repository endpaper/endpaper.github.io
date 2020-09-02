[网页版](https://endpaper.github.io)

[TOC]



# Machine Learning

## Gradient Descent

$$
Loss(\theta)=\frac {1} {2m}\sum_ {i=0}^{m}(f_\theta(x^i)-y^i)^2
$$

### Batch Gradient Descent(BGD)

​	在每轮迭代时，BGD计算所有样本的残差平方和，然后计算梯度并更新参数，结束本轮迭代。也就是说BGD算法中，每完成一轮迭代（观测完所有的训练样本）只能进行一次参数更新。

​	对应的算法伪代码：

```python
for it in range(1000): # 1000次迭代
    params_grad = evaluate_grad(loss_function, data)
    params := params - learning_rate * params 
```

### Stochastic Gradient Descent(SGD)

​	SGD每观测到一条数据，都会进行一次参数更新，当完成一轮迭代时，参数已经更新了m次（假设训练样本的样本容量为m，相比于BGD，SGD参数更新的速度更快。

​	对应的算法伪代码：

```python
for it in range(1000): # 1000次迭代    
    shuffle(data)
    for (x_i, y_i) in data: # 观测所有训练样本
    	params_grad = evaluate_grad(loss_function, x_i, y_i)    
    	params := params - learning_rate * params 
```

| 算法 | 优点                                                         | 缺点                                                      |
| ---- | ------------------------------------------------------------ | :-------------------------------------------------------- |
| BGD  | 每次都向着整体最优化方向                                     | 参数更新比较慢，训练速度慢                                |
| SGD  | 参数更新比较频繁，训练速度快，但会造成 cost function 有严重的震荡 | SGD的噪音较BGD要多，使得SGD并不是每次都向着整体最优化方向 |

**BGD 可能会收敛到local minimal，当然 SGD 的震荡可能会跳出local minimal。**

### Mini-batch Gradient Descent(MBGD)

​	为了平衡训练时模型收敛的速度和收敛的稳定性，同时又能够充分利用深度学习库中高度优化的矩阵操作进行梯度计算，我们每一次采用n个样本进行计算，更新参数。

​	对应的算法伪代码：

```python
for it in range(1000): # 1000次迭代    
    shuffle(data)
    for mini_batch in get_mini_batch(data): # 观测所有训练样本
    	params_grad = evaluate_grad(loss_function, mini_batch)    
    	params := params - learning_rate * params 
```

## Feature Scaling

### 为什么需要Feature Scaling?

​	不同特征对应的特征值范围不同，往往这些特征值对应的范围相差很大，例如在做房价预测任务中，房子占地面积（S）这个特征的范围可能是100-200平方米 ，但是房子离最近地铁站的距离（L）可能在5000-10000米之间，两个特征值范围之前的差距很大，这时候就可能会产生一些问题：（1）损失函数中L对损失函数的函数值的影响更大，那么模型其实就潜在假设，相比于S特征，L特征对房价的影响程度更大。但是我们并不想模型会有这样一个偏见，也就是对特征值的差异化对待，我们希望模型训练之后从数据中学习到特征值的差异化，我们希望模型将这个差异化对待体现在参数theta中。（2）模型训练的速度缓慢。例如，在多变量线性回归任务中，将按照以下算法对参数进行更新。
$$
\theta_j:=\theta_j-\alpha\frac {1} {m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{i})x_j ^{i}
$$
由于不同特征值对应的范围空间差异很大，在学习率一定的情况下，范围越大的特征对应的参数更新的速度更快，反之那些范围极小的特征对应的参数更新的速度就很慢。对照下面这张图。

![Image for post](https://miro.medium.com/max/600/1*yi0VULDJmBfb1NaEikEciA.png)

​																				 	[Photo Credit](https://stackoverflow.com/questions/46686924/why-scaling-data-is-very-important-in-neural-networklstm/46688787#46688787)

​	首先看左边这张图，特征x1对应的范围远大于特征x2，也就是相比于x2，x1对损失函数的影响更大，x1对应参数w1的梯度值更大，在进行参数更新时，x1对应参数w1更新的速度更快。由图可以看出，固定w1时，w2的变化对损失函数的函数值影响不是很大，但是在固定w2时，w1的变化对损失函数的函数值影响非常大。所以左边这张图呈现扁平椭圆状，假设选定参数初始值在红线的起点位置，那么红线所画出来的就是参数更新的轨迹（每次朝等高线法向方向移动），参数更新时要走很多弯路，导致参数更新的很慢。反之，右图是我们在预处理中做过Feature Scaling之后，模型训练时参数的更新轨迹，x1特征值和x2特征值对应的范围相当，所以等高线更接近圆形，参数更新时，一直按着圆的半径方向走向loss funciton的global mimal。

### Feature Scaling的几种方法

#### Min-Max Scaler

$$
x_{new} = \frac {x-x_{min}} {x_{max}-x_{min}}
$$

​	将特征值尺度映射到0-1。

#### Standatd Scaler

$$
x_{new} = \frac {x-\mu} {\sigma}
$$

\mu是均值，\sigma是方差，将特征值映射到均值是0方差是1的标准高斯分布上。

### 逻辑回归中的梯度推导

- 假设函数:

$$
h_{\theta}(x)=\frac 1 {1+e^{-\theta^Tx}}
$$

$$
\frac {\partial h_{\theta}(x)} {\partial\theta_j}=\frac {e^{-\theta^Tx}} {(1+e^{-\theta^Tx})^2}x_j
$$

- 损失函数：

$$
J(\theta)=-\frac 1 m \sum_{i=1}^{m}[y^{(i)}log(h_\theta(x^{(i)}))+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]
$$

$$
\frac {\partial J(\theta)} {\partial\theta_j}=-\frac 1 m \sum_{i=1}^{m}[y^{(i)}\frac {e^{-\theta^Tx^{(i)}}} {1+e^{-\theta^Tx^{(i)}}} x^{(i)}_{j} -(1-y^{(i)})\frac 1 {1+e^{-\theta^Tx{(i)}}} x^{(i)}_{j}]
$$

$$
=-\frac 1 m \sum_{i=1}^{m}[y^{(i)}(1-h_\theta(x^{(i)}))x^{(i)}_{j}-(1-y^{(i)})h_\theta(x^{(i)})x^{(i)}_{j}]
$$

$$
=\frac 1 m \sum_{i=1}^{m} (h_{\theta}(x^{(i)})-y^{(i)})x^{(i)}_{j}
$$

- 参数更新公式：

$$
\theta_j:=\theta_j-\frac \alpha m \sum_{i=1}^{m} (h_{\theta}(x^{(i)})-y^{(i)})x^{(i)}_{j}
$$

- 向量化表示：

$$
\theta:=\theta-\frac \alpha m X^T(g(X\theta)-\vec y)
$$

​	其中：
$$
g(z)=\frac 1 {1+e^{-z}}
$$

### 二元分类为什么不能用MSE做为损失函数？

### ROC曲线理解

- 何为ROC曲线？

  接受者操作特性曲线（receiver operating characteristic curve，简称[ROC曲线](https://baike.baidu.com/item/ROC曲线)），又称为感受性曲线（sensitivity curve）。得此名的原因在于曲线上各点反映着相同的感受性，它们都是对同一信号刺激的反应，只不过是在几种不同的判定标准下所得的结果而已。接受者操作特性曲线就是以虚惊概率（伪阳率）为横轴，击中概率（真阳率）为纵轴所组成的坐标图，和被试在特定刺激条件下由于采用不同的判断标准得出的不同结果画出的曲线

- 理解混淆矩阵

  在评价分类模型性能的时候，可以使用多个指标，其中精确率（Precision）、召回率（Recall）和F1-Score是分类问题中经常使用的模型评价指标。如图为模型评价时使用到的混淆矩阵（Confusion Matrix）。其中每个统计量代表的含义如下：

  •  TN：预测值为负（Negative），真实值也为负（Negative），预测正确

  •  FN：预测值为负（Negative），真实值为正（Positive），预测错误

  •  FP：预测值为正（Positive），真实值为负（Negative），预测错误

  •  TP：测值为正（Positive），真实值也为正（Positive），预测正确

![https://github.com/endpaper/endpaper.github.io/blob/master/images/%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5.png)

​	首先，介绍精确度和召回率的计算公式（以正类为例）：
$$
Precision = \frac {TP} {TP + FP}
$$

$$
Recall = \frac {TP} {TP + FN}
$$

​	然后，引入两个概念，True Positive Rate（真阳率）和False Positive Rate（伪阳率）：
$$
TPRate = \frac {TP} {TP + FN}=\frac {TP} P
$$

$$
FPRate = \frac {FP} {FP+TN}=\frac {FP} {N}
$$

​	**TPRate的意义是所有真实类别为1的样本中，预测类别为1的比例。**

​	**FPRate的意义是所有真实类别为0的样本中，预测类别为1的比例。**

- ROC曲线的绘制

  该曲线的横坐标是伪阳率，即FPRate，纵坐标为真阳率，即TPRate。

  当绘制完成曲线后，就会对模型有一个定性的分析，如果要对模型进行量化的分析，此时需要引入一个新的概念，就是AUC（Area under roc Curve）面积，就是指ROC曲线下的面积大小，而计算AUC值只需要沿着ROC横轴做积分就可以了。真实场景中ROC曲线一般都会在![y=x](https://math.jianshu.com/math?formula=y%3Dx)这条直线的上方，所以AUC的取值一般在0.5~1之间。AUC的值越大，说明该模型的性能越好。

  ![img](https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1599068113839&di=8310211a27812cdd69ea7bdde2651990&imgtype=0&src=http%3A%2F%2Fimg.mp.itc.cn%2Fupload%2F20170326%2Fb5b1832aec144072b4056e54d71b1a62_th.png)

- Why AUC?

  ROC曲线的应用场景有很多，根据上述的定义，其最直观的应用就是能反映模型在选取不同阈值的时候其敏感性（sensitivity, FPR）和其精确性（specificity, TPR）的趋势走向。不过，相比于其他的P-R曲线（精确度和召回率），ROC曲线有一个巨大的优势就是，当正负样本的分布发生变化时，其形状能够基本保持不变，而P-R曲线的形状一般会发生剧烈的变化，因此该评估指标能降低不同测试集带来的干扰，更加客观的衡量模型本身的性能。假设负类样本数量扩大十倍，那么FP和TN的数量也会随之增大，这将影响到正类的精确度，假设正类样本数量扩大十倍，那么TP和FN的数量也会随之增大，这将影响到负类的召回率。但ROC曲线的横纵坐标俩个值，FPRate只考虑第二行，N若增大10倍，则FP、TN也会成比例增加，并不影响其值，TPR更是只考虑第一行，不会受到影响。

- ROC曲线绘制原理

  如图为一二分类模型的真实输出结果，Score为其预测为正类的概率，假设我们把阈值设置为0.9，那么只有一个正类样本被正确分类，因此真阳率为0.1，因为没有负类样本误分为正类，所以伪阳率为0，于是坐标（0,0.9）落在ROC曲线上。依次选择不同的阈值（或称为“截断点”），画出全部的关键点以后，再连接关键点即可最终得到ROC曲线如下图所示。

  <img src="https://upload-images.jianshu.io/upload_images/11525720-cb0c836e33757b87.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp" alt="img" style="zoom:50%;" />

  [Photo Credit]: https://upload-images.jianshu.io/upload_images/11525720-cb0c836e33757b87.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp	"二分类模型的真实输出结果"

  ![img](https://upload-images.jianshu.io/upload_images/11525720-dd2545eaaaa7c2ba.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

[Photo Credit]: https://upload-images.jianshu.io/upload_images/11525720-dd2545eaaaa7c2ba.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp	"ROC曲线示意图"

- AUC面积的意义

  如果两条ROC曲线没有相交，我们可以根据哪条曲线最靠近左上角哪条曲线代表的学习器性能就最好。但是，实际任务中，情况很复杂，如果两条ROC曲线发生了交叉，则很难一般性地断言谁优谁劣。在很多实际应用中，我们往往希望把学习器性能分出个高低来。在此引入AUC面积。

  在进行学习器的比较时，若一个学习器的ROC曲线被另一个学习器的曲线完全“包住”，则可断言后者的性能优于前者；若两个学习器的ROC曲线发生交叉，则难以一般性的断言两者孰优孰劣。此时如果一定要进行比较，则比较合理的判断依据是比较**ROC曲线下的面积**，即**AUC**(Area Under ROC Curve)。

- 参考：

​	https://www.jianshu.com/p/2ca96fce7e81

​	https://baike.baidu.com/item/AUC/19282953?fr=aladdin



# Deep Learning

# Reinforcement Learning

# Transfer Learning

# Federated Learning

# 周懂一点[论文精读]
## Data-to-text
### Title: A Hierarchical Model for Data-to-Text Generation
- Author：

Cl´ement Rebuﬀel, Laure Soulier, Geoﬀrey Scoutheeten, and Patrick Gallinari

- Abstract：

将结构化数据转录为自然语言描述已成为一项具有挑战性的任务，称为“数据到文本”。这些结构通常会重新组合多个元素及其属性。大多数尝试都依赖于将元素线性化为序列的翻译的编码器/解码器方法。但是，这丢失了大多数数据中包含的结构信息。在这项工作中，我们提出使用分层模型来克服此限制，该模型在元素级别和结构级别对数据结构进行编码。在RotoWire数据集上的评估显示了我们模型w.r.t.在定性和定量指标上的有效性。

- Why：

一方面，目前大多数的研究工作都是集中在解码器部分，对于数据仅仅是简单地线性化处理。
另一方面，大多数在编码器部分采用RNN（包括GRU和LSTM），这就需要顺序输入数据，这种编码无序数据的方式影响了学习效果。

- What：

Encoder：a two-level Transformer architecture with key-guided hierarchical attention

Decoder：a two-layers LSTM network with a copy mechanism [引用]

- How：

该文的主要贡献和研究是在编码器部分。作者首先将数据集处理为key-value形式的键值对记录r_(i,j)，而实体e_i={r_(i,1),r_(i,2),…,r_(i,J)}则是与该实体(人名或者团队名)相关的记录的集合，总的数据结构表示为s={e_1,e_2,…,e_I}，每条数据表示为(s,y)，其中y表示人工撰写的参考文本。
为了获取数据的结构信息，作者引入两层Transformer分别对记录和实体进行编码，并且通过hierarchical attention机制得到编码器的上下文语境再反馈给解码器。

![Image](https://raw.githubusercontent.com/endpaper/endpaper.github.io/master/data%20to%20text%20p1_1.png)

Low-level encoder：通过transformer比较不同记录r_(i,j)从而得到各个记录的隐藏层表示h_(i,j)。

High-level encoder：与low-level encoder相似，通过transformer比较不同底层实体表示h_i从而得到各个实体对应的隐藏层状态e_i。对所有实体得出的隐藏层状态e_i进行求平均值得到编码器最终的输出z，并用于解码器的初始化。

\alpha表示实体层(high)的注意力分值，\beta表示记录层(low)的注意力分值，与传统分层注意力机制不同的是，\beta如果只关注记录(key,value)中的key效果会比关注整个记录更好，value会造成噪音影响。如下图所示，同时关注key和value会使得模型无法准确地选择最恰当的数据。

![Image](https://raw.githubusercontent.com/endpaper/endpaper.github.io/master/data%20to%20text%20p1_2.png)

- Result：

![Image](https://raw.githubusercontent.com/endpaper/endpaper.github.io/master/data%20to%20text%20p1_3.png)

Wiseman是一种标准的带拷贝机制的编码器-解码器结构的模型。

Li是一种标准的带延迟拷贝机制的编码器-解码器结构的模型。

Puduppully-plan是先生成plan再根据其生成文本的模型。

Puduppully-updt是具有动态实体表示模块的编码器-解码器结构的模型（分层注意力机制）。

Flat是本文中编码器部分采用Transformer替换RNN的标准的编码器-解码器结构的模型。

Hierarchical-kv是本文中编码器部分采用对key和value都关注的分层Transformer的模型。

Hierarchical-k是本文中编码器部分采用只关注key的分层Transformer的模型。

通过比较Flat与Hierarchical-k/Hierarchical-kv模型，可以看出对结构化数据进行分层编码比直接顺序输入结构化数据的效果更好。
通过比较Hierarchical-k与Hierarchical-kv模型，可以看出只关注key而忽略value的效果更好，关注value可能会造成噪音效果。
通过比较Flat与Wiseman模型，可以看出Transformer应该比RNN更适合于结构化的数据。
通过比较Hierarchical-k/Hierarchical-kv与Li、Puduppully-plan模型，可以看出在编码器部分获取结构化信息比在解码器部分预测结构化信息效果更好。
通过比较Hierarchical-k与Puduppully-updt模型，可以看出Puduppully-updt在CO和RG-P%指标上更好而在其他指标上则相对较差，并且Hierarchical-k模型对数据的结构化信息只需要编码一次就可全程通用而Puduppully-updt则需要不断更新实体表示明显增加了计算的复杂度，Hierarchical-k模型显得更轻量且以更人性化的方式获得准确的流畅描述。

事实上，Hierarchical-k与Puduppully-updt有很大的相似之处，二者都在各自的模型中引入了分层注意力机制，通过先关注实体再关注实体对应的记录来获取数据的结构化信息，不同的是前者将该分层思想应用于编码器部分而后者则应用于解码器部分。另外，前者采用Transformer来替换后者所采用的RNN(LSTM)并且在分层注意力机制中选择性去除对value的关注。

## Recommender System

### Title: Fighting Boredom In Recommender Systems With Linear Reinforcement Learning

- PDF:https://github.com/endpaper/endpaper.github.io/blob/master/PDF/7447-fighting-boredom-in-recommender-systems-with-linear-reinforcement-learning.pdf
- PPT:https://github.com/endpaper/endpaper.github.io/blob/master/PPT/7447-fighting-boredom-in-recommender-systems-with-linear-reinforcement-learning.pptx
