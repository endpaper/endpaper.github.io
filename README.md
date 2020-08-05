[网页版](https://endpaper.github.io)

# Machine Learning

> ## Gradient Descent

$$
Loss(\theta)=\frac {1} {2m}\sum_ {i=0}^{m}(f_\theta(x^i)-y^i)^2
$$

1. ### Batch Gradient Descent(BGD)

   ​	在每轮迭代时，BGD计算所有样本的残差平方和，然后计算梯度并更新参数，结束本轮迭代。也就是说BGD算法中，每完成一轮迭代（观测完所有的训练样本）只能进行一次参数更新。

   ​	对应的算法伪代码：

   ```python
   for it in range(1000): # 1000次迭代
       params_grad = evaluate_grad(loss_function, data)
       params := params - learning_rate * params 
   ```

2. ### Stochastic Gradient Descent(SGD)

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

3. ### Mini-batch Gradient Descent(MBGD)

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

1. Min-Max Scaler

$$
x_{new} = \frac {x-x_{min}} {x_{max}-x_{min}}
$$

​	将特征值尺度映射到0-1。

2. Standatd Scaler
   $$
   x_{new} = \frac {x-\mu} {\sigma}
   $$
   \mu是均值，\sigma是方差，将特征值映射到均值是0方差是1的标准高斯分布上。

# Deep Learning

# Reinforcement Learning

# Transfer Learning

# Federated Learning
