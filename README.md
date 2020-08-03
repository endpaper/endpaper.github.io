[网页版](https://endpaper.github.io)

```php+HTML
<head>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            inlineMath: [['$','$']]
            }
        });
    </script>
</head>
```
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

   

# Deep Learning

# Reinforcement Learning

# Transfer Learning

# Federated Learning
