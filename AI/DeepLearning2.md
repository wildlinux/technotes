[Toc]

# 1. LESSION1: Neural Networks and Deep Learning

## 1.1. Week1



## 1.2. Week2 Logistic Regression as a Neural Network

### 1.2.1. Binary Classification

### 1.2.2. Gradient Descent

Slope of the cost function
slop=derivative
partial derivative, is the slope on given directions, if function has more than one variants. 
calculus

dJ(w,b)/dw denote how much the slope of J(w,b) at the direction of w.

### 1.2.3. Computation Graph

J = 3(a+bc) is consist of three steps:
- u=bc
- v=a+u
- J=3v

The derivatives of J to a, b, c can be caculated backward, from J to v to u to a, b and c.

### 1.2.4. Python and Vectorization

a = np.random.randn(5,1), dont use a = np.random.randn(5)

assert(a.shape == (5,1)) helps but it's expensive

### 1.2.5. Execise

- Reshape each train data to a vertical vector (x, 1)
- Standardize the elements

。

## 1.3. Week3 Shallow Neural Network

### 1.3.1. What's a Neural Network

### 1.3.2. Nerual Network Representation

- 矩阵X 是列向量的集合

- 中间生成的矩阵 Z^[i]^ , A^[i]^ 
    - 每个列下标都，都对应一个样本（0层的输入），或者一个中间的输入。
    - 每一个行下标，都对应该层的一个神经元。
- 输入参数 W^[i]^在第一层都是一个矩阵, b^[i]^ 是列向量。
    - W的形态要根据输入输出来计算
        - 行数对应样本数或者输入数（上一层神经元数）
        - 列数对就输出（下一层神经元数）
  
### 1.3.3. Actiation Function

- Don't use sigmoid
- tenh is better than sigmoid
- Relu is default choice
- Leaking Relu

### 1.3.4. Why do we need non-linear activation functions

### 1.3.5. Derivtives of Activation Function

if a=g(z)= sigmoid(z) g'(z)=a(1-a)

if a=g(z)=tanh(z)  g'(z) = 1 - a<sub>[2]</sub><sup>[3]</sup>

if g(z)=Relu(z) = max(0,z)


### 1.3.6. Formulas

Forward propagation

### 1.3.7. Ian Goodfellow interview

- 对抗学习网络

    - 训练一个非常深的网络，读所有的网络文章；然后把某类文章（谣言）大批给它，让它打出类似的。
    - 背后的思想：
        - 先训练一个非常专于“阅读”的深度网络：
            - 一个方法是让它分辨有人类文献与随机文本
            - 既然google翻译学习的全球所有文字，那它可以作为这样一个“语言专家”
        - 再让尝试找出“谣言”的识别模式
- Security
    - security has to be set from the beginning, it's hard to patch it later
    - Secucrity should be taken into consideration before development
    - anticipate security problems regarding AI now, and put that into the algorithm. 

## 1.4. Week4 Deep Neural Network

### 1.4.1. What's a deep neural network

### 1.4.2. Forward Propagation in a Deep Neural Network

### 1.4.3. Getting Your Matrix Demension Right

### 1.4.4. Why Deep Representation

- 深度网络学习到的权重是可以复用的。
    - 就像视频中说到的，第一层是识别“边”的，那这一层的权重可以利用到任何需要识别“边”的网络中。
- 任务分解
    - 把大任务分解成小任务比直接完成大任务要开销小的多
    - 类似于深度学习比shallow学习的优势

### 1.4.5. Forward and Backward Propagation

A[i] => W[i+1] b[i+1] -> Z[i+1] => A[i+1] =>....=>A[e], Y => L

dA[e]即 D(L)/D(A[e])

dA[i] -> dZ[i] -> dW[i], db[i] => dA[i-1]

# 2. LESSION2: Improving DNN - Hyperparameter Tuning, Regularization, Optimization

## 2.1. Week1

### 2.1.1. Setting Up

- Train/dev/test 
    - ratio is like 98/1/1 in big data era.
    - come from the same dataset
- Bais and Variance
    - Bais : underfitting, 常量影响太大
    - Variance : overfitting，变量影响太大
    - Samples, train error vs dev error:
        - %1 vs 15% : high variance
        - %15 vs %16: high bais, given the base error is 0
        - %15 vs 30%: high variance and high bais
        - %0.5 vs 1%: low variance and low bais
- Basic Recipt for Maching Learning
    - High Bais 
        - Choose bigger network
        - Trainning longer
        - Choose another NN arch
    - High Variance
        - More Data
        - Regularization
        - Choose another NN arch
  
### 2.1.2. Regularization

REGULARIZATION: 平衡参数，让参数都差不多，不太大。

- L2 Regularization
    - Add penalty to big W
    - Why it works
        - lambda penilize big W
        - By seting lambda to a big value, it zero W that simplify the NN
- Dropout Regularization
    - set probability to remove nodes of each layer
    - Samples
        - Inverted Dropout
    - Why it works
        - can't rely too much on certain features
- Other Regularization Methods
    - Data Augement
    - Early Stop
        - Computational cheap option
        - Downside
            - mix two problems, underfitting and overfitting, to make it complicated. It's better to address one at a time.

## Optimization

### Normalize

- 让输入分布正态化

### Vanishing and Expoding gradients

- 对于一个很深的NN（L很大）
  - 中间所有层的总体效果可以简明理解为 W^L^，类似指数增长
    - 如果W的每个值小于1，那W^L^将接近于0，即vanishing
    - 如果W的每个值大于1，那W^L^将非常大，即expoding
- Solution
  - carefully choose initial W

### Checking your derivative computation

estimate derivative: f(a+e) - f(a-e) / 2e

## Week2 Optimization Algorithms

### Mini-batch gradient descent

How to choose batch size? 

A size that best use the H/W and vectorization alogrithm while takes less time. The overall target the balance of time and accuracy. 

- Bigger size, more accurate
- Smaller size, faster
- Typical size:
    -  2^n^, n > 8 
    -  smaller than the memory size of CPU/GPU

> 每一迭代使用不同的batch size

### Exponentially weighted averages

- 指数加权平均

- Bias correction
    - 对于启动初期的值进行一定的调整，避免由于启动值为0，而导致的初期值偏小的问题。

### Gradient descent with momentum

- 理解：
  - 由于使用了最近几次（1/(1-beta)）的（指数加权）平均值去更新W/b，使得震荡相互抵消了，但主移动方向得到了保留和加速。
  - 加速滚落的雪球。每次速度都有提高。

### Root Mean Square prop

- 用以保证每次更新都不太大。通除以均方根，大的除大的，小的除小的，结果就差不多了。
- 感觉主要是预防 b 和 W 调整时，某个调整过大，导致影响过大。通过除一个平均值，直到类似normalization的效果。

### Adam optimization algorithm

### Learning rate decay

## Week3 Hyperparameter tuning, Batch Normalization and Programming Frameworks

### Hyperparameter Tuning

- 搜索空间对不同量级使用同样的搜索资源。例如
  - 0.1 0.01 0.001 这三个范围内设置同样数量的采样点

### Batch Normalization

- Key idea
    - As A^[0]^,  Normalize A^[l-1]^ to improve efficency of learning W[l] and b[l] for a DNN.
    - In practice, Z^[l]^ is normalized than A^[l]^
    - Gussain
        - Convert Z to a Gussian Distribution value with controlled mean and variance.
- Fitting Batch Norm into a NN
    - alpha and beta is updated as W b 
    - initialized as ...
    - b^[l]^ is not needed any more, since beta has the similiar effect, shifting and adding bias.
    - the sacale of beta and alpha is [n^[l]^, 1]
    - 因为是对X或A加了变化，所以对w求导不变
- Why does it work
    - First
      - it decouple or weaken the dependency between layers
      - later layer will see a more stable input
      - each layer can learn indenpendently
    - Second
        - it has regularization effect
        - similiar with dropout, it add noise by shifting and rescale input. It force later unit can't rely on certain set of input.
        - Used together with dropout, it will has more powerful regularization effect.
        - big mini-batch size has small regularization effect for it has more accurate mean and variance.
        - regularization is only a side effect.
- Batch norm at test time
    - the mu and sigama^2^ are computed based on current minibatch in trainning process. 
    - How to get a reasonable mu and sigama in test?
    - Exponential Weight Avarage is used to put together all iteration of mu and sigama to get the final values, which could be used in test and prediction.

> 讲到算法在更新W3时，其实是基于已经给定的W1和W2；结果呢，W1和W2也被更新了，那W3对更新后的W1和W2说不定是正向调整还是反向调整了。那理论上讲，应该先把W1调到最优，再W2，依此类推。反思：这个确实不符合实际，人的理论设计是一层层学习，实际经验是可能学到W4后（写文章后），反调W1（再认识更多的字）。

> NN和人一样，要在多干扰的情况下练习，那抗干扰能力也强。

### Softmax Regression

- concept
    - Softmax is a certain type of activation function
    - it output the same shape of vector as the input
    - computation is quite simple: a^i^ = e^zi^/sum(e^zi^), which indicate the probability.

### Programming Framework

- The critiria to choose framework
    - easy to develop and deploy
    - true open in the long run
    - run rapidly 
- Tensorflow
    - tf.Variable(formula, name)
      - need initialization
      - 如果没有算式，直接调用tf函数，则不需要定义Variable
        - tf的函数可以想像成一个子图，自己会initialize自己的Variable.
      - 如果自己定义的计算过程（计算图），则需要Variable，需要init.  
# LESSION3: Structuring Machine Learning Projects

## WEEK1

### Introduction to ML Strategy

- What
  - To find the promising diretion to improve ML algorithm
  - So what optimiation mothed, which is worth a while to try.
- Orthogonalization
  - 含义：指一个开关调节一种指标，单一相关，互不干扰。
  - ML需要调节的主要指标
    - 首先是保证在training set上表现优秀
      - big network
      - Adam
      - ... 
    - then dev set
      - Regularization
      - Bigger trainning set
    - then test set
      - Bigger dev set
    - then real world
      - change cost function
      - bigger dev set 
    - 如果某种调节（如 early stop）会同时影响以上两种指标，那么在调节时就会引起困惑。到不是绝对不可有。
- Setting up your goal
    - 使用一个单一数字来衡量模型的好坏。不然就很难决策。
- Satisficing and Optimizing metric
    - If you have metric to consider, it's better o choose one as Optimizing metric, which need to be as better as possible, and the rest N-1 as Satisficing metric, which need to be above certain given threshold.
- Train/dev/test distributions
    - dev set + metric is the target of your which, that it's very important to keep the dev and test set come from the same distribution.
- Size of the dev and test sets
    - use small ratio of data for dev and test if you have a real big data set
- When to change dev/test sets and metrics
    - when your dev metric don't match your actual targets.

> 启示：单一衡量指标+若干基线指标 这个思路在生活工作中也有个不错的思路。

## WEEK2

# 3. Python

for l in range (L):
    print (" l = " + str(l) + " of " + str(L))

loop from 0 to L-1

# 4. 一些想法

## 4.1. 人物对话中学到的

### 4.1.1. Geoffrey Hinton interview

- Brain is not sympolic
- Unsupervised learning is more important
- read some paper then follow your instinct
- 跟随自己的直觉。跟随自己的直觉当然可能失败，但如果不跟随自己的直觉什么都不会做成。

### 4.1.2. Pieter Abbeel interview

- 算法还不擅长在长时间维度上识别 pattern , 从一天或一生中总结出什么规律。
- 算法可以识别一些pattern， 但如何使用这些pattern是另一个问题。像人会有一些认识或想法，如何利用。
- 设计一个生成 增加深度学习算法 的算法。
- 学习就要自己多多练习

### Yoshua Bengio interview

- long term dependency 
- Joint distribution
- neural machine translation with **ATTENTION**
- Toy experients are faster to iterate and easy to manipulate. It's a nice way to research more quickly.
- Ask good questions and under the underlying principles, **WHY** is more important
- **ICLR** NIPS ICML have good papers

### Yuanqing Lin interview



## 4.2. 方字就是抽象概念

任何现实生活中的色相，都是用文字描述的，文字即概念，没有人可以离开文字思考，文字及抽象。

图、音乐有类似的抽象作用。

所以建议一个AI，通过少量观察提取出共同的“模式pattern”，然后给这个pattern起个名字，这就是抽象的过程。然后拿这个pattern去套各种东西，套出新的pattern（原pattern的变形、组合、异化），进行新的抽象（起名），以此扩展。现通过与其他AI或要人类的确认（增强学习：问这是xxx吗？回答：Y/N），从而进一步增强。

## 4.3. 同样的网络不同的权重

尽量使用同构的网络，这样使用不同的参数值就可以识别不同的东西。

使用多个这样的网络，类似模拟大脑的不同部分。但可以比大脑大的多。

## 4.4. 实现不同功能的互通

音乐转成描述音乐的文字

图片转成描述

描述转成图

以上就是类似人类的想像。

## 4.5. 常识

人们常拿重力来举例人类的常识。可实则重力是对生活观察的结果的抽象。有了抽象能力就能形成重力的概念，也就是能“想象/生成”提起重物时的图像。

## 4.6. 抽象概念

抽象概念即是连接：图片可以连接到文字，文字可以连接到声音，声音可以连接到图片和文字

但文字好像更基础，思考时大部分用声音，或图片。盲人用声音，哑巴用图片。所以什么都是可以用来思考的。

连接连接连接。

概念涉及另一个重要的点就是：人这个个体与其他体会的共同、同理心。例如：我的妈妈是具体的，很容易定义；然后相对于我，你也是你的妈妈；然后就提出了妈妈这个抽象概念。这里我想说的重点是：人类是由不同的同质个体组成的，所以有共同的一些概念。如果AI是一个大个体，它也要能切换视角来理解一些概念。对于它没有的概念（如赤道人对雪的理解）它也不能真正理解。

人的所谓经验，其实就是感官的输入。AI也有各种sensor，所以经验应该给人要丰富的多。

一个简单的故事 这个世界每个人都想当影帝