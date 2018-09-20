# Neural Networks and Deep Learning

## Week1

### Geoffrey Hinton interview

- Brain is not sympolic
- Unsupervised learning is more important
- read some paper then follow your instinct

## Week2 Logistic Regression as a Neural Network

### Binary Classification 

### Gradient Descent

Slope of the cost function
slop=derivative
partial derivative, is the slope on given directions, if function has more than one variants. 
calculus

dJ(w,b)/dw denote how much the slope of J(w,b) at the direction of w.

### Computation Graph

J = 3(a+bc) is consist of three steps:
- u=bc
- v=a+u
- J=3v

The derivatives of J to a, b, c can be caculated backward, from J to v to u to a, b and c.

### Python and Vectorization

a = np.random.randn(5,1), dont use a = np.random.randn(5)

assert(a.shape == (5,1)) helps but it's expensive

### Execise

- Reshape each train data to a vertical vector (x, 1)
- Standardize the elements

### Pieter Abbeel interview

- 算法还不擅长在长时间维度上识别 pattern , 从一天或一生中总结出什么规律。
- 算法可以识别一些pattern， 但如何使用这些pattern是另一个问题。像人会有一些认识或想法，如何利用。
- 设计一个生成 增加深度学习算法 的算法。
- 学习就要自己多多练习。

# Week3 Shallow Neural Network

## What's a Neural Network

## Nerual Network Representation

- 矩阵X 是列向量的集合

- 中间生成的矩阵 Z^[i]^ , A^[i]^ 
    - 每个列下标都，都对应一个样本（0层的输入），或者一个中间的输入。
    - 每一个行下标，都对应该层的一个神经元。
- 输入参数 W^[i]^在第一层都是一个矩阵, b^[i]^ 是列向量。
    - W的形态要根据输入输出来计算
        - 行数对应样本数或者输入数（上一层神经元数）
        - 列数对就输出（下一层神经元数）
  
### Actiation Function

- Don't use sigmoid
- tenh is better than sigmoid
- Relu is default choice
- Leaking Relu

### Why do we need non-linear activation functions

### Derivtives of Activation Function

if a=g(z)= sigmoid(z) g'(z)=a(1-a)

if a=g(z)=tanh(z)  g'(z) = 1 - a^2^

if g(z)=Relu(z) = max(0,z)

### Formulas

Forward propagation

### Ian Goodfellow interview

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

# Week4 Deep Neural Network