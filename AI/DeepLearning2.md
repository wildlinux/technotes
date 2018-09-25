[Toc]

# Neural Networks and Deep Learning

## Week1



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

。

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

if a=g(z)=tanh(z)  g'(z) = 1 - a<sub>[2]</sub><sup>[3]</sup>

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

## What's a deep neural network

## Forward Propagation in a Deep Neural Network

## Getting Your Matrix Demension Right

## Why Deep Representation 

- 深度网络学习到的权重是可以复用的。
    - 就像视频中说到的，第一层是识别“边”的，那这一层的权重可以利用到任何需要识别“边”的网络中。
- 任务分解
    - 把大任务分解成小任务比直接完成大任务要开销小的多
    - 类似于深度学习比shallow学习的优势

## Forward and Backward Propagation 

A[i] => W[i+1] b[i+1] -> Z[i+1] => A[i+1] =>....=>A[e], Y => L

dA[e]即 D(L)/D(A[e])

dA[i] -> dZ[i] -> dW[i], db[i] => dA[i-1]

# Improving DNN - Hyperparameter Tuning, Regularization, Optimization

## Week1

### Setting Up

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
  
### Regularization

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

# Python

for l in range (L):
    print (" l = " + str(l) + " of " + str(L))

loop from 0 to L-1

# 一些想法

## 人物对话中学到的

### Geoffrey Hinton interview

- Brain is not sympolic
- Unsupervised learning is more important
- read some paper then follow your instinct
- 跟随自己的直觉。跟随自己的直觉当然可能失败，但如果不跟随自己的直觉什么都不会做成。

### Pieter Abbeel interview

- 算法还不擅长在长时间维度上识别 pattern , 从一天或一生中总结出什么规律。
- 算法可以识别一些pattern， 但如何使用这些pattern是另一个问题。像人会有一些认识或想法，如何利用。
- 设计一个生成 增加深度学习算法 的算法。
- 学习就要自己多多练习


## 方字就是抽象概念

任何现实生活中的色相，都是用文字描述的，文字即概念，没有人可以离开文字思考，文字及抽象。

图、音乐有类似的抽象作用。

所以建议一个AI，通过少量观察提取出共同的“模式pattern”，然后给这个pattern起个名字，这就是抽象的过程。然后拿这个pattern去套各种东西，套出新的pattern（原pattern的变形、组合、异化），进行新的抽象（起名），以此扩展。现通过与其他AI或要人类的确认（增强学习：问这是xxx吗？回答：Y/N），从而进一步增强。

## 同样的网络不同的权重

尽量使用同构的网络，这样使用不同的参数值就可以识别不同的东西。

使用多个这样的网络，类似模拟大脑的不同部分。但可以比大脑大的多。

## 实现不同功能的互通

音乐转成描述音乐的文字

图片转成描述

描述转成图

以上就是类似人类的想像。

## 常识

人们常拿重力来举例人类的常识。可实则重力是对生活观察的结果的抽象。有了抽象能力就能形成重力的概念，也就是能“想象/生成”提起重物时的图像。

## 抽象概念

抽象概念即是连接：图片可以连接到文字，文字可以连接到声音，声音可以连接到图片和文字

但文字好像更基础，思考时大部分用声音，或图片。盲人用声音，哑巴用图片。所以什么都是可以用来思考的。

连接连接连接。

一个简单的故事 这个世界每个人都想当影帝