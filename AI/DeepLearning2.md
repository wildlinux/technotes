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

REGULARIZATION: 平衡参数，让参数都差不多，不太大。脱敏，让参数对个别样本不敏感。

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

## 2.2. Optimization

### 2.2.1. Normalize

- 让输入分布正态化

### 2.2.2. Vanishing and Expoding gradients

- 对于一个很深的NN（L很大）
  - 中间所有层的总体效果可以简明理解为 W^L^，类似指数增长
    - 如果W的每个值小于1，那W^L^将接近于0，即vanishing
    - 如果W的每个值大于1，那W^L^将非常大，即expoding
- Solution
  - carefully choose initial W

### 2.2.3. Checking your derivative computation

estimate derivative: f(a+e) - f(a-e) / 2e

## 2.3. Week2 Optimization Algorithms

### 2.3.1. Mini-batch gradient descent

How to choose batch size? 

A size that best use the H/W and vectorization alogrithm while takes less time. The overall target the balance of time and accuracy. 

- Bigger size, more accurate
- Smaller size, faster
- Typical size:
    -  2^n^, n > 8 
    -  smaller than the memory size of CPU/GPU

> 每一迭代使用不同的batch size

### 2.3.2. Exponentially weighted averages

- 指数加权平均

- Bias correction
    - 对于启动初期的值进行一定的调整，避免由于启动值为0，而导致的初期值偏小的问题。

### 2.3.3. Gradient descent with momentum

- 理解：
  - 由于使用了最近几次（1/(1-beta)）的（指数加权）平均值去更新W/b，使得震荡相互抵消了，但主移动方向得到了保留和加速。
  - 加速滚落的雪球。每次速度都有提高。

### 2.3.4. Root Mean Square prop

- 用以保证每次更新都不太大。通除以均方根，大的除大的，小的除小的，结果就差不多了。
- 感觉主要是预防 b 和 W 调整时，某个调整过大，导致影响过大。通过除一个平均值，直到类似normalization的效果。

### 2.3.5. Adam optimization algorithm

### 2.3.6. Learning rate decay

## 2.4. Week3 Hyperparameter tuning, Batch Normalization and Programming Frameworks

### 2.4.1. Hyperparameter Tuning

- 搜索空间对不同量级使用同样的搜索资源。例如
  - 0.1 0.01 0.001 这三个范围内设置同样数量的采样点

### 2.4.2. Batch Normalization

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

### 2.4.3. Softmax Regression

- concept
    - Softmax is a certain type of activation function
    - it output the same shape of vector as the input
    - computation is quite simple: a^i^ = e^zi^/sum(e^zi^), which indicate the probability.

### 2.4.4. Programming Framework

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
# 3. LESSION3: Structuring Machine Learning Projects

## 3.1. WEEK1

### 3.1.1. Introduction to ML Strategy

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

- Why human-level performance?
  - human level error (estimate of Bayes error)
    - difference between human and train called Avoidable bias
    - bias reduction method should be used
        - Bigger Model
        - Training Longer/Better Optimization method: mometum, RMSprop, ADAM
        - NN arch/hyperparameters search 
  - Train error
    - difference between Train and Dev error means Variance problem, Variance reduction method should be used. 
        - More data
        - Regularization: L2, Dropout, Data argument 
        - NN arch/hyperparameters search
  - Dev error
  - Other:  
    - if your AI does not surpass human, then human can help it
      - human expert can help to figure out why your algorithm get wrong
      - human can label more data 
      - human can analyse and improve bias and variance 
    - We want trainging performance to be as close as to bayes error.
    - Then we want dev performance to be as close as to training performance 
- Surpass Human-level performance
    -  Suerpassed: decision making with structrued data
    -  Partial Surpassed: speech recoginzation, some vison recognization 
    -  Not surpassed: some nature perception issues.
- Improving your model performance

> Bias: 偏移，通过改变它来改变一条曲线的中心位置。
> Variance: 方差，通过改变它来改变一条曲线的弯曲程度。

## 3.2. WEEK2

### 3.2.1. Error Analysis

- Find out exactly what type of samples being mislabelled and count each categories being mislabeled, which will inspire you in the right direction to improve your algorithm.
- Cleaning up incorrectly labeled data
    - DL algorithm is quite robust to **random** errors, while not that robust to systematic errors.
    - train set may come from a slightly different data set, while dev/test should come from the same distribution
- Build your first system quickly, then iterate
    - set traind/dev/test set and metric
    - setup the system and iteration
    - use bias/variance/error analysis to prioritize next step. 

> 想法：给train set加点随机噪声会如何？反转部分Y的值。

> 任何一种方法采取与否，都得要数据说话。它能带来多少百分比的提升。

### Mismatched training and dev/test set

- Bias and Variance with mismatched data distributions
    - human {avodable bias} train set {variance} train-dev set {data-mismatch} dev {overfitting to dev} test set
    - to determine: bias variance or data-mismatch problem
- Addressing data mismatch
    - look inside what your dev set differs from train set
    - synthesis your your train set with dev set, be careful not to overfit to your synthesising samples. 

> train set需要尽量多的数据，就通过读书掌握阅读能力，读什么书到无所谓了；dev/test set要为最终目标设置，如果目标是读懂学术论文，那dev/test就应该都是学术论文。这是说得通的，低层神经元学到的是一些共同的文法处理的知识，可以通过大量阅读得到提高。

### Learning From multiple tasks

- Transfer Learning
    - Terminology
      - pre-training: the reused training
      - fine-tuning : re-train for the new target
    - How  
      - reserve weight and structure of several beginning layers, then replace and/or add new layers with randomized weight and retrain it.
    - When
      - pre-trained NN have a bigger ( much more training data) or more general target, and fine-tuning target is a special case or smaller problem.
      - low level features from A could be helpful to B.
- Multi-tasks Learning
    - What
        - one NN to perform more tasks
    - How
        - simple extend Y to a vector. 
    - When
        - a set of tasks benefit from shared low level features
        - each task has a similar dataset
        - can train a big NN to do well on all tasks. Small NN for multi-tasks may hurt the performance.
- Terms
    - Logistic Regressiong: label is either 0 or 1
    - Softmax: label is 0 or 1 or 2 or ...
    - Multi-task: label is 0 and 1 and 2 and 3

### End2End Deep Learning

- Concept
  - example in camera to id
  - example in voice recognizaton
  - it could reduce effort on hand-designed sub-component.
- when
  - end2end learning need far more data. It work pretty well if you have plenty of data.
- when not
  - break it down to several sub-tasks, which may have sufficient data to train on
- Whether to use end-to-end deep learning
  - Pros
    - Let data speak
    - Less hand-designing
  - Cons 
    - need huge data
    - exclude potentially useful hand-designing components.
      - hand-designing actually bring in human experience in the DNN
  - How complex to map X to Y intuitionly? 

# LESSION 4: Convolutional Neural Networks

## WEEK1：Foundations of Convolutional Neural Networks

### Computer Vision

1000x1000 picture will have a 1 million featuers. If connected with a 1000 hidden unit layer, the W will be a maxtrix with 3 million element, which could unavoidably overfitting to training set, and is costive of computation.

### Edge Detection Example

1 0 -1
1 0 -1
1 0 -1

the above filter or core, want find the area where the left is bright and right is dark while don't care about the middle.

### More Edge Detection

researchers may use other numbers as filter.

### Padding

- Cons of convolutions
  - shrink of size
  - edge element involve less computer
- Valid convolution: no pading
- Same convolution: output has the same size of input
  - p = (f -1) /2 
- How
  - pad 0 

### Strided Convolutioins

move more than on cell at a time.

(n+2padding-filter)/stride+1

### Convolution Over Volumn

- if you convolute a RGB image 6x6x3 with a filter 3x3x3, the output will be 4x4
- if you apply D filters, then the output will be 4x4xD

### One Layer of a Convolutional Network

- Pros
  - the same number of filters can be applied to any size of input, while W of a DNN has to changed according to the input dimension.
  - small number of parameters(filters) are less prone to overfitting
- Shapes
  - Input: 
    - nh^[l-1] x nw^[l-1] x nc^[l-1]^
  - Current Layer
    - Filter: f^[l]^ x f^[l]^ x nc^[l-1]^ 
    - Filter count/Channels count: nc^[l]^
    - Padding: p^[l]
    - Stride: s^[l]^
    - b: nc^[l], each filter has one bias, since one computation of a filter is one single number.
  - Output: 
    - nh^[l] = (nh^[l-1] + 2p - f^[l] ) / s^[l]^ +1
    - nw^[l] = (nw^[l-1] + 2p - f^[l] ) / s^[l]^ +1 
    - nc^[l]^, filters count
  - Volumns
    - if having m input images, then each shape will add an additional dimention.

### Simple Convolutional Network Example

- Trend of hyper parameter chooice
  - shrink size while improve channels  
- Typical Layers of a CNN
  - Convolution (CONV)
  - Pooling (POOL)
  - Full Connected (FC) 

### Pooling Layers

- why
  - reduce size, fast computation, more robust
  - computation is lightly different convolution
- How
  - Hyper parameter
    - f: filter size
    - s: stride
    - No parameter to learn
  - Max Pooling
    - pick the max number of a fxf matrix as the output
    - It actually require a big number to reserver a features, that the previous layers should produce a big number for a feature area to make it go through the max pooling layer.
  - Average Pooling
    - Less often used
    - mainly to reduce size 
  - Pooling doesn't change the channel number, same computation is applied to all channels.

> Max pooling layers像是加了一条要求在CNN中，要求前层的网络对重点特征要生成一个较大的数字。相当于明确了==考核指标==。

### CNN Example

- CONV1 POOL1 CONV2 POOL2 FC3 FC4 SOFTMAX5
- Pooling layer has no parameters to learn that it convetionly is not denote a independent layer
- Dimentions tend to reduce grandualy with layers

### Why Convolution?

- Parameter Sharing
- Connection Sparsity

### Backpropagation

- W 中的一些，经过max_pool或avg_pool，影响到cost，其影响如下：
  - Max pool
    - cost = f (w1 * 0 + w2*0 + w3*1 + w4*0)
    - Backpropagation 只调整w3
  - Average pool
    - cost = f(w0/4 + w1/4 + w2/4 + w3/4)   
    - Backpropagation第个都调整一点 
- Backpropagation再理解
  - cost = f(A^[l]^), 当A^[l]^确定时，导数值当然也是确定的。
  - dJ/dA，简单称dA，是确定的值，即该点的slope。如果A是由若干元素ai组成的，那每个元素对该值都有贡献。根据每个元素贡献，再求dai，即dA/dai，类似反推。
  - 每次求出的dA，可以理解成一个样本数据求得一个Cost，反算出一个dA。如果是一组样本，那就加上求平均就可以了。本质上还是理解为一个样本的。
  - 另外每一层都是单独计算的，都是把dA的值，往回分摊。
- 具体算一个
  - Z = W*X +b 
  - A = sigmoid(Z)
  - L = L(A, Y)
  - 当正向算过来后, Z/A/L都是已知的值
  - 函数也是已知的。L'已知，dL/dA已知，dA/dZ已知，dZ/dW1 dz/dw2 dz/db1 的函数都是已知的。
  - 反向计算过程，就是先计算dL/dA，得到的是一个值。然后(dL/dA) * (dA/dZ)又求得下一个值，再(dL/dA) * (dA/dZ) * (dZ/dW) 之类的值，以此类推。
  - 注意，这里可以把具体的参数想像为一个曲线上的点，每步得到的都是具体的数据，指的是L在这点的斜率。

## WEEK2:

### Case Studies

- Classic networks
  - LeNet-5
    - 6K parameters
  - AlexNet
    - 6M parameters 
  - VGG
    - CONV * 13 + POOL * 5 + FC * 2
    - 138M parameters
- ResNet (152 layers)
- Inception

> 把二十个Logistic Regression堆叠到一起就能输出2^20^个值。

> 不要固定连接，要可以越过某些层直接连接

- ResNets
  - How
    - Add A^[1]^ to Z^[3]^, that layer 1 now goes directly to lay3
    - similar setup is applied to every two layers 
    - it could improve performane for very deep network
  - Why
    - if W^[2]^ and b^[2]^ are zeros, given the activation function is **Relu**, then A^[3]^ = A^[1]^, that there are chances that **shortcut** between layer1 and layer3 won't hurt the performance. 

- 1x1 Convolution
  - used to change numbers of channels
  - add another layer of non-linear computation
- Inception Network Motivation
  - Instead of choosing a specific size of filter by yourself, the Inception Layers encourge you integrate all possible size of filters and let the alogrithm to decide which effect the output
    - Special technic need to used to keep the uniform of output of all filters
  - Computional Cost
    - if using 32 5x5 filters to a 28x28x192 volumn, the total multiply need to do is: 28x28x32(the output volumn)x5x5x192=120M
  - Bottleneck layer
    - 1x1 CONV layer could be used to reduce the above computional cost to 12M while hurt not the performance.
    - 28x28x192 -> 1x1x16 CONV -> 5x5x32 CONV => 28x28x32
  - Inception Network consists of layers as above
  - "Inception"  comes from the file "The Inception", means we need go deeper into the NN. In this case, it step deeper to let the net choose the filters. 
  - GoogleNet is Inception Net

> 这个有两点想法和我一致，一是要尽量使用同质的网络，通过重复来实现性能增强；二是在中间层产生一个输出，我是想给这个输出更多的用途。

### Practicl Advices for using ConvNets

- Using Open-Source Implementation
  - Github has most of the implementation of algorithms 
- Transfer Learning, like you want catagory all cat images into three type: Tigger, Misty and Neither
  - You can
    - keep the net and freeze weights, replace and train only the last layer like softmax
    - keep the early layers of net and weights, replace nad train the later layers
    - keep the net and weights as a initinalization, and retrain the whole parameters.  
  - Concretely
    - Take the whole network and parameters from Github
    - freeze the early layers by setting trainable_parameters=0
      - The more data you have the less layers you need to freeze 
    - replace and train only the last layer for your data set 
    - you can computer your training set to reach the last frozen layer, like A^[i]^, use A^[i]^ to train your trainable layers, which will save the computation cost in each iteration to compute the previous layers' activation vectors.
- Data Augmentation
  - Mirror
  - Randomly Croping
  - Color Augmentation 
- State of Computer Vison
  - the more data the less hand-engneering 
  - Tips for competition
    - Ensembling: Train several several nn and average the performance
    - Multi-crops 

### Programming Assignment


```python

def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    ### START CODE HERE ###
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well. 
    X_input = Input(input_shape)
    X = ZeroPadding2D((3,3))(X_input)
    print("X shape after padding is : " + str(X.shape))
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    print("X shape after CONV0 is : " + str(X.shape))
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    print("X shape after BN0 is : " + str(X.shape))
    X = Activation('relu')(X)
    print("X shape after Activation is : " + str(X.shape))
    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)
    print("X shape after MaxPool is : " + str(X.shape))
    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    print("X shape after Flatten is : " + str(X.shape))
    X = Dense(1, activation='sigmoid', name='fc')(X)
    print("X shape after Dense is : " + str(X.shape))

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')
    ### END CODE HERE ###
    
    return model

happyModel = HappyModel((64, 64, 3))
happyModel.compile(optimizer="Adam", loss ="binary_crossentropy", metrics = ["accuracy"])
happyModel.fit(x=X_train, y = Y_train, epochs =5, batch_size=32)
happyModel.evaluate(x=X_test, y = Y_test)

X shape after padding is : (?, 70, 70, 3)
X shape after CONV0 is : (?, 64, 64, 32)
X shape after BN0 is : (?, 64, 64, 32)
X shape after Activation is : (?, 64, 64, 32)
X shape after MaxPool is : (?, 32, 32, 32)
X shape after Flatten is : (?, ?)
X shape after Dense is : (?, 1)


happyModel.summary()
plot_model(happyModel, to_file='HappyModel.png')
SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))

```

- vanishing gradients
  - 最后一层计算出的导数值会向前分配到每一层的Weight中，在一层层传递过程中，每层都会乘以WeightMatix(??)，其中的值都是小于1的，这样到第一层时值已经很小了，那参数调整的就很小，学习的过程就很慢。

## WEEK3: Object detection

### Object Localization

- Image classification : Is this a car?
- Image classification with localization: Is this a car? where is it?
- Detection: How many cars are there and where are them?

All objects need to be labeled in training and proper loss function defined.

### Landmark Detection

Add corrodinate label to image, you can have your NN to detect landmarks.

### Object Detection

- Sliding Windows Detection
  - Crop the pictures with a size of a sliding windows and detect object in the cropped sub-pictures and repeat the process.
  - parameters: stride + windows size 
  - computaiton cost is high
  
### Convolutional Implementation of Sliding Windows

- Turning FC layers into Convolutional layers
  - it's equvilent to FC
  - it's pre-request for the following algorithm
- Convolutional Implementation of Sliding Windows
  - 直接使用大尺寸的图像输入到原来为小尺寸图像训练的CNN
  - 结果就类似于对大图像进行Crop,并对每个Crop进行CNN
  - 其中Sliding Windows的stride取决与CNN中间，如使用了几个改变图像尺度layer。如例子中有一个2x2的pool,结果就是相当于Stride是2.
  - 由上一条，可以自然推导出，对于深的CNN，POOL layers太多，那可可能会让Stride过大。
  - sliding windows的大小即是原始CNN中的输入大小。

### Bounding Box Predictions 

- YOLO: You Only Look Once
  - 把原始图片等分成若干小方格，分别对各小格进行 Object detection and Localization.
  - 只要分的足够小，就可保证每格只有一个object
  - 对于横跨多个方格的object，计算结果保证只出现在一个格式中（？？） 
  - 合适的选择中间的各层，让结果是满足要求形状的volumn。

### Intersection Over Union 

- It's a defination of 'accuracy' of object detection, to evaluate your detection algorithm.
- IoU = the size of the intersection / the size of the union, of your predicted rectangle and the precise rectangle. More general, IoU is a measure of overlap between two bounding box. 
- If IoU >=0.5 ,then conventionally, it is regarded as correct.

### Non-max Suppression

- each objects may be detected multiplely 
- each detected rectangle has its probability Pc in the output vector
- discard all rectangle whose Pc <= 0.6
- pick the rectangle with highest probability
- remove the rectangle overlappted with it, like IoU > 0.5
- pick the next highest and repeat
- if having multiple objects, you need apply this process independently on each object.

### Anchor Boxes
 
- 在Non-max Suppression中，重叠多的两个box会被删除一个
- 那万一有两个object真的重叠呢
- 就使用Anchor Boxes，可以同时返回这两个重叠的Box，而不是非得删除一个。
- 对于更多特殊情况，这种算法都处理不了：如三个物体重叠，重叠的物体形状相同等。
- Archor Boxes与检测到的Box的相似度也用IoU来计算

### YOLO , Put Everything Together

> 使用越来越复杂的结果向量，也需要越来越复杂的人工标识。

### Programming

```python

    box_classes = K.argmax(box_scores, -1) 合并最后一维
    box_class_scores = K.max(box_scores, -1)
     

    out_scores, out_boxes, out_classes = sess.run( [scores, boxes, classes] , feed_dict={yolo_model.input: image_data , K.learning_phase(): 0})
 


```

## WEEK4：Special applications: Face recognition & Neural style transfer

### Face Recognition

- Face Verification
  - Input: Image and Id/name
  - Output:  whether the image is the that of the claimed person
  - 1 to 1 problem, simple

- Face Recognization
  - Has a db of K persone
  - Input: image
  - Output: ID

- One Shot Learning
  - Similarity Function
    - d(img1, img2) = the degree of difference of the two images

- Siamese Network
  - encoding the input image 
  - Learn a NN to output a vector representation of the inputted image. 

> Siamese Network中的一个思路就是把图像，降维成一个128bit的表示，这就是一种抽象啊！就是概念的形成。然后，任何其他图像也转换成同样的表示，这两个向量的距离就表示这两个图像的相似度。

- Triplet Loss
  - Triplet: Anchor , Positive, Negative
  - Want: d(A,P) + margin <= d(A,N) , d means difference
  - Loss Function
    - L = max(d(A,P) + margin - d(A,N), 0) , if the differen is less the margin than we don't care how small is it.
    - J =sum(L^(i)^)
  - Choosing Triplets hard to learn difference 

- Face Verification and Binary Classification
  - Turn it into a binary classification: 1 for pair of pictures from same person and 0 for different persons. 

> pre-computation of known samples is a common way to save live computation.

### Neural Style Transfer

- What's Neural Stype Transfer
  - Combine layers of different CNN into a New CNN

- What are deep ConvNets learning 
- Neural Stype Transfer Cost Function
  - J = J_conent(Generated,Content) + J_style(Generated,Style)
  - randomly initilize G, and update G to minimize L using gradient descent.
- Content Cost Function
  - Choose a layer to measure the content similarity of G and C
  - the shallower the chosen layer, the higher of the similarity. The deeper layer only requires conceptual similarity, like having a dog. 
- Style Cost Function
  - Style is defined by the correlation of channels of each layer 
    - each layer will has a Nc x Nc vector to denote its correlation
    - sum of correlation of all layers is the stype of the image
    -  we want minimize "style(G) - style(S)" 
- 1D and 3D Generaliations
  - similar to 2D

### Programming

- Session
  - run一次就是把计算图run到参数对应的结点，返回值就是这个参数本身
  - generated_image = model['input']，就是把名为input的参数当前值取出来
  - VGG model只是为了计算 J_style, J_content。相应对只需要计算到那几个选定的层就可以了。 
- Why Adam update input_image
  - model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
  - print(model) tell us input is a Variable, and optimizer updates Variables.   
- L2 Distane
  - square_root(square(v1 -v2))

# LESSION 5: Sequence Models

# 4. Python

## Basic 

### for

for l in range (L):
    print (" l = " + str(l) + " of " + str(L))

loop from 0 to L-1

### Array

sub_set=super_set[1,2:4,2:4,:], shape is [1,2,2,all]

## Tensorflow

```python

sess = tf.Session()

sess.run(tf.global_variables_initializer())

sess.run(model['input'].assign(input_image))

sess.run(train_step) // or any node in the computation graph

input_image = model['input']

neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1) axis=-1 表示降一维，最后一维

```

### Placeholder and Real Input

Computation Graph is consist of ==Placeholders==, not real data set, which is used only in session.run(feed_dict)

# 5. 一些想法

## 5.1. 人物对话中学到的

### 5.1.1. Geoffrey Hinton interview

- Brain is not sympolic
- Unsupervised learning is more important
- read some paper then follow your instinct
- 跟随自己的直觉。跟随自己的直觉当然可能失败，但如果不跟随自己的直觉什么都不会做成。

### 5.1.2. Pieter Abbeel interview

- 算法还不擅长在长时间维度上识别 pattern , 从一天或一生中总结出什么规律。
- 算法可以识别一些pattern， 但如何使用这些pattern是另一个问题。像人会有一些认识或想法，如何利用。
- 设计一个生成 增加深度学习算法 的算法。
- 学习就要自己多多练习

### 5.1.3. Yoshua Bengio interview

- long term dependency 
- Joint distribution
- neural machine translation with **ATTENTION**
- Toy experients are faster to iterate and easy to manipulate. It's a nice way to research more quickly.
- Ask good questions and under the underlying principles, **WHY** is more important
- **ICLR** NIPS ICML have good papers

### 5.1.4. Yuanqing Lin interview

### 5.1.5. Andrej Karpathy interview

- Optimization writes the code, and sometimes it's better than human

### Ruslan Salakhutdinov interview

- Deep enhancement learning
  - 用已有的人为设计的多层NN,训练end2end的NN,评估新NN的性能。
- Nutual language understanding

### Yann LeCun Interview

Give back to learn. Contribute to the open source.

## 5.2. 方字就是抽象概念

任何现实生活中的色相，都是用文字描述的，文字即概念，没有人可以离开文字思考，文字及抽象。

图、音乐有类似的抽象作用。

所以建议一个AI，通过少量观察提取出共同的“模式pattern”，然后给这个pattern起个名字，这就是抽象的过程。然后拿这个pattern去套各种东西，套出新的pattern（原pattern的变形、组合、异化），进行新的抽象（起名），以此扩展。现通过与其他AI或要人类的确认（增强学习：问这是xxx吗？回答：Y/N），从而进一步增强。

## 5.3. 同样的网络不同的权重

尽量使用同构的网络，这样使用不同的参数值就可以识别不同的东西。

使用多个这样的网络，类似模拟大脑的不同部分。但可以比大脑大的多。

## 5.4. 实现不同功能的互通

音乐转成描述音乐的文字

图片转成描述

描述转成图

以上就是类似人类的想像。

## 5.5. 常识

人们常拿重力来举例人类的常识。可实则重力是对生活观察的结果的抽象。有了抽象能力就能形成重力的概念，也就是能“想象/生成”提起重物时的图像。

## 5.6. 抽象概念

抽象概念即是连接：图片可以连接到文字，文字可以连接到声音，声音可以连接到图片和文字

但文字好像更基础，思考时大部分用声音，或图片。盲人用声音，哑巴用图片。所以什么都是可以用来思考的。

连接连接连接。

概念涉及另一个重要的点就是：人这个个体与其他体会的共同、同理心。例如：我的妈妈是具体的，很容易定义；然后相对于我，你也是你的妈妈；然后就提出了妈妈这个抽象概念。这里我想说的重点是：人类是由不同的同质个体组成的，所以有共同的一些概念。如果AI是一个大个体，它也要能切换视角来理解一些概念。对于它没有的概念（如赤道人对雪的理解）它也不能真正理解。

人的所谓经验，其实就是感官的输入。AI也有各种sensor，所以经验应该给人要丰富的多。

一个简单的故事 这个世界每个人都想当影帝

## End2End

设计一种端到端的检测模型，使用已有的训练好的网络作为master。
- 识别情绪的
- 识别文字的
- 识别图片的
- 识别用户的
- ...
- 最后综合识别fakenews，重点是在end2end+enhancement+unsupervised

## Inception: we should go deeper 

- 根本是两个方向
  - 算法可以自主选择网络
    - DNN 应该是等价于CNN的，但计算量要大得多。
    - DNN 通过重复类似CNN的参数值，就可以实现CNN的效果。 
  - 算法可自动切换权重
    - 要求算法有类似输入、输出与结构或者有agent来处理输入输出
    - 像计算机一样，算法的输入输出要独立出来，与核心松耦合   

## 两个想法

输入模拟：大脑的思考好像是模拟音、视输入，如何在NN中应用这种现象呢？

新闻摘要：自动生成真假摘要用来训练识别网络。

## 概念或抽象

抽象主是降维

概念就是多个NN的共识，如CAT

一个NN生成一个概念CAT,然后用GAN再反推出图片，以及对应的概念，让另一个NN识别这个图片得到同样的概念，那就是达成共识，就可以用这个概念沟通了。

## 可解释性

用基因着色类似的方法给NN着色

## 其他 

做个一次消息验证服务：只传输，不在任何中间的任何地方存储，包括日志中
由来：一些重要信息如账号、密码等，都不应该搜集存储，只在需要的时候获取，如获取密码或CSV码

# 其他知识

## 贝叶斯公式 先验 后验

[参考文献]<https://www.cnblogs.com/hapjin/p/6664545.html>

其实研究的一直是参数theta的可信度问题，在已知信息的条件下，我们要找到在最可信的参数值，即P(theta)最大。然后用这个theta，生成预测模型（类似结果的概率密度函数）来预测后来的值。

- 理想化的方案：
  - 解出后验概率，然后用后验概率进行预测
- 我们想知道，根据已知信息，theta最有可能是什么值，即 p=max(P(theta | data))时theta的取值。
  - 根据贝叶斯公式 P(theta| data) = P(data | theta) P(theta) / P(data)
  - 解出以上方程，然后求极值就可以了。实际上很验解。于是我们就通过假定先验概率P(theta)服从特定分布，从而简化这个问题。
  - 最终简化为求似然的极值。
- 如何求极值
  - 后验分布中包含了模型参数，使用 点估计(point estimate--the MAP solution) 方法对模型参数进行估计。这方法用到了牛顿法，首先选择一个初始点，然后通过偏导数不断选择下一个点更新初始点，直到偏导数等于0，最终得到极值。当样本的特征是多维的时，就是使用Hessian矩阵来更新了。显然，这是一种不断迭代的思路。因此，该方法需要考虑“收敛性”问题---最终能不能得到最值(极值)？要迭代多少次才能收敛？收敛得快不快？能不能收敛到最优值，可以根据Hessian矩阵的正定性(负定性)来判断，收敛得快不快则需要根据实际情况具体分析了。
  - 在“点估计”近似方法中，是直接针对模型的参数进行点估计，没有牵涉到 后验分布函数，而Laplace近似方法就是对后验分布函数进行近似。前面提到，我们无法精确求得后验分布函数，那可以找一个 与 后验分布 相似的函数，用它来作为我们的后验分布函数就行了。这称为Laplace approximation.这里的Laplace近似的基本思路就是，先将后验分布表示成Log函数的形式，然后使用泰勒展开对Log函数展开到二阶，由于我们需要最大化后验分布函数，那么一阶泰勒展开项为0，而二阶展开项，则使用高斯分布来近似表示。对于高斯分布而言，需要选择一个合理的均值和方差，具体如何选择的细节就不说了。(我也不知道)