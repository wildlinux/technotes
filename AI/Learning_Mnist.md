如何让大脑关闭保护模式：让目标意识到自己的已有模型是不可靠的，进而开放大脑，迎接更新

## 人工智能

> open.163.com 麻省理工学院公开课

理由是说给别人听的 原因是驱动自己的

人类是通过视觉计算的，即看即出结果->多看正确的东西！

人类的想像：已有的图像场景的组合+常识。想像一下你提着一桶水跑过街道。重点是“先模拟图形现在，再找到感觉，并根据感觉做出判断”。

连接：连接的广度很重要，人与人的连接。

### L10 Learning Alogrism

- Human Like : Constraint
	- One Shot
	- Explaination based learning
- Bulldozere : Regularity 
	- Nearest neighbor learning
		- Feathures -> Match <-Features Database
			- 平面两个点相似性的测量：距离  角度 
			- 样本分布不均匀：正则化样本：x'=x/均方差，结果均方差将等于1
			- 机械臂
				- 学习：通过练习积累如何运行到某一点，并记录下来形成一个大表格。
				- 应用：在需要时运动在某点时就查表，确定参数。
		- Other Tips
			- 大脑有10^10个神经元
			- 小脑有10^11个个神经元 每个神经元有10^5个突触 所以10^16个连接
	- Neural nets
	- Boosting

### L11 Learning: Identification Trees, Disorder

#### Identification Tree

- Which test at the first?
	- the one with the highest homogeneous distribution 
	- 选完第一个测试，在第一轮剩下的测试中，重新选择the highest homogeneous distribution
	- 重复这个过程

- homogeneous distribution -> Disorder rate: 测试无序性
	- ... 公式 
	- 熵
- 如果测试结果是数值呢
	- 按一定值为线(threshold)把数值转为boolean
	- 可以按不同值为threshold，对同一组数据进行分类，看哪种效果好

- 和上节课的Nearest neighbor learning 比较
	- ID Tree是二分法，每次测试（某种或某几种联合）都有两种可能，可定性或需要增加一层测试。
	- Nearest neighbor是把所有测试都做了，生成一个向量，然后和预设值比，找到最近的一个，以确定为自己的类型。
	- 所以理论上ID Tree工作量将小于Nearest neighbor。

- 实践应用：将ID Tree转为行动规则

```
	- IF （TestA & TestB） then DONE!
		- ELSE IF (TEST C & TEST D) then DONE!
			- ELSE IF ...   	
```

#### L12: Learning: Neural Nets, Back Propagation

- Naive Neural
	- Synaptic weight
	- Cumulative stimulus
	- All or None 
- Model a neural 
	- Neural model
		- 输入的信息的处理：简单加权(Synaptic weight)相加(Cumulative stimulus)
			- sum= input*WEIGHT
		- 输出(All or None)：
			- 在生物学意义上讲，就是说如果sum>某值(THRESHOLD)即输出1,否则不输出（输出0）
				- output= f(sum) 
				- f()的实现: if sum > THRESHOLD then f= THRESHOLD else f=0
				- f()形状如"_-"
			- f（）又称激活函数
				- 最简单的激活函数的形状就是："_-"
				- 实际f会选择S形状的（sigmoid）函数，因为形状要近似"_-"且平滑
			- 所以Neural Model就是：output= f(input*Wi - W0)
				- 在计算中THRESHOLD同权重，故标识为W0
				- 平移"_-"中的THRESHOLD到原点位置，故写为（input*Wi - W0）
				- output与input是已知的训练或测试集，激活函数f是预先选择的
		- 学习就是得到最优的WEIGHT。如何评估WEIGHT好与坏呢? : 
			- 引入performance function P，评估结果有多好
				- 有的函数是评估结果有多坏的，被称为LOSS function。
			- P(aov, dov)=-|aov-dov|^2/2
				- aov: actual output vector, dov: desired output vector
				- search for WEIGHT to maximize P
		- 如何优化WEIGHT
			- Hill climbing + depth first search to optimize P: 高维空间方向太多，无法有效使用该算法
			- Gradient ascent
			- 多元积分 相当于直接找到山脊线往上爬
			- 求出任意weight对P的导数，延爬升的方向改变weight。
		- 为了使用Gradient Ascent需要激活函数f具有可求导的性质，即曲线平滑可导
			- 所以f会选S形状的（sigmoid）函数，因为要满足形状要近似"_-"且平滑
-  Back Propagation Algorism
	- 如果NN是一个多层的连接，可以通过Gradient Ascent先求导离P最近的一层，然后逐层往回算
	- 这样一层一层往回算即反向传播
	- 反向传播算法的核心是选择的激活函数：f=1/(1-e^-x),f对x求偏导=f(1-f)

 

- Problems
	- 1.NN是做什么的：曲线拟合：生成一个函数，将调整函数的计算输出去匹配期望的输出 
	- 2.如何将输入向量输入到NN中：Tensor解决了这个问题
	- 3.Overfitting：其本质是生成一条曲线，不断折弯以拟合预期的点，这个过程中可能会导致拐弯的地方拐的太过：弯的太厉害。即过拟合。
	- 4.调整过大会造成振荡

L13: Learning: Genetic Algorithms
## Mnist 学习笔记

每种概念都需要一个数学解释和一个直观的解释。

[学习来源](https://www.tensorflow.org/get_started/mnist/beginners)



### Mnist Expert

关于通道的很直观的解释：https://zhuanlan.zhihu.com/p/27642620

注意：三个channels的权重并不共享。 即当深度变为3后，权重也跟着扩增到了三组。



### 神经网络

神经元：有输入、有输出、有内部处理、有信号强弱

神经元即运算结点，是算法的封装。
输入+输出是神经元的总连接数

输入的数量是本神经元与上级神经元接触点的多少，重点词汇是**“接触点”**。神经元连接的不是上一级神经元，而是上一级神经元的输出（突触）。


### 线性与非线性
y=x*w+b 这就是线性，能表示成二维坐标的一条直线或者三维坐标的一个平面。卷积也是线性变换，因为是基本乘法与加法。

非线性如二维曲线（指数、正态分布）、三维的曲面。

### 卷积

https://zh.wikipedia.org/wiki/%E5%8D%B7%E7%A7%AF

卷积从滤波器中看最直观：是指两波型的重叠区域的面积。如上wiki中的图形所示，有两种理解：

- 门中的东西。A中B越多，卷积越大。
- 两个波的重叠度 

从神经网络的视角理解卷积：

- 传统神经网络使用全连接
	- 例：输入为i个（神经元或数值），输出为o个（神经元或数值），则连接数（权重）为i*o个。
- 卷积就是共享权重的部分连接：拿同一份权重矩阵扫描整个图片即共享；该权利矩阵大小小于图片，即部分。
- 

### 激活函数

数学解释：在CNN中提供非线性映射的一种方法。
直观解释：

### 卷积与图像

https://www.zhihu.com/question/39022858
要通读，每个回答都从一些很好的角度切入的问题。

其中的一个视频https://v.qq.com/x/page/o03988v6cow.html

### Q1 第一次train完的W,b如何查看呢？train的过程呢，一个数据集就train一次还是多次？

### Q2
http://www.tensorfly.cn/tfdoc/tutorials/mnist_pros.html
print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
为啥可以用.eval呢? 
- tf.reduce_mean的eval函数？

tensorflow是一张运算图
如下的语言就是构造了图中的计算结点，终点就是cross_entryopy（当然如网页中描述，前面还有若干代码来说明每个变量的来源）。我们就可以使用session来运行这张图，结果就会输出cross_entropy。
`cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))`

下面这句是我花了些时间理解的。tf内置了若干子图，或子图的连接方式，下面这个相当于生成了一个更为复杂的图。但有一点是一样的：train_step是终点，应该可以通过session来运算。

`train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)`

loss function是指导训练收敛的，并不直接为结果（模型对真实数据的预测准确性）负责。
## Q3
 x = tf.placeholder(tf.float32, [None, 784])
 是所x当成一个二维的输入，还是任意多个一维的输入
 近cross_entropy计算的解释，好像是当成多个一维的输入才对

 x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b
类似上面的问题 
	- x的形状是[100,784]
	- W的形状是[784,10]
	- matmul(x,W)的形状是[100,10]
	- b的形状是[10]
	- matmul(x,W)和b怎么能相加呢？

## 