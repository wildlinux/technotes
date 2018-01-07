当前机器（深度）学习的核心的
- 利用可微编程进行优化（收敛）
- 利用反向传播增加深度及收敛

# 1. Courera Machine-learning

## 1.1. Week

- Supervised Learning: give machine a labled dataset to find the relationship between the data and the labelss
	- Regression
		- map input variables to some continuous function 
	- Classification
		- predict results in a discrete output
	- 只强调连续、离散性
- Unsupervised Learning : 
	- give machine a unlabled dataset to let it find out the structure of the data
	- With unsupervised learning there is no feedback based on the prediction results.
	- Clustering（聚合） and Cocktail Party Algorithm(分解)


## 1.2. Week

### 1.2.1. Advanced Optimizaion

根据前面学过的内容，一般的梯度下降需要我们自己选合适的alpha，选定迭代次数，收敛的结果还不一定保证到最优。
Octave提供了一些使用优化的方法来达到theta的快速收敛。

- 如每次迭代使用不的alpha...具体的实现见课程了
- 我们需要将待定的J(theta)，及J(theta)对theta_i的偏微分告诉优化算法，其他的就由优化算法自己处理了。

#### 1.2.1.1. Solving the problem of Overfitting

- Overfitting
	- underfit
	- fit
	- overfit: high variance

- Addressing overfitting
	- Reduce number of features
		- Manually select
		- Model selection algorithm
	- Regularization
		- J(theta)=J(theta)+1000xtheta1^2+1000xtheta2^2 
		- to keep the theta as small as possible
		- too small may elimate the effect of (thetai x Xi) 

### 1.2.2. ex

- 小心 X(1+2)不会解释为乘法，而是引用X中的元素
- 注意 ./ 与 / 的差别
- Regularition 
	- 注意 J（theta）及求导计算时均没有对theta0的penalty. 

## 1.3. WeekNeural Network

NN使用

- **线性组合+激活函数**来定义每一层，我感觉是利用的激活函数指数处理的效果，达到类似高阶多项式的功能，但更灵活高效。
- 每一层都是对上层输出的再计算，每一层都取样不同的特征。课程中手写识别的视频非常直观，类似于上层函数调用底层函数的效果，形成的知识的复用。

本节中讲的激活函数还是Logistic regression中的。

### 1.3.1. max (x)

- max (x, [], dim)
- [w, iw] = max (x) 
- max (x, y)

Find maximum values in the array x ， 亦可返回索引iw; 对二维矩阵默认返回一行，如果想返回列就需要设定dim=2，同时中间加一个空矩阵参数[]，我猜是实现时为了区别参数而采用

### 1.3.2. find

index_zero = find( pred1 == 10);
找出pred1中所有数据为10的元素的下标。

pred1( index_zero, 1) = 0 ;
然后将这些元素的值设置为0.

## 1.4. Week

![](./DR/backpropogation.png)

- error of cost is: y - a ,对最后一层而言
- 可以方便计算前一层的error
	- 利用的激活函数的数据特征，
- 这个error就是激活函数偏导数

- octave 矩阵操作
- “;”分行，“，”或“ ”分列
- X = [1 2; 3 4 ; 5 6]
- X = [1,2; 3,4; 5,6]
- X(2:end,:) 去除第一行


## 1.5. Week


### 1.5.1. Model Selection

数据集分为：Training Set, Cross Validition Set, Test Sec

Training Set 用来得到theta；Cross Validtaion Set用来选择lambda、多项式的阶数等决定模型的参数。

这个确定模型的过程就是Model Selection。

如果所有选择都通过Training Set得到，就容易产生overfitting的问题。

### 1.5.2. Bias vs Variance
以只有一个输入变量为例,当然本质上x^p可以当成其他输入。

h = theta0 + (theta1\*x + theta2\*x^2 + theta3\*x^4....) 

theta0对h影响太大，就是underfitting，就是Bias问题，结果就是预测结果曲线太平，不能反映变化的式样；
（theta1\*x + theta2\*x^2 + theta3\*x^4....）对h影响太大，就是overfitting，就是Variance问题，就是预测结果曲线太曲，genelization不够好。

### 1.5.3. Choose Regulariztion

lambda是加在CostFunction里，惩罚theta1-n的。

lambda太小，惩罚不够，产生Overfitting；
lambda太大，惩罚守重，首先underfitting。

### 1.5.4. Learning Curve

是数据集大小的函数。随着数据集变大，Jtraining, Jcv和Jtest值。

随着测试集变大：

- 当Jcv和Jtraining靠近到一起时
	- 如果error较小则好
	- 如果error较大，说明有high bias问题，此时太增加测试集也没有用，因为theta0影响太大，后面的多项式再调整也没用。
- 当Jcv和Jtraining中间有gap时：
	- High Variance问题，增大测试集，理优的THETA，能改善该结果。

**编程时的小细节**：使用CostFunction学习THETA时，lambda是有值的；计算Jcv,Jtraining,Jtest时，lambda需要为0，因为此时只有计算error,不需要惩罚theta1-n。	 	 
### 1.5.5. Summary: What to do to improve performance

- More training data: fix high variance
- Few features: fix high variance
- More features: fix high bias
- Polynomial: as above
- higher lambda: fix high variance
- lower lambda: fix high bias


## 1.6. NN

- large NN
	-	 
- small NN
	- computitonal efficent
	- underfitting	 

## 1.7. Eror Analysis

Evidence driving deision making instead of 'gut feeling'

single raw number evidene like Jcv make it easy to see the improvement

Mannually analyse the training examples where your algorism poorly works on. 

Try any new ideas or tools to get the number evidence to determine to use it or not.

## 1.8. Using Large Data Set

A high performance algorism should meet:

1. Check if a human experts can predicy y with given features 
2. and a model with sufficient x which eliminate high bias.
2. Larget training set, which will eliminate high variance.


## 1.9. Week

### 1.9.1. Support Vector Machine, SVM

Sometimes is called Large margin classifier, since it provide a margin between positive and negtive examples.

Change the cost function of logistic regression

### 1.9.2. Math behind SVM
u=[u1,u2] v=[v1,v2]
u'v=p\*||u||=p\*(u1^2 + u2^2 ) ^(1/2)

- ||u||是向量u的长度
其中p是v在u上的投影；当两个向量夹角小于90度（同向）时p为正，反之为负；
*结论*：u与v夹角越小（相似），p越大，u'v也就越大。


根据SVM Cost function的定义

- 当y=1时，我们希望theta与x越相似越好（theta'x>>1）
- 当y=0时，我们希望theta与x越不像越好（theta'x<<-1）

这就是选theta的直观依据。

### 1.9.3. Kernels

1.Put several Landmark in the coordinate system。
2.Similarity function or Kernal function, caculate the distance between the sample data and the Landmark

首先选选择合适的landmark/kernel点，这些点就是样本应该聚拢的点；
其次要选择Kernel Function，这些函数描述了离开landmark点的距离会造成多大的衰减。
离landmark近，结果应该是1;离landmark远结果就应该是0;Kernel Function就是描述距离与结果的关系。
Gussian Kernel:

- 使用高斯函数作为Similarity Function, f,结果就是：
	- f~=1 当x与landmark相似时
	- f~=0 当x与landmark不同时
	- 高斯函数有一参数，决定了山的半径与坡度，其值与半径成正比。即与landmark离多近可以叫相似，多远后就是完全不同。

### 1.9.4. Support Vector Machines

Kernel

- 一般用Gussian核
- 每个样本本身就是一个landmark
- 对每个样本(x,y),新的feature是每个样本与其他样本的“相似度”，所以新features是m个。

Choose SVM parameters C:

- Large C : high variance,Low bias 
- Small C: lower variance,higher bias

### 1.9.5. Excise

- 选定C和sigma
- 训练得theta
- 交叉验证
- 重复以上步骤，以得到最优的交叉验证结果

```
v=[0.01,0.03,0.1,0.3,1,3,10,30];
len=length(v);

for i=1:len
   for j=1:len
      C_temp = v(i);
      sigma_temp = v(j);
      model= svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp));
      predictions = svmPredict(model, Xval);
      err_temp = mean(double(predictions ~= yval));
      
      if ( i == 1 & j ==1)
         C = C_temp;
         sigma = sigma_temp;
         err = err_temp;
      elseif (err_temp < err)
         C = C_temp;
         sigma = sigma_temp;
         err = err_temp;
      endif
   endfor
endfor

``` 

## 1.10. Week

### 1.10.1. Unsupervied Learning

- Unlabeled data set
- aim to find the structure of the data set
- clustering alogrism and 

### 1.10.2. Clustering: K-Means Alogrism

- randomly pick centroid
- assigned xi to neareast centroid
- move centroid to means of x assigend to it
- repeat this process until centroid stop moving

#### 1.10.3. Optimization Objectives

C^\(i) is the index of centroid, x^i is assigned to.  

Cost function J:

\\[
sum(||x^i - u_\ci || ^2)/m
\\]

### Dimensionality Reduction
### PCA

#### mean normalization 

replace xi with xi - mean(x)

#### scale

scale difference features to comparable values

#### how to get u

sigma = X' * X /m 

[U,S,V] = svd (sigma)   the firest K columns is the u you need

Uk = U (:,1:k); n.k
z = Uk' * x ; k.n * n.1 = k.1  
Z = X * Uk' ;  % m.n * n.k = m.k

注意：这里说到vector都是一列
但样本集：X 中一个样本是一行啊！！

#### how to choose k

- retain 95%-99% of variance

- variance = mean(sum((xi - xi-variance)^2 )) / mean(sum(xi ^2 )) 

= 1 - sum^1-k ( sii)/ sum^1-n (sii)

- find the minium k : sum^1-k ( sii)/ sum^1-n (sii) >= 0.99, which is the minimum k retain 99% of the variance.

#### Advice for applying PCA

## Week 9 Anomaly Detection

