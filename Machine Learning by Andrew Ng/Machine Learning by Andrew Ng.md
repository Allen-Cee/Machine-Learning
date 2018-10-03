Notes by Allen Cee



## Lecture 01 机器学习简介



### 机器学习算法 Machine learning algorithms

**常见算法**

1. Supervised learning 监督学习
2. Unsupervised learning 无监督学习

**其他**

1. Reinforcement learning 强化学习
2. Recommender systems 推荐系统

### 监督学习 Supervised Learning

1. 监督学习：已知部分数据集，给出算法，预测新数据
   1. 回归问题 Regression Problem：监督学习的一种，预测连续值 continuous values（如房价、股价）[Fig 01-01]
   2. 分类问题 Classification Problem：监督学习的一种，预测离散值输出 discrete valued ouput（如肿瘤性质） [Fig 01-02] & [Fig 01-03]

* 学习算法能够处理无穷多的属性（支持向量机 Support Vector）

### 无监督学习 Unsupervised Learning

1. 无监督学习：数据集中所有数据是一样的、没有属性，通过算法，找到某种结构（如谷歌新闻分类、特定基因判定）【个人感觉和主成分分析、因子正则化比较像】
   1. 聚类 Cluster：无监督学习的一种，具体定义还不清楚，将无属性数据通过算法分出不同属性的类（如社交网络分析、市场分割、星体分类等）[Fig 01-04]

* 鸡尾酒宴会问题 Cocktail Party Problem：不同音源如何区分；有鸡尾酒算法
* Octave常用来开发程序原型，因为内置了很多学习算法，如`svd()`奇艺值分解



## Lecture 02 模型基础



### 模型表示 Model Representation

**Eg 02-01: Housing Prices** [Fig 02-01]

* 训练集 Training Set：监督学习中已知的数据集 [Fig 02-02]

**机器学习中常见表示** [Fig 02-03]

### 成本函数 Cost Function

* 确定模型参数的依据是「使假设函数预测的$h(x)$与$y$尽可能接近」，所用的函数称为成本函数 $J(\theta_0, \theta_1)$ [Fig 02-04]

* 通常用平方误差代价函数Squared Error Function，其他函数也可，平方误差代价函数对大多数回归问题效果都不错
* 区分假设函数和代价函数：一个是$x\to y$的映射，一个是$\theta \to J$的映射；图为1个参数代价函数选取最佳参数值的直观展示 [Fig 02-05] & [Fig 02-06] & [Fig 02-07] & [Fig 02-08]
* 两个参数的代价函数会形成弓形曲面，竖向值代表代价函数的值，投影到地面形成轮廓图Contour Plot/Contour Figure【类似等高线图】[Fig 02-10]

### 梯度下降算法 Gradient Descent

* 梯度下降算法：用于使代价函数$J$最小，也用于机器学习其他领域，非常常用

**梯度下降算法的思想** [Fig 02-11]

1. 选择代价函数$J$中$\theta$的初始值，一般都设为$0$
2. 不断改变$\theta$的值，直至代价函数最小

* 就像下山一样，像下降最快的方向前进，通过偏微分判断$\theta_i$的改变方向 [Fig 02-12]
* 梯度下降算法可能出现多个局部最优解，而初始值可能非常接近
* $:=$是赋值符号 [Fig 02-13]
* $\alpha$是对应的学习速率，即$\theta_i$下降的步长
* 注意：需要同时更新$\theta_0$和$\theta_1$的值
* $\alpha$过大可能导致结果无法收敛，甚至发散 [Fig 02-14]
* 因为导数会越来越小，所以$\alpha$保持不变也会使结果收敛（步长趋近于$0$，达到局部最优点）

### 线性回归 Linear Regression

**线性回归代价函数的偏微分求解** [Fig 02-15] & [Fig 02-16]

* 梯度下降算法对应的代价函数需要是弓形曲线/曲面，即凸函数Convex Function
* 批处理梯度下降算法 Batch Gradient Descent：一次性用所有训练集中的样本计算最佳参数，即$\sum$的范围是$1$到$m$【意思应该是有时候可能处理训练集的子集】



## Lecture 03 线性代数



### 矩阵和向量 Matrices and Vectors

**矩阵的行、列、维度** [Fig 03-01]

**矩阵的元素：下标先行后列** [Fig 03-02]

**向量：一列/一行矩阵【一般用列】** [Fig 03-03]

* 数学（线性代数中）索引通常从1开始，机器学习中通常从0开始，具体情况具体讨论
* 大写字母通常表示矩阵，小写字母表示数字

### 矩阵与标量的乘法

* 维度相同才能加减

### 矩阵与向量的乘法

* 向量的维度和矩阵的列数相同，向量由上至下和矩阵由左至右相乘 [Fig 03-04] & [Fig 03-05]
* 原始数据和假设函数一起计算预测值时，用矩阵乘向量更简单、更快 [Fig 03-06]

### 矩阵与矩阵的乘法

* 分别用第一个矩阵乘第二个矩阵按列拆分的向量，再合并 [Fig 03-07] & [Fig 03-08]
* 原始数据和多个假设函数一起计算预测值时，可以用矩阵乘矩阵【为调整参数比较收益提供了思路】 [Fig 03-09]

### 矩阵乘法的性质

* 不具备交换律 not commutative
* 单位矩阵 Identity Matrix：用$I$表示，行列相同，左上角到右下角是1，其余是0；满足$A\times I=I\times A=A$【注意两个$I$不是同一个，前者是$A$的列数，后者是$A$的行数】 [Fig 03-10]

### 逆运算和转置 Inverse and Transpose

* 逆矩阵的概念对应实数的倒数
* 只有行列相同的矩阵才可能有逆矩阵，即方阵；但不是所有方阵都有逆矩阵；0没有倒数，显然所有元素是零的矩阵没有逆矩阵；逆矩阵不存在的矩阵称为奇异矩阵 singular 或退化矩阵 degenerate
* 直观上可以将没有逆矩阵的矩阵想象为非常接近于0
* $A$的转置矩阵$A^T$是将原来的行从上至下依次变为从左至右的列，即$A_{ij}=A^T_{ji}$ [Fig 03-11] 



## Lecture 04 线性回归



### 多元线性回归 Multivariate Linear Regression

* 记法：$x^{(i)}_j$表示第$i$个样本中的第$j$个变量 [Fig 04-01]
* 将一个样本的多个变量写为一个向量（竖向，$x_0=1$），对应参数也写为向量，则$h_\theta(x)=\theta^Tx$ [Fig 04-02] 
* 将模型参数看做一个向量，同样的，成本函数的作用也是$n+1$维的向量 [Fig 04-03] & [Fig 04-04]

### 特征缩放 Feature Scaling

* 特征缩放：使多个特征处于相近的范围；使用梯度下降算法的时候会更快，因为更加均匀 [Fig 04-05]
* 一般缩放到$-1 \sim 1$的范围，如果是$-\frac{1}{3} \sim \frac{1}{3}$或者$-3\sim 3$，也可以接受 [Fig 04-06]
* 均值归一化 Mean Normalization：缩放特征的时候，分子减去均值，以使缩放后的值均值接近0，分母为最大值减最小值或者是标准差等等 [Fig 04-07]

### 学习速率 Learning Rate

* 通过绘出迭代步数和代价曲线的图形可以判断出迭代是否进行正确，曲线需要是减函数，进行到接近平行于$x$轴时即可以认为已经收敛了；选择阈值（自动收敛测试 Automatically Convergence Test）很难适应所有情况，最好都画图感知 [Fig 04-08]
* 不同问题需要收敛的迭代步数差距可能很大（也和选取的初始值有关系）
* 如果迭代是增函数或呈现周期性，通常需要选择更小的学习速率
* 一般选择$0.001$，$0.003$，$0.01$，$0.03$，$0.1$，$0.3$，$1$等等，隔三倍

### 特征选择 Features

* 充分考虑特征的含义，如房价特征选择面积而不是长和宽【多因子选股也要充分考虑因子背后的含义，计量上的因子乘积项也起了这样的作用，不过有时候意义很难理解】 [Fig 04-09]

### 多项式回归 Polynomial Regression

* 多项式：一个变量，多个次项【文中举了housing Price和size的关系的例子，房价在不同面积区间的表达可能很难用一个多项式函数去拟合，即使用全部数据拟合，预测效果也很差；无法仅通过size决定housing Price，size对housing price的影响应该控制其他变量不变，即不能使用现实数据，应该使用实验数据，得到的不是房价结果而是房价的实验结果比较结果，因此这里得到复杂的多项式没有意义，不应该将其他变量承载的解释力转移到函数关系式上】
* 可以将不同次项当作不同的变量进行多元线性回归处理

### 正规方程 Normal Equation

* 标准方程：用于求解参数$\theta$【应该通常是用于线性回归】
* 对于每个样本按行排列的矩阵$X$（即矩阵$X$由样本向量$x^{(i)}$的转置$(x^{(i)})^T$按行构成），结果向量$y$，参数向量$\theta$，有$\theta = (X^TX)^{-1}X^Ty$ [Fig 04-10]
* 标准方程法不需要特征缩放，也不需要梯度下降的迭代和学习速率选择；梯度下降法在特征数量很大（上百万）的时候也适用，标准方程法因为要求$X^TX$及求其逆矩阵（逆矩阵的计算量约为维度的三次方），如果特征数量太大矩阵维度会过大；一般特征数量上百或上千会选择标准方程法，上万会用梯度下降法；还有其他学习算法如分类算法和逻辑回归，不能使用标准方程法，只能使用梯度下降法



* 在正规方程中，若$X^TX$具有不可逆性noninvertibility，octave的pinv()函数同样可以计算它的逆矩阵，即可求它的伪逆，inv()函数则不行（inv()中有数值计算的概念）

1. 若$X^TX$不可逆，通常有两种情况
   1. 有多余的特征redundant features，参数间线性相关linearly dependent
   2. 特征数量太多，通常是特征数量$n\geq m$样本量（小数据样本得到大量参数值可以通过正则化regularization）



## Lecture 05 Octave



### 基本操作 Basic Operation

* 不等于`~=`，异或`xor(1, 0)`
* 抑制打印输出：句末加`;`；打印和类型化输出：`disp(sprintf('%0.10f', pi))`
* 矩阵生成：`A = [1 2; 3 4; 5 6]`，分号隔开不同行；同一行不同列可用空格隔开，也可以用逗号；步长列表生成：`v = 1:0.1:2`，步长为0.1；生成元素同一的矩阵：`C = 9*ones(2, 3); W = zeros(1, 3);`；元素随机的矩阵：`R = rand(3, 3)`，数值介于0-1之间；高斯随机/正态分布随机：`N = randn(3, 3)`；单位矩阵：`I = eye(4)`；魔方阵/幻方：`A = magic(3)`，每行每列每条对角线加和相同
* 直方图/柱状图：`w = -6 + sqrt(10)*(randn(1, 10000)); hist(w); hist(w, 100)`；注意正常显示的条数比较少
* 矩阵维度：`size(A); size(A, 1); size(A, 2)`，依次返回行数和列数；向量维度：`length(v)`
* 加载文件：`load file.dat`或者`load(file.dat)`；保存文件：`save file.dat v;`，二进制形式；`save file.txt v -ascii;`文本格式
* 查看工作空间中所有变量：`who; whos`，后者还会显示size和数据类型class；清除变量：`clear var; clear`，后者清除所有变量
* 切片：`v(2:5); A(1:6);`，行/列向量均可切片，矩阵按列由左至右；切片结果会平面化为行向量；`A(:)`是切片为列向量；查看/赋值元素：`A(3, 2); A(2, :); A(:, 3); A([1, 3], :)`，返回元素/行/列/某几个元素或行列；也可进行赋值
* 基础运算：`A*B; A.*B; A.^2; 1 ./ A`，第二组是标量点乘，第三组对标量进行运算，第四组是点除【如果实数对矩阵每个元素做除法一定要用点除，但加减乘都可以直接进行，也满足交换律，`A/2`也可以】；`log(v); abs(A); -v;  `；`sum(A); prod(A); floor(A); ceil(A);`求每列所有元素之和/积以及向下向上取整，相当于`sum(A, 1)`，可以用`sum(A, 2)`对每行操作
* 矩阵操作：`A‘`转置；`pinv(A)`逆矩阵；`[value, index] = max(v)`求最大值及其索引，可以只有value，注意`max(A)`是对每一列求最大值，返回行向量，可以用`max(A, [], 2)`来求每行最大值，也可以用`max(max(A)) or max(A(:))`来求矩阵中所有元素的最大值；`max(A, B)`，按每个元素比较得到最大的元素组成的新矩阵；`A<5`对每个元素进行`bool`化，满足为1否则为0，`[r, c] = find(A<5)`求索引，矩阵按列，返回列向量；`flipup(A); flipud(A)`向上向下翻转

### 绘制数据 Plotting Data

* 折线图：`plot(v1, v2)`，行/列向量均可；`plot(v1, v2, 'r')`，红色
* 同一张画布画多个图像，画下一个前用`hold on;`
* 添加标签等：`xlabel('time'); ylabel(value)； legend('line1', 'line2'); title('plot 01')`
* 保存图像：`print -dpng 'plot.png'`
* 关闭图像：`close`【只有close能够正常关闭，目前的电脑不能点红叉或快捷键，会出问题orz】
* 定义画布：`figure(1); plot(...)`
* 定义子图：`subplot(1, 2, 1);`，分别是行、列、第几个子图【pyplot应该是一样的，即按照顺序进行读取，所以在splt上画/处理与plt直接处理有时候是一样的】
* 刻度变换：`axis(0.5 1 -1 1)`，先后分别是x轴和y轴
* 热度图：`imagesc(A)`，用不同颜色显示不同数字；`imagesc(A), colorbar, colormap gray;`，显示颜色条，用灰度图

### 控制语法与函数

* for循环：

  ```octave
  for i=1:10,
    v(i) = 2^i;
    if i == 2,
      break;
    elseif i == 1,
      continue;
    end;
  end;
  ```

* while大致如此，continue和break照常；区别在于不用冒号用逗号，以及控制语句结束加`end;`

* 函数：`myfunction.m`

  ```octave
  function y = squareThisNumber(x) % function y 的含义是返回一个值y, 同时有一个参数x
  y = x^2
  ```

  ```octave
  function [y1, y2] = squareAndCubeThisNumber(x)
  y1 = x^2
  y2 = x^3
  ```

  调用：`[a, b] = squareAndCubeThisNumber(5)`

### 向量化 Vectorization

* 未向量化即for循环，向量化即将加和转化为向量的积 [Fig 05-01] & [Fig 05-02]



## Lecture 06 逻辑回归 Logistic Regression



### 分类 Classification

* 标记为0的类为负类negtive class；标记为1的类为正类positive class；一般红叉表示正类，蓝圈表示负类
* 分为二元分类问题binary classification problems和多元分类问题multiclass classification problems [Fig 06-01]
* 为什么不用线性回归：对异常点的处理差；回产生大于1和小于0的值，在解释上比较无意义 [Fig 06-02]

### 逻辑回归的假设函数 Hypothesis Representation

* 类似线性回归假设函数$h(x) = \theta^Tx$，稍作改变为$h(x) = g(\theta^T x)$，其中$g(z)=\frac{1}{1+e^{-z}}$即S型函数sigmoid function/logistic function [Fig 06-03]
* 这里的假设函数给出的是给定$x$和$\theta$的概率，正类

### 分类的决策边界 Decision Boundary

* 即线性函数$z = \theta^T x$；当$z\geq0$的时候，$g(z)\geq 0.5$，$h(x)$取1，分为正类

### 逻辑回归的成本函数 Cost Function

* 相比线性回归的成本函数，逻辑回归的代价函数保留了样本数量的均值化，即$J(\theta)=\frac{1}{m}cost\ function$，其中$cost(h, y)=\frac{1}{2}(h-y)^2$，但在逻辑回归里面这样会产生非凸函数non-convex function，不利于梯度下降 [Fig 06-04]
* 所做的改变是对假设函数取对数，消减指数及倒数的影响，得到逻辑回归成本函数$cost(h, y)=\begin{cases}  -\log(h(x)) \quad if \ y = 1\\ -\log(1-h(x)) \quad if\ y=0 \end{cases}$ [Fig 06-05]
* 逻辑回归的成本函数通过统计学的极大似然法the principle of maximum likelihood得到

**逻辑回归成本函数的直观理解【通过y->h(x)->cost(h, y)的连结，会出现当h接近y的时候cost接近0，背离的时候cost接近无穷大的结果】** [Fig 06-06] & [Fig 06-07] & [Fig 06-08]

* 代价函数可以优化为$cost(h, y)=-y\log(h(x))-(1-y)(1-h(x))$ [Fig 06-09] 

### 逻辑回归中梯度下降的应用

* 对代价函数求偏导，迭代即可 [Fig 06-10]
* 逻辑回归梯度下降的迭代公式和线性回归很相似，只是假设函数的内涵不同 [Fig 06-11]
* 注意应用使梯度下降结果收敛的方法，以及通过特征缩放提高梯度下降效率

### 高级优化 Advanced Optimization

* 重要的是代价函数的式子和偏导数的求解式，解决这两部分后除梯度下降还有很多优化算法pptimization algorithm可以使用，如共轭梯度法Conjugate Gradient、变尺度法BFGS和限制变尺度法L-BFGS，这些算法通常无需选择学习速率、计算速度更快，同时也更复杂 [Fig 06-12]

* 建议直接使用软件库，无需理解高级算法内涵；使用库也要比较不同库的区别，找到一个实现表现好的库

* octave实现：

  ```octave
  function [jVal, gradient] = costFunction(theta)
  jVal = ... % cost function
  gradient = zeros(n+1, 1) % vector of theta
  gradient(i) = ...
  ```

  然后使用高级优化算法`fminunc()`，该函数会自动选择学习算法（如果选择梯度下降也会自动选择学习速率）；注意该函数的theta至少是二维向量 [Fig 06-13]

### 多元分类：一对多算法 Multi-class Classification: One-VS-All Algorithm

* 一对多算法：通过创建伪训练集，将多元分类问题转化为多个二元分类问题，具体样本的分类结果取可信度最大的一个【？】 [Fig 06-14] & [Fig 06-15]