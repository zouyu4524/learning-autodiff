# Learning AutoDiff

Automatic differentiation, 缩写为 AutoDiff 是机器学习中重要概念——backpropagation——的基础。

**概念厘清**: AutoDiff 并不是利用数值化方式*近似计算*梯度, 也不是诸如 Mathematica 数学软件符号化计算梯度；而是介于两者之间, 提供了一种符号化的计算过程。

## “三步走”

实现 AutoDiff 主要可以概括为三个步骤<sup>[1]</sup>

> 1. Tracing the computation to build the **computation graph**  
> 2. Implementing **Vector-Jacobian Products (VJPs)** for each primitive op  
> 3. Backprop itself  

其中包括两个重要概念：**computation graph** 以及 **VJP**。

**Computation Graph**: 如下图给出了computation graph的示例: 

<p align="center">
	<img src="https://ws1.sinaimg.cn/large/93d8f721gy1g52xwddg0pj219s0gf75f.jpg" width="300" alt="Computation Graph">
</p>

其表示的计算过程为: 

<p align="center"><img alt="$$&#10;\begin{aligned}&#10;z &amp;= \omega x + b\\&#10;y &amp;= \sigma(z)\\&#10;\mathcal{L} &amp;= \frac{1}{2} (y-t)^2 \\&#10;\mathcal{R} &amp;= \frac{1}{2} \omega^2 \\&#10;\mathcal{L}_ {\text{reg}} &amp;= \mathcal{L} + \lambda \mathcal{R}&#10;\end{aligned}&#10;$$" src="svgs/b8dba3fb159fcf25482909e43fb5a7cf.svg" align="middle" width="117.79069335pt" height="146.20866644999998pt"/></p>

**VJP**: 是多元微分链式法则的计算方式。计算<img alt="$\frac{\partial {\bf y}}{\partial {\bf x}}$" src="svgs/4fa325906023a9fd587d4783383aee28.svg" align="middle" width="15.788260949999996pt" height="30.648287999999997pt"/>, 即Jacobi矩阵, 表示如下:  

<p align="center"><img alt="$$&#10;{\bf J} = \frac{\partial {\bf y}}{\partial {\bf x}} = \left(&#10;\begin{array}{ccc}&#10;\frac{\partial y_1}{\partial x_1} &amp; \cdots &amp; \frac{\partial y_1}{\partial x_n}\\&#10;\vdots &amp; \ddots &amp; \vdots \\&#10;\frac{\partial y_m}{\partial x_1} &amp; \cdots &amp; \frac{\partial y_m}{\partial x_n}&#10;\end{array}&#10;\right),&#10;$$" src="svgs/965077606744b9ae1951c697f602ffbd.svg" align="middle" width="243.24782624999997pt" height="75.9398541pt"/></p>

有了以上两个重要概念以后, backprop中的“递推”公式如下给出:  

<p align="center"><img alt="$$&#10;\bar{{\bf z}} = \left( \frac{\partial {\bf y} }{\partial {\bf z}} \right)^\top \bar{ {\bf y} },&#10;$$" src="svgs/4a21a4847e5daa447542e41c16db020a.svg" align="middle" width="106.98720284999999pt" height="43.379419049999996pt"/></p>

其中, <img alt="$\bar{{\bf z}}$" src="svgs/55a605a5d7dc12ea8c9c715768e1276a.svg" align="middle" width="8.40178184999999pt" height="19.123288799999997pt"/> 以及 <img alt="$\bar{{\bf y}}$" src="svgs/ae966f8f005960a18836e48773a14326.svg" align="middle" width="10.239687149999991pt" height="19.123288799999997pt"/> 分别表示loss function对 <img alt="${\bf z}$" src="svgs/4c7bed0ef6238b85271cb1c6f6636cbb.svg" align="middle" width="8.40178184999999pt" height="14.611878600000017pt"/>, <img alt="${\bf y}$" src="svgs/6c8f2d192cdede4e4c5e958c56ea43aa.svg" align="middle" width="10.239687149999991pt" height="14.611878600000017pt"/>的梯度, 这个符号表示是Grosse教授在[1]中使用的符号。而<img alt="${\bf z}$" src="svgs/4c7bed0ef6238b85271cb1c6f6636cbb.svg" align="middle" width="8.40178184999999pt" height="14.611878600000017pt"/>到<img alt="${\bf y}$" src="svgs/6c8f2d192cdede4e4c5e958c56ea43aa.svg" align="middle" width="10.239687149999991pt" height="14.611878600000017pt"/>之间是fully connected关系, 即他们之间的computation graph如下图所示<sup>[3]</sup>:   

<p align="center">
	<img src="https://ws1.sinaimg.cn/large/93d8f721gy1g533niybivj2074074aal.jpg" width="128" alt="fully connected graph">
</p>

为了理解这个表达式, 我们拆开单独看<img alt="$\bar{z_ j}$" src="svgs/a419b83f70819d9c99e8610d8741d4b0.svg" align="middle" width="13.74916289999999pt" height="18.666631500000015pt"/> 的计算过程如下: 

<p align="center"><img alt="$$&#10;\bar{z_ j} = \sum_k \bar{y_k} \frac{\partial y_k}{\partial z_j}&#10;$$" src="svgs/026a34d613d94e35851820d18a6cc648.svg" align="middle" width="106.88042474999999pt" height="42.30795525pt"/></p>

将其向量化即可得到以上利用Jacobi矩阵表示的"递推"公式。

**注**: VJP的构造只是为了说明“递推”公式的计算原理, 而实际实现时并不一定需要实际构造Jacobi矩阵, 可以根据不同的运算规则特定地编写VJP计算过程。举例而言, 假设<img alt="${\bf z}\rightarrow{\bf y}$" src="svgs/49bdc762252fe4a296e73498c1375f48.svg" align="middle" width="44.21207009999999pt" height="14.611878600000017pt"/>的运算规则是element-wise的, 那么相应的VJP实际上是对角阵, 如果相应构造Jacobi矩阵则计算开销较大, 实际上可以直接用element-wise（Hadamard乘, <img alt="$\circ$" src="svgs/c0463eeb4772bfde779c20d52901d01b.svg" align="middle" width="8.219209349999991pt" height="14.611911599999981pt"/>）即可。

### Tracing, 构造computation graph

`Autograd` 实现tracing computation graph的重要原理在于设计了`Node`类, 封装（重载）了所有`numpy`的premitive op, 使其可以如同`numpy`一样的写法, 但实际内部运作机制多`numpy`一层。`numpy`的运算符输入、输出均为numpy array, 而`Autograd`封装的`Node`使得其重载后的操作符输入、输出均为`Node`类, 而该类具有如下四个属性:   

> 1. `value`, the actual value computed on a particular set of inputs  
> 2. `fun`, the primitive operation defining the node  
> 3. `args` and `kwargs`, the arguments the op was called with  
> 4. `parents`, the parent `Node`s  

相应的`Node`类操作符（以`sum`为例）的运算流程/逻辑如下图所示:

<p align="center">
	<img src="https://ws1.sinaimg.cn/large/93d8f721gy1g534by5fktj21950fk40x.jpg" width="500" alt="node computation">
</p>

具体的构造computation graph的实现是`Autograd`的核心代码, 将来可以进一步阅读。  
综上, 将`numpy`的premitive op重载以符合`Node`类的运算流程；然后计算VJP； 反向传播即可。

## 参考资料

1. [Lecture 6: Automatic Differentiation](http://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/readings/L06%20Automatic%20Differentiation.pdf)
2. [A pedagogical implementation of Autograd](https://github.com/mattjj/autodidact)
3. [CSC421/2516 Lecture 4: Backpropagation](http://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/slides/lec04.pdf) 