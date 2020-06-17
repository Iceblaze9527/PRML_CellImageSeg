﻿# 《模式识别与机器学习》课程项目报告

> 新雅62 / CDIE6 
> 2016013327 项雨桐

## 1. 摘要
随着显微成像与荧光成像技术的发展，细胞图像能够为人们提供的信息越来越丰富，例如细胞的形态、数量、分布以及基因表达量等等。通过机器学习方法对细胞图像进行图像分割，智能化地提取细胞图像中的细胞及其所包含的信息，在医疗诊断、生物学研究等领域有着重要的应用价值。本项目以生物医学领域常用的卷积神经网络架构U-Net为基础，结合Otsu阈值分割、分水岭算法等传统的机器视觉手段，尝试对提供的两类细胞图像进行实例分割，取得了良好的结果。

> 排行榜用户名：2016013327IceTorch
> 任务一准确率：0.731
> 任务二准确率：0.796

## 2. 文献调研
### 2.1 图像语义分割与实例分割[^1]
图像分割是一项经典的计算机视觉任务，而图像语义分割的目标是将图像的每个像素所属类别进行标注。图像分割任务的输入通常是一张图片，输出通常是和原图大小相同的一张类别预测标签或者概率的分割图。与实例分割不同，实例分割除了关心像素是否属于不同类别之外，还关心它们是否属于不同的对象。

图像分割被广泛应用于医学图像领域。对于我们此次细胞标注的任务，可以按需采用实例分割的手段，也可采用语义分割的手段经处理后获得不同的实例。本项目采用的是第二种方案。

### 2.2 语义分割的深度学习方法
图像分割领域的传统算法包括阈值分割等，但效果有限。由于深度学习方法、特别是卷积神经网络在其他计算机视觉领域的出色表现，因此也被用于图像分割领域，取得了不错的成效。下面介绍两种经典的图像语义分割卷积神经网络架构。

#### 2.2.1 全卷积网络（Fully Convolutional Networks, FCN）[^1]
全卷积网络是利用卷积神经网络处理图像分割任务的重要突破，其核心改进点如下：

* 用卷积层替换经典的全连接层
* 利用上采样操作重建图像
* 利用短路连接保持图像细节信息

全连接层的替换
: 常规的卷积神经网络的卷积层-全连接层并不能将输出维持在原图大小（或者计算量十分庞大），因此，全卷积网络将AlexNet等结构中全连接层的分类器去掉，全部换成卷积层，最后采用转置卷积进行上采样操作。

![FCN](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-20-at-9.53.20-AM.png)

上采样（Upsampling）
: CNN的下采样操作削弱了原图中的位置信息，因此很难得到细粒度的预测结果；而语义分割的任务希望得到的是一张和原图对齐的预测图，也即一张与原图分辨率大小相同的预测图。因此FCN采取转置卷积的上采样操作恢复图像的分辨率。


跳跃连接（Skip Connection）
: FCN的前级卷积层分辨率较高、像素的定位比较准确；后级卷积层分辨率比低，像素点的分类比较准确。因此，为了得到更加准确的分割，我们可以把前面高分辨率的特征和后面的低分辨率特征结合起来，恢复图像细节，建立更准确的分割边界。

![FCN-8s](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-20-at-12.26.53-PM.png  =800x)

#### 2.2.2 U-Net[^2]
![U Net](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-20-at-1.46.43-PM.png  =1000x)

U-Net的整体结构像一个“U”形，因此得名，最早被设计用来做细胞等医学图像的分割。其和FCN的主要区别如下：

* 对称结构
* 短路连接的特征融合部分采用的是特征叠加而非求和。

对称结构
: U-Net将FCN中的上采样层替换成了与之前下采样操作完全对称的结构，只不过将其中的池化变为了上采样操作。前面的部分被称为收缩网络（contracting network），后面的部分被称为扩张网络（expansive network）。

特征叠加
: 特征叠加是U-Net非常重要的一种改进措施。U-Net的扩张网络保留了大量的特征通道，并且是把收缩网络对应部分的特征直接拼接过去，这样就可以向高分辨率层传递图像的上下文信息。

#### 2.2.3 FCN与U-Net的性能比较
虽然FCN结构引入了上采样操作，但分割的粒度仍然不足；此外，FCN的特征提取器和上采样-跳跃连接部分是分开训练的，步骤十分繁琐。而与FCN相比，U-Net结构简单清楚，不需要将特征提取部分和上采样部分分开训练；特别是U-Net的结构是针对小数据集的细胞分割而设计的，与本项目的目标基本契合。因此，本项目采用U-Net结构完成分割任务。

## 3. 数据处理流程
项目数据处理整体的流程是：

* 原图像数据：数据格式转换 $\rightarrow$ 单通道归一化（灰度变换）$\rightarrow$ 零填充 $\rightarrow$ 数据增强
* 标签数据：（数据标注）$\rightarrow$ 数据格式转换 $\rightarrow$ 二值化 $\rightarrow$ 数据增强

下文对各处理步骤作简要介绍。

### 3.1 数据转换
项目中提供的原图像数据和标签数据都是16位图像，都需要转换为8位图像后才能使用。

为了方便训练，原细胞图像在8位图像的基础上需要再变为单通道归一化，也即进行灰度变换后使用；原先的标签数据区分了不同的细胞实例，为了适应语义分割的二分类模式，还要把像素标签进行二值转换。

特别地，对于U-Net，如果使用valid模式的卷积，输出结果大小往往与输入不同；即使是same模式的卷积，也需要输入图像的大小可以被 $2^{\text{depth}-1}$ 整除才可以保持输出大小不变，其中 $\text{depth}$ 表示网络深度。我们的数据就面临着相应问题。一种解决的方法是对原图像进行零填充，使其大小满足要求。因此为了方便，我们在U-Net中采用了same模式的卷积（尽管这与原U-Net的实现不同），并且将任务一的 $628\times 628$ 图像扩充到 $640 \times 640$ （网络深度不超过8层），将任务二的 $500\times 500$ 图像扩充到 $512 \times 512$ ，也即都做`pad = 6`的填充。

### 3.2 数据增强（Data Augmentation）
由于医学图像领域通常可用的数据量很小，因此U-Net论文作者指出网络训练需要进行大量的（excessive）数据增强；对于细胞显微图像而言，图像具有平移和旋转不变形，对灰度变换和变形也具有鲁棒性。[^2] 基于此，我们设计了如下几种数据增强方式：

#### 3.2.1 翻转（flip）
翻转操作包括水平翻转和垂直翻转，以及二者的组合，可以直接用`opencv`提供的`cv2.flip()`函数实现。

#### 3.2.2 转置（transpose）
转置操作的定义和矩阵操作相同，可以直接用`opencv`提供的`cv2.transpose()`函数实现；另一种类似的操作是基于副对角线的转置，这可以通过将原图水平+垂直翻转后再转置实现。

#### 3.2.3 旋转（rotation）
虽然细胞图像具有旋转不变性，但旋转本身需要插值操作，而且将图像变为原来的大小很困难。不过旋转角度为90度倍数的情况是特例，它们都可以通过将翻转和转置组合操作来完成；旋转180度的操作等同于水平+垂直翻转。

#### 3.2.4 变形（deformation）
在U-Net论文中，作者指出变形操作是提升模型预测性能的关键。原文的平滑变形基于$3\times 3$ 大小网格的随机位移，随机位移从10像素大小标准差的高斯分布中抽样得到，之后每个像素的位移用双三次插值计算得到。这个操作可以用Python的`elasticdeform`包实现。

### 3.3 数据标注
任务二中的标记很少，如果使用监督学习的方法就需要给数据打标签。由于技术原因，这里选用`labelme`进行数据标注。共计标注了21张图像，标注后及转换的结果在附件中。

## 4. 算法原理
### 4.1 卷积神经网络
#### 4.1.1 结构参数[^2]

![Screen Shot 2020-06-16 at 21.20.21.png](https://i.loli.net/2020/06/16/pZS42vJDHMA3tIB.png =400x)

如图所示，一个卷积神经网络的层包括卷积部分、激活部分和池化部分。在这部分中，U-Net采用的卷积核大小均为$3\times 3$，激活函数均为ReLU，池化方式均为最大池化，池化大小和上采样卷积核大小均为$2\times 2$，步长为2。最后一层的输出利用$1\times 1$卷积进行映射，用sigmoid函数激活。网络共计23个卷积层。

除此之外，U-Net的设计允许变化网络的深度，原论文的深度 `depth = 5`；此外，U-Net中每一层的特征通道数也可以变化，收缩路径中每一层的特征通道数都是前一层的两倍，初始的通道数一般为2的幂次，这里用`wf`标定，原文中 `wf = 6`；为了保持输出图像大小与输入相同，我们的填充选择了same模式。

#### 4.1.2 训练过程[^3]

卷积神经网络的训练过程与所有的前馈神经网络相同，优化目标是最小化损失函数。网络将信息从输入端前向传递到输出端，在输出端计算损失后将损失梯度反向传播到输入，如此直至网络收敛到损失达到最小。具体的算法流程如下图所示：

![Screen Shot 2020-06-16 at 21.23.23.png](https://i.loli.net/2020/06/16/Heo1unyqfTxt8Xd.png =600x)

![Screen Shot 2020-06-16 at 21.23.34.png](https://i.loli.net/2020/06/16/a2YZMRB5pNWJ3kV.png =600x)

### 4.2 损失函数[^4]
#### 4.2.1 交叉熵损失函数（ `Cross Entropy Loss`）
语义分割中的交叉熵损失函数是逐像素计算的，如下图所示：

![cross entropy](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-24-at-10.46.16-PM.png =600x)

#### 4.2.2 均衡化交叉熵损失函数 （`Balanced Cross Entropy Loss, BCE Loss`）
BCE Loss函数被设计用于正负样本不平衡的情况，定义如下：
$$\mathrm{BCE}(p, \hat{p})=-(\alpha p \log (\hat{p})+(1-\alpha)(1-p) \log (1-\hat{p}))$$

其中$\alpha$是调节正负样本比例的参数因子。

#### 4.2.3 `Focal Loss`
Focal Loss 被设计用于难易样本的平衡，定义如下：
$$\mathrm{FL}(p, \hat{p})=-\left(\alpha(1-\hat{p})^{\gamma} p \log (\hat{p})+(1-\alpha) \hat{p}^{\gamma}(1-p) \log (1-\hat{p})\right)$$

其中$\gamma$是用于调节难易样本的因子。特别地，当$\gamma = 0$时，Focal Loss函数退化为 BCE Loss。

#### 4.2.4 `Dice Loss`
Dice 系数是一种二分类准确率的判据，定义如下：
$$\mathrm{DC}=\frac{2 T P}{2 T P+F P+F N}=\frac{2|X \cap Y|}{|X|+|Y|}$$
Dice 系数可以被“软化”成一种损失函数，定义如下：
$$\operatorname{DL}(p, \hat{p})=1-\frac{2 p \hat{p}+1}{p+\hat{p}+1}$$
其中分子、分母中的常数项是为了防止上下同为0的情况。

#### 4.2.5 含有距离加权惩罚项的交叉熵损失函数[^2]
为了应对细胞图像分割中常见的细胞粘连的问题，U-Net论文作者设计了一种利用距离加权的交叉熵损失函数，定义如下：
$$E=\sum_{\mathbf{x} \in \Omega} w(\mathbf{x}) \log \left(p_{\ell(\mathbf{x})}(\mathbf{x})\right)$$其中，权重函数为：$$
w(\mathbf{x})=w_{c}(\mathbf{x})+w_{0} \cdot \exp \left(-\frac{\left(d_{1}(\mathbf{x})+d_{2}(\mathbf{x})\right)^{2}}{2 \sigma^{2}}\right)$$

细胞的分离边界由形态学计算得到，前面的$w_{c}(\mathbf{x})$是BCE Loss权重，$d_1$ 和 $d_2$ 分别是到最近和第二近的细胞的边界的距离，论文中设置$w_0 = 10$和$\sigma =5$。这样就可以强迫网络关注那些粘连细胞间小的分割边界。

### 4.3 优化器：适应性矩估计（Adaptive Moment Estimation，Adam）

Adam算法是一种收敛速度很快的优化算法。和固定学习率的随机梯度下降不同， Adam 通过计算梯度的一阶矩估计和二阶矩估计而为不同的参数设计独立的自适应性学习率，同时获得了AdaGrad算法（自适应梯度，为每一个参数保留一个学习率） 和 RMSProp 算法（均方根传播，基于权重梯度最近量级的均值为每一个参数适应性地保留学习率）的优点。算法伪码如下所示：

![cross entropy](https://i.loli.net/2019/12/26/T5SrXlw79fGY6Bp.png =500x)
### 4.4 后处理算法
U-Net得到的图像是二分类概率图，需要经过一定的后处理手法才能削弱细胞粘连问题、得到不同的细胞实例。

#### 4.4.1 大津算法（Otsu's Method）[^5] 
Otsu算法将图片分割视为像素分类问题，算法追求的是像素分类的平均误差最小，也即贝叶斯最小错误率决策；而Otsu是在像素分布正态等方差条件下的闭式最优解，也即离散化的Fisher线性判别方法（又译线性判别分析: Linear Discriminant Analysis, LDA），该方法使得各类的类内离散度最小、类间离散度最大。

Otsu算法的原理如下:

![Screen Shot 2020-06-16 at 21.07.00.png](https://i.loli.net/2020/06/16/OiH7g26eyLJC3pc.png =600x)
![Screen Shot 2020-06-16 at 21.07.09.png](https://i.loli.net/2020/06/16/gHnVGhfB5zdwio2.png =600x)

Otsu算法也可以很容易地扩展为多分类算法，具体步骤不再赘述，可以参见图像分割有关资料[^6]。

#### 4.4.2 分水岭算法（Watershed Algorithm）[^6]
分水岭（Watershed）是基于地理形态分析的图像分割算法，模仿地理结构来实现像素的分类。

在分水岭算法中，图像的灰度空间被比喻成地球表面的地理结构，每个像素的灰度值代表高度。我们考虑以下3类点集：

1. 局部灰度极小值点集：相当于孤立的山谷；
2. 最小值的分水岭（汇水盆地）点集：如果在点集中的任意一点放置一个水滴，那么这个水滴一定会沿梯度落入对应的局部极小值。
3. 分水线点集：如果在点集中的任意一点放置一个水滴，这个水滴会等可能地落入多个局部极小值点。

分水岭算法的目标是找出分水线。首先从这些局部极小值开始注水，当水平面均匀上升到一定高度时，不同极小值处的水就会溢出分水岭，这时修建一座水坝来阻止这种溢出，直至水面达到图像灰度最大值，这时的水坝形成的分水线就可以把图像不同区域分开。

#### 4.4.3 基于标记的分水岭算法[^7]
普通的分水岭算法容易因为噪声和梯度的其他局部不规则性产生过分割问题，因此像`opencv`这样的算法库采用了基于标记的分水岭算法。`opencv`官方给出了基于标记的分水岭算法完成图像分割的过程：

1. 对图像进行Otsu二值化
2. 对二值化图像进行滤波和形态学操作处理，去掉噪声
3. 利用膨胀操作找到确定的背景区域
4. 利用距离变换找到找到确定的前景区域
5. 上述两个区域的差就是不确定区域，也就是分水岭算法需要完成的分割区域
6. 获取确定的前景区域的连通域，以寻找标记
7. 在原图像上利用标记应用分水岭算法，获得最终的分割结果

#### 4.4.4 基于多分类Otsu算法改进的分水岭算法
上述的分水岭算法实际应用效果并不理想，因此我们提出了一些改进措施：

* 将原图像上的Otsu二值化变为三值化。这是基于我们的输入是预测的概率图像、概率越高的部分越有可能是细胞的事实；如果将这些概率值以Otsu方法分成三类，那么将直接得到确定的前景（概率最高）、未知区域（概率次高）和非细胞部分（概率最低），而无需通过膨胀操作得到确定的背景后再经计算得到未知区域。
* 用二值化图像替代原图应用于分水岭算法，以减小噪声的影响。

实践证明这种方法的效果更加优秀。

### 4.5 准确度评价
语义分割项目准确率的评价可以有多种定义，这里我们选择的是匹配细胞间的Jaccard系数。

#### 4.5.1 Jaccard指数
对于真实标注图片中的一个细胞R与分割结果中的一个细胞S，我们称R与S 相互匹配，当且仅当：$$|R \cap S|>0.5|R|$$其中$|\cdot|$表示细胞区域在图片中对应的像素个数。对于匹配的标注细胞R和分割结果S，Jaccard相似度计算方式如下：$$J(S, R)=\frac{|R \cap S|}{|R \cup S|}$$可以知道，Jaccard相似度取值范围在[0,1]之间，其中1代表完全匹配，0代表完全不匹配。多个细胞的Jaccard相似度取平均为最终评分。

#### 4.5.2 二值Jaccard指数作为一种粗略估计
对于二值化的语义分割，为了简便，我们可以采用Otsu二值化后的结果和标签计算二值化的Jaccard，以作为最终分割准确率的粗略估计。

## 5. 实现过程
项目代码实现的整体结构如下所示：

![PRML_PRO.jpg](https://i.loli.net/2020/06/17/Uqryw285QRSGj9f.png)

（注；这并不是一个严格的程序框图，只是为了辅助说明整个项目代码的控制流）

部分程序说明：
* `taskx.ipynb`：模型训练与验证程序
	* `DG.py`：数据集类模块，完成数据的预处理
	* `Oper.py`：
		* `run_model()`：运行（训练+验证）模型
		* `train_model()`：训练模型
		* `val_model()`：验证模型
	* `Loss.py`：自定义损失函数
		* `focal_loss()`
		* `dice_loss()`
	* `Stats.py`：数据分析模块
		* `print_data()`：打印loss和jaccard数据
* `continue.ipynb`：继续训练已保存的模型
* `output.ipynb`：输出测试集结果
	* `Output.py`：输出处理
		* `test_out()`：输出测试集结果
		* `post_processing()`：后处理程序

更详细的细节说明见程序所附`readme.md`

## 6. 实验结果与模型性能分析—任务一
### 6.0 模型baseline参数
#### U-Net结构的默认参数：
```py
in_channels=1 #输入通道为1，不调整
n_classes=2 #分类个数为2，不调整
depth=3 #网络深度，可调整
wf=6 #特征通道数，可调整
padding=True #填充选项，设置为True使得输入和输出大小相同
batch_norm=False #正则化选项，可调整
up_mode='upconv' #上采样方式，这里只采用U-Net原文的转置卷积模式
```

#### 模型训练的默认参数：
```py
epochs =  16 #世代数，可调整
pad =  6 #原图像填充大小，这里两个任务均取为6
train_ratio =  0.8 #训练集/验证集划分比例，可调整
```

#### 优化器参数：
```py
optim_name = 'Adam' #优化器名称，这里提供Adam和SGD两种选项
lr =  1e-5 #学习率，可调整
momentum =  0.99 #for SGD，暂不调整
betas = (0.9, 0.999) #默认参数，暂不调整
eps =  1e-08 #默认参数，暂不调整
weight_decay = 0 #正则化选项，可调整
```

#### 损失函数：
```py
loss_func='cross_entropy' #损失函数名称
gamma =  0 #focal_loss 参数，gamma=0即为BCE Loss
alpha =  0.75 #focal_loss 参数
```

### 6.1 模型结构选择
#### 6.1.1 参数设置
模型结构可以通过深度`depth(d)`和特征通道数`wf(w)`两个参数调节，实验中选择了8组参数（以`-`标示）进行对比：
| w\d | d2 | d3       | d4 | d5 |
|----|----|----------|----|----|
| w5 |    | --       |    |    |
| w6 | -- | baseline | -- | -- |
| w7 |    | --       | -- |    |
| w8 |    | --       |    |    |

其他参数保持默认值。

#### 6.1.2 实验结果
实验中得到的最小`train_loss`及对应世代数如下表：
| w\d(e) | d2 | d3 | d4 | d5 |
|----|----|----------|----|----|
| w5 |    | 0.078(16) |    |    |
| w6 | 0.089（16）| 0.061(16) | 0.054(16) | 0.051(16) |
| w7 |    | 0.056(15) | 0.052(14) |    |
| w8 |    | 0.056(12) |    |    |

实验中得到的最大二值化`train_jaccard`及对应世代数如下表：
| w\d(e) | d2 | d3       | d4 | d5 |
|----|----|----------|----|----|
| w5 |    | 0.696(16) |    |    |
| w6 | 0.674(16) | 0.754(16) | 0.751(16) | 0.736(16) |
| w7 |    | 0.760(15) | 0.759(12) |    |
| w8 |    | 0.761(12) |    |    |

实验中得到的最小`val_loss`及对应世代数如下表：
| w\d(e) | d2 | d3       | d4 | d5 |
|----|----|----------|----|----|
| w5 |    | 0.086(16) |    |    |
| w6 | 0.096(16) | 0.064(15) | 0.059(16) | 0.055(15) |
| w7 |    | 0.058(15) | 0.056(12) |    |
| w8 |    | 0.060(12) |    |    |

实验中得到的最大二值化`val_jaccard`及对应世代数如下表：
| w\d(e) | d2 | d3       | d4 | d5 |
|----|----|----------|----|----|
| w5 |    | 0.677(16) |    |    |
| w6 | 0.643(16) | 0.747(16) | 0.752(16) | 0.735(15) |
| w7 |    | 0.756(15) | 0.765(14) |    |
| w8 |    | 0.754(10) |    |    |

所有实验的loss变化曲线如下所示：
![str_loss.png](https://i.loli.net/2020/06/17/JdZpVnehjLaqM6o.png)
所有实验的二值jaccard变化曲线如下所示：
![str_jac.png](https://i.loli.net/2020/06/17/9PylvtwDYC6qHc1.png)

#### 6.1.3 实验分析
* 所有的模型都拟合良好，训练损失和验证损失趋势相同，没有欠拟合和过拟合的情况发生
* 所有的模型网络损失都在16个epoch后趋于平缓，因此可基本判定网络已经收敛
* 从网络损失来看，`d4w6`、`d5w6`、`d3w7`、`d4w7`、`d3w8`损失的收敛值最小（<0.06）并彼此接近，是可行的方案；但`d4w7`、`d3w8`稳定性较差
* 但从jaccard值来看，`d3w6`的jaccard值和上述模型也比较接近，`d5w6`组合的jaccard值较低
* 因此综合网络的性能、训练效率等考虑，`d4w6`为最优的网络组合，其次为`d3w7`（训练周期比`d4w6`长）

### 6.2 模型损失函数选择
#### 6.2.1 参数设置
在6.1得到`depth=4`和`wf=6`的基础上，继续测试`bce_loss`和`focal_loss`。根据`focal_loss`原论文的建议，实验中选择了8组参数（以`-`标示）进行对比：

| $\gamma$ \ $\alpha$ | 0.25 | 0.5 | 0.75 |
|-----|------|----------|------|
| 0   |      | --       | --   |
| 0.5 | --   | --       | --   |
| 1   |      | --       |      |
| 2   |      | baseline | --   |

其他参数保持默认值。

#### 6.2.2 实验结果
实验中得到的最小`train_loss`及对应世代数如下表：

| $\gamma$ \ $\alpha$ (e) | 0.25 | 0.5 | 0.75 |
|-----|------|----------|------|
| 0   |      | 0.027(16) | 0.022(16) |
| 0.5 | 0.017(16) | 0.019(15) | 0.015(16) |
| 1   |      | 0.014(15) |      |
| 2   |      | 0.007(16) | 0.006(16) |

实验中得到的最大二值化`train_jaccard`及对应世代数如下表：

| $\gamma$ \ $\alpha$ (e) | 0.25 | 0.5 | 0.75 |
|-----|------|----------|------|
| 0   |      | 0.756(15) | 0.756(16) |
| 0.5 | 0.699(12) | 0.698(15) | 0.742(15) |
| 1   |      | 0.655(11) |      |
| 2   |      | 0.633(11) | 0.647(14) |

实验中得到的最小`val_loss`及对应世代数如下表：

| $\gamma$ \ $\alpha$ (e) | 0.25 | 0.5 | 0.75 |
|-----|------|----------|------|
| 0   |      | 0.029(14) | 0.024(16) |
| 0.5 | 0.019(15) | 0.021(16) | 0.017(15) |
| 1   |      | 0.015(14) |      |
| 2   |      | 0.008(16) | 0.006(15) |

实验中得到的最大二值化`val_jaccard`及对应世代数如下表：

| $\gamma$ \ $\alpha$ (e) | 0.25 | 0.5 | 0.75 |
|-----|------|----------|------|
| 0   |      | 0.755(16) | 0.743(10) |
| 0.5 | 0.700(12) | 0.707(15) | 0.737(14) |
| 1   |      | 0.678(10) |      |
| 2   |      | 0.689(10) | 0.633(10) |

所有实验的loss变化曲线如下所示：
![lossfunc_loss.png](https://i.loli.net/2020/06/17/6QCT9A8jPVtRSbe.png)
所有实验的二值jaccard变化曲线如下所示：
![lossfunc_jac.png](https://i.loli.net/2020/06/17/mVhauj8e3JQdvsz.png)

#### 6.2.3 实验分析
* 所有的模型都拟合良好，训练损失和验证损失趋势相同，没有欠拟合和过拟合的情况发生
* 所有的模型网络损失都在16个epoch后趋于平缓，因此可基本判定网络已经收敛
* `focal_loss`和`bce_loss`计算出的损失绝对数值要小于`cross_entropy`
* 对于`focal_loss`，增大`gamma`值会显著降低损失的绝对数值；`alpha`的值对损失的影响相对较小，但也是`alpha`越大损失越小
* 但从jaccard值来看，增大`gamma`值也会显著降低网络的jaccard值，也即准确率
* 因此综合考虑，`gamma=0`，`alpha=0.75`是最优的参数组合（也即`bce_loss`）

### 6.3 模型学习率选择
#### 6.3.1 参数设置
在6.1、6.2 得到`depth=4`，`wf=6`，`alpha=0.75`，`gamma=0`的基础上，继续测试学习率参数，分别为`1e-4`和`1e-6`。

其他参数保持默认值。

#### 6.3.2 实验结果
实验中得到的数据如下表：
| lr | 1e-4 | 1e-5(baseline) | 1e-6 |
|----|------|------|------|
| train_loss | 0.057(12) | 0.022(16) | 0.314(16) |
| train_jaccard | 0.744(10) | 0.756(16) | 0.360(2) |
| val_loss | 0.057(12) | 0.024(16) | 0.316(16) |
| val_jaccard | 0.763(12) | 0.743(10) | 0.339(1) |

所有实验的loss变化曲线和二值jaccard变化曲线如下所示：

![lr_all.png](https://i.loli.net/2020/06/17/cxfEi7uQFe1tYlm.png)

#### 6.3.3 实验分析
* 学习率为`1e-4`的网络趋于发散，说明学习率过大
* 学习率为`1e-6`的网络收敛速度过慢，说明学习率过小
* 综上，学习率为`1e-5`是最佳选择

### 6.4 数据增广
#### 6.4.1 参数设置
之前的训练都没有用到数据增广。在6.1、6.2、6.3 得到`depth=4`，`wf=6`，`alpha=0.75`，`gamma=0`，`lr=1e-5`的基础上，试验不同种数据增广的方式和数据量。

其中:
* `hvflip`（只用翻转变换）、`rotdef`（只用旋转和变形）损失函数选择的是`cross_entropy`
* `all`系列是`hvflip`和`rotdef`的结合，选择的损失函数是`focal_loss`，并且使用了不同的训练集比例（0.8，0.95，0.99）
* `all_0.99`的模型还采用了转置操作增广数据，但训练结果没有参与评测环节。

其他参数保持默认值。

#### 6.4.2 实验结果

基于`cross_entropy`的实验数据如下：

| aug | null(baseline) | hvflip | rotdef
|----|------|------|------|
| train_loss | 0.054(16) | 0.053(15) | 0.058(14) | 
| train_jaccard | 0.751(16) | 0.755(15) | 0.752(16) | 
| val_loss | 0.059(16) | 0.051(9) | 0.056(14) |  
| val_jaccard | 0.752(16) | 0.772(15) | 0.760(14) | 

基于`bce_loss`的实验数据如下：

| aug | null(baseline) | all | all_0.95 | all_0.99
|----|------|------|------|------|
| train_loss | 0.022(16) | 0.021(16) | 0.020(15) | 0.018(15) |
| train_jaccard | 0.756(16) | 0.761(16) | 0.773(15) | 0.793(15) |
| val_loss | 0.024(16) | 0.020(16) | 0.021(16) | 0.020(16) |
| val_jaccard | 0.743(10) | 0.767(12) | 0.787(15) | 0.819(16) |

基于`cross_entropy`的loss变化曲线和二值jaccard变化曲线如下所示：

![aug_ce.png](https://i.loli.net/2020/06/17/mhRyt4gQK6TnAda.png)

基于`bce_loss`的loss变化曲线和二值jaccard变化曲线如下所示：

![aug_fl.png](https://i.loli.net/2020/06/17/ANlQUBfEbxS8OFa.png)

#### 6.4.3 实验分析
* 所有的模型都拟合良好，训练损失和验证损失趋势相同，没有欠拟合和过拟合的情况发生
* 所有的模型网络损失都在16个epoch后趋于平缓，因此可基本判定网络已经收敛
* `hvflip`和`rotdef`对网络性能均有正面作用；但弹性形变操作的效果并没有U-Net论文所表明的那么显著，甚至整体而言，包含旋转和弹性形变操作`rotdef`的数据增强的贡献并不如`hvflip`
* 单纯地增加训练数据量对损失的数值影响不大，但在jaccard值上的提升效果比数据增强显著（`all_0.95`只比`all`多约184张图片，而`all`的数据量是`baseline`的7倍之多，但对于jaccard值的提升效果几乎相同甚至更显著）
* 总体而言，数据量越多，网络收敛越快

### 6.5 最优模型与实际评测结果
综上，最优模型的参数为
```py
depth = 4
wf = 6
loss_func = 'focal_loss'
alpha = 0.75
gamma = 0 # bce_loss
lr = 1e-5
epochs = 16
train_ratio = 0.99
```
使用了所有可能的数据增广手段。

实际参与测评的模型是`train_ratio = 0.95`、增广手段不包括转置的模型。利用基于Otsu三分类的分水岭算法得到的最优准确率为0.73。

### 6.6 总结
#### 6.6.1 网络的拟合
* 只要学习率选择适当（本实验的最佳学习率为`1e-5`），一定结构下的U-Net都是恰好拟合的，因此也无需进行网络正则化有关实验
* U-Net网络在16个epoch下都趋于收敛，也即loss不再下降，说明网络已达该参数设置下的性能极限

#### 6.6.2 网络的结构
如果将各个模型预测的灰度图打印出来（篇幅原因不再赘述），可以看出：

* 在一定范围内，网络越深，则提取到的特征越高级，分类越准确
* 在一定范围内，特征通道数越多，则提取到的特征越丰富，分类越准确
* 但网络深度增加会极大影响网络训练效率，也容易引发过拟合
* 特征数的增加会使得网络倾向于将噪声捕捉为特征，影响网络的鲁棒性

因此网络的结构选择需要综合考虑。

#### 6.6.3 网络的损失
* `cross_entropy`和`bce_loss`、`focal_loss`损失随世代数变化的趋势相近，但`bce_loss`和`focal_loss`的绝对数值较低
* 特别地，对于本项目，`focal_loss`对于网络预测准确率有负向影响，因此我们选择的是`bce_loss`

#### 6.6.4 数据增强
* 数据增强、或者单纯增大数据量对网络性能提升都有正面效果
* 总体而言，数据量越多，网络收敛越快

#### 6.6.5 后处理算法
* 后处理算法对最终准确率影响显著，改进后的算法相对之前`opencv`的官方算法准确率提升了约9%
* 但现在的后处理算法仍存在比较严重的细胞粘连问题，影响预测准确性

### 6.7 改进措施
基于以上的实验，可以制定以下改进措施（按优先级排序）：

1. 对于网络训练已达极限、损失趋于稳定的问题，可以考虑更换损失函数、特别是U-Net论文中提及的有利于解决粘连问题的加权损失函数
2. 进一步改进后处理算法，更好地解决粘连问题
3. 进一步增大数据量，尝试更多的数据增强方法，验证弹性形变操作的有效性
4. 对于目前的损失函数，也可以尝试采用`learning rate decay`的方法，逐步减小学习率，看网络的损失是否能进一步下降
5. 对U-Net结构的进一步改进，比如改变每一层卷积核和池化的大小，设计更细粒度的特征金字塔。

## 7. 实验结果与模型性能分析—任务二
### 7.0 模型baseline参数
考虑到任务二的实验数据就是U-Net原论文所使用的数据，因此baseline的深度选择了U-Net的默认深度`depth=5`；另外，根据任务一的实验，损失函数被更换为`bce_loss`。除此之外，任务二的baseline参数和任务一完全相同。

### 7.1 模型结构选择
#### 7.1.1 参数设置
模型结构可以通过`depth`和`wf`两个参数调节。
由于任务二难度较任务一有很大提升，所以训练世代数也被作为一个变量。
由于数据量较少，只有29个，因此采用了任务一中`all`的增广方式（不包括转置），训练集比例调整为0.95。
实验中选择了4组参数（以`-`标示）进行对比：

| e\dw | d5w6 | d5w7 | d6w6
|----|----|----------|----|
| 16 | baseline |  |  |
| 32 |  | -- | -- |
| 48 |  | -- |  |

其他参数保持默认值。

#### 7.1.2 实验结果
实验中得到的最小`train_loss`及对应世代数如下表：

| e\dw(e) | d5w6 | d5w7 | d6w6
|----|----|----------|----|
| 16 | 0.162(16) |  |  |
| 32 |    | 0.082(32) | 0.108(32) |
| 48 |    | 0.056(48) |  |

实验中得到的最大二值化`train_jaccard`及对应世代数如下表：

| e\dw(e) | d5w6 | d5w7 | d6w6
|----|----|----------|----|
| 16 | 0.486(11) |  |  |
| 32 |    | 0.558(32) | 0.429(4) |
| 48 |    | 0.630(46) |  |

实验中得到的最小`val_loss`及对应世代数如下表：

| e\dw(e) | d5w6 | d5w7 | d6w6
|----|----|----------|----|
| 16 | 0.145(16) |  |  |
| 32 |    | 0.083(32) | 0.109(32) |
| 48 |    | 0.075(45) |  |

实验中得到的最大二值化`val_jaccard`及对应世代数如下表：

| e\dw(e) | d5w6 | d5w7 | d6w6
|----|----|----------|----|
| 16 | 0.480(10) |  |  |
| 32 |    | 0.619(29) | 0.531(31) |
| 48 |    | 0.741(37) |  |

所有实验的loss变化曲线和二值jaccard变化曲线如下所示：

![task2.png](https://i.loli.net/2020/06/17/5XNsmgSHFhinvI7.png)

#### 7.1.3 实验分析
* 所有的模型都拟合良好，训练损失和验证损失趋势相同，没有欠拟合和过拟合的情况发生
* 所有的模型网络损失都在趋于下降，说明训练的世代数仍然不够充分
* 在`epochs=32`时，`d5w7`的性能要明显优于`d6w6`
* 因此综合网络的性能、训练效率等考虑，`d5w7`、`epochs=48`为最优的网络组合

### 7.2 实际评测结果
综上，最优模型的参数为
```py
depth = 5
wf = 7
loss_func = 'focal_loss'
alpha = 0.75
gamma = 0 # bce_loss
lr = 1e-5
epochs = 48
train_ratio = 0.95
```
使用了翻转、旋转和转置的数据增广手段。

利用基于Otsu三分类的分水岭算法得到的最优准确率为0.79。

### 7.3 总结
#### 7.3.1 网络的拟合
* 只要学习率选择适当（本实验的最佳学习率为`1e-5`），一定结构下的U-Net都是恰好拟合的，因此也无需进行网络正则化有关实验
* 任务二的模型需要更多的世代数才能收敛到最优值

#### 7.3.2 网络的结构
* 任务二需要比任务一更为复杂的网络结构，网络的深度和特征通道数都要增加，训练的成本也随之增加

#### 7.3.3 数据增强
* 任务二的数据总共只有29张，因此必须引入大量的数据增强。实验证明数据扩充7倍后的预测效果是比较显著的

#### 7.3.4 后处理算法
* 后处理算法对最终的预测准确度仍有较大影响，但目前的算法在任务二上的表现要优于任务一

### 7.4 改进措施
 基于以上的实验，可以制定以下改进措施（按优先级排序）：

1. 进一步增加训练的世代数
2. 增加标注数据量，尝试更多的数据增强方法，验证弹性形变操作的有效性
3. `d5w6`，`epochs = 32`的实验出了一些差错，应当重新实验以确定其性能和`d5w7`的优劣
4. 进一步改进后处理算法，更好地解决粘连问题
5. 考虑更换损失函数、特别是U-Net论文中提及的有利于解决粘连问题的加权损失函数

## 8. 总结
### 8.1 基于U-Net的细胞图像语义分割实验
细胞图像的分割是医学图像处理领域的经典问题。自从深度学习在计算机视觉领域得到广泛应用后，各种深度学习进行细胞图像分割的手段层出不穷。其中，U-Net就是该领域的典型代表。基于U-Net的结构，我们对细胞图像的语义分割做了一定尝试，取得了一定成果，也收获了很多深度学习领域的知识和经验。

本项目针对不同的U-Net结构、损失函数、优化器参数等都进行了实验，并且尝试了数据增强的方法进一步提升网络的性能。经过一系列实验，我们确定了合适的U-Net参数，使得网络的性能尽可能达到了最优值；在实验的过程中，我们也更深入地了解了卷积神经网络各参数对网络性能的影响，并确定了改进方案；本项目的目标是实例分割，因此对于U-Net的语义分割图还需要在后期用传统的机器视觉方法处理，由此也可见在深度学习大行其道的今天，传统的图像算法仍有其不可磨灭的价值。

### 8.2 未来展望
由于时间和能力所限，本项目仍存在进一步提升的空间。主要有：

* 优化损失函数和优化器参数，进一步发挥网络性能
* 更加有效的后处理算法
* 进一步的数据增强
* U-Net结构的深度改进
* 集成学习

当然，针对本项目的任务，也必然存在着比U-Net更加优秀的解决方案，这些方案也值得去尝试。

### 8.3 一些反思
诚然，本项目的实施过程中难免有一些失败的尝试和遗憾，例如：

#### 8.3.1 损失函数的尝试
除了上文提到的损失函数外，我们也尝试了`dice_loss`和软化的`jaccard_loss`，以及U-Net原论文中描述的加权损失。但是使用`dice_loss`的网络性能较差；而`jaccard_loss`的代码有误、加权损失的代码在运行时会使得程序崩溃。这些都是自己算法和编程能力不足导致的。因此日后要加强相关的算法和编程训练，能够顺利地实现自定义的损失函数。

#### 8.3.2 优化器的尝试
除了Adam优化器，我们在实验中也尝试了SGD优化器，但网络的性能较差；此外，目前的优化器都是恒定学习率，在损失函数不下降的情况下，学习率衰减是有可能进一步发挥网络的性能的，因此日后可以尝试动态学习率的优化器。

#### 8.3.3 后处理算法的尝试
后处理算法的性能也是本任务的一大瓶颈。但是许多尝试，包括腐蚀、膨胀、Otsu四分类等都没有带来更好的分割效果，这也是自己缺乏图像处理领域的相关经验导致的。因此日后要加强这方面的算法的学习和理解。

#### 8.3.4 数据增强的尝试
数据增强是本项目的一个重要手段。但是在设计数据增强策略时，自己漏掉了转置系列操作，使得模型没有达到最佳状态；另外弹性形变的部分效果不显著的原因也应当进一步探究，特别是有可能出现负效果，这些都是应当设计实验进行排除的。

另外，由于时间所限，任务二自己只完成了一部分标注，也没有探究半监督的学习方法，这些都是需要日后弥补的。

#### 8.3.5 集成学习的尝试
由于时间原因，自己没有来得及对模型做集成学习（主要是考虑到模型不存在过拟合并且预测效果相近，而且存在优先级更靠前的优化手段）。没有用实验验证集成学习对模型预测性能的影响也是一大遗憾。

#### 8.3.6 神经网络的撰写
由于时间和能力所限，本项目的U-Net等部分程序来自开源代码库，并非亲自撰写。因此，日后若想实现自己的神经网络项目，还需要提升神经网络相关的编程能力。

#### 8.3.7 一些程序错误
在项目工程实现的环节也存在着一定障碍，其中继续训练代码的错误影响最为严重。继续训练代码`continue.ipynb`本身是为了防止训练中遇到GPU显存溢出或者训练世代数不够的情况而设计的，但是由于自己的失误，把一个重新初始化的模型和优化器传进了训练函数，相当于重新训练一个模型；虽然损失函数的打印输出已经明显提示了问题，但自己在项目几近结束时才意识到问题的症结，然而为时已晚，也由此导致了很多无效的实验。这些都是由于自己工程能力不足导致的，因此日后也要加强编程基本功的训练、提升对程序错误的敏感程度。

除此之外，在图像通道转换、数据格式、PyTorch Dataset类、CPU和GPU数据、显存维护等相关程序的撰写上也遇到了一些问题，这些都需要通过日常提升工程能力来解决。

### 8.4 结语
经过本次项目的训练，自己对卷积神经网络的理解更深了一层，特别是接触到了语义分割的全新领域，学习到了该领域的一些特殊方法，这对于日后的科研学习都是非常有裨益的。

此外，项目本身对工程能力也是极大的挑战。经过本次项目的训练，自己对PyTorch框架有了较深层次的理解，工程能力得到了极大提升，已经可以做到自己实现一个完整的项目。

当然，这次项目也暴露出自己在机器学习领域的一些问题，最核心的是工程实现能力：项目核心部分的神经网络代码非自己实现，初期也花费了大量时间处理程序错误上，最终的结果也受到了影响；另外，自己对深度学习领域的理解仍然十分有限，还不清楚怎样在更深层次上调整网络的结构以提升网络的性能，这些都要在日后的学习中加深理解。


## 9. 参考文献
[^1]: Jeremy Jordan. An overview of semantic image segmentation [EB/OL]. (2018-05-21) [2020-06-16]. https://www.jeremyjordan.me/semantic-segmentation/#fully_convolutional
[^2]: Ronneberger, Olaf & Fischer, Philipp & Brox, Thomas. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. 
[^3]: [美] Ian Goodfellow 等, 著. 赵申剑等, 译. 深度学习[M]. 北京: 人民邮电出版社, 2017.8. 
[^4]: Lars Nieradzik. Losses for Image Segmentation [EB/OL]. (2019-08-16) [2020-06-16]. https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
[^5]: 大津算法-维基百科，自由的百科全书 [EB/OL]. (2017-09-08) [2020-06-16]. https://zh.wikipedia.org/zh-hans/%E5%A4%A7%E6%B4%A5%E7%AE%97%E6%B3%95
[^6]: [美] 冈萨雷斯等, 著. 阮秋琦等, 译. 数字图像处理(第三版) [M]. 北京: 电子工业出版社, 2011.6
[^7]: OpenCV: Image Segmentation with Watershed Algorithm [EB/OL]. (2020-01-17) [2020-06-16]. https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html