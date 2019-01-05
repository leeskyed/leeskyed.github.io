---
layout: post
title:  "Paper | Self-Attention Recurrent Network"
date:   2019-01-05 11:32:26 +0800
categories: ComputerVision
tag: Paper
---
<!--
 * @Description: 
 * @Author: Leesky
 * @Date: 2019-01-05 11:32:26
 * @LastEditors: Leesky
 * @LastEditTime: 2019-01-05 11:52:22
 -->

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


* content
{:toc}

#### Inrto&Abstract

- 应该是第一篇用SAN和RNN结合的做Image Saliency detection的paper
- Feature maps in deep neural network generally contain different seman- tics. Existing methods often omit their characteristics that may lead to sub-optimal results. In this paper, we propose a novel end-to-end deep saliency network which could effectively utilize multi-scale feature maps according to their characteristics. Shallow layers often contain more local information, and deep layers have advan- tages in global semantics. Therefore, the network generates elaborate saliency maps by enhancing local and global information of feature maps in different layers. On one hand, local information of shallow layers is enhanced by a recurrent structure which shared convolution kernel at different time steps. On the other hand, global information of deep layers is utilized by a self-attention module, which generates different attention weights for salient objects and backgrounds thus achieve better performance. Experimental results on four widely used datasets demonstrate that our method has advantages in performance over existing algorithms.
- 深度神经网络中的特征图通常包含了不同的语义信息。现有的方法常忽略了它们的可能导致次优结果的特性。在这篇论文中，我们提出了一个end2end深度saliency网络，充分利用了multi-scale各自特性的特征图。浅层网络通常更多包含了局部信息，而深层网络则在全局语义上有优势。因此，这个网络通过提高的不同网络层中局部和全局的特征图信息来生成更好的saliency map。一方面，浅层的局部信息由在不同time steps上共享卷积核的递归模型生成。另一方面，深层的全局信息则利用了一个self-attention模块来实现，该模块为salienct object和background生成了不同的注意力权重，从而实现更好的performance。实验结果在四个广为使用的数据集上表明我们的方法比现有的算法有了很大的进步。
- 缺少有效的方法能够提高和利用好浅层的局部信息，深层的全局信息也应该被用来高亮salient区域和抑制背景干扰

#### Contribution

- 一个end2end的CNN架构，一部分是基于VGG16来收集包含了视觉反差信息的multi-scale下的特征，另一部分ARN用来生成更巧妙和鲁棒性的saliency map
- 一个叫RCL的递归结构被用来处理浅层次的特征图，通过一个共享不同时间步长的递归卷积op增强了这些特征图的局部显著性信息
- SA被用来增强全局显著性信息。这类attention通过输入的feature maps来生成attentional weights，并给显著区域更多权重来获得更准确的结果。

#### Related Work

- 早起的machine learning
- Recurrent fully convolutional network [1]
  - 为显著性检测编码了高级语义特征
  - reﬁned outputs by the same network structure at different time steps for elaborate saliency maps
- DHSnet [2]
  - 第一个基于VGG16的子网络用来生成coarse saliency map
  - 第二个叫做hierarchy recurrent convolutional nn 被用来从细节上优化coarse saliency map。
- Recurrent Attentional Networks [3]
  - 使用 spatial attention transformer 来生成robust attention feature，可以被用来refine 最后的saliency map

#### Motivation

- 在HDSnet的基础上加上RFCN和RAN的一些优点

#### Model

- Overall
  - ![model](/image/SARN/overall.png)
  - 特征图提取部分，被用来生成multi-scale的局部特征图和全局特征图 
    - VGG16, 中间的输出和最后的输出都用来生成multi-scale features（$$L_2, L_3, L_4, and~S_5$$），传进attentional recurrent network来增强multi-scale的语义信息
  - 注意力递归网络，被用来强化生成的特征图和输出saliency map
    - 一系列的deconv layers来恢复特征图的size，然后用一些卷积操作来获得更多全局信息
    - Self-attention module 抑制背景干扰
    - recurrent conv layers in ARN 增强局部信息
    - $$L_2~L_3~L_4$$被丢进RCL单元用来强化特征，$$S_5$$通过self-attention module生成attentional feature maps $$S^{att}_5$$，然后用deconv上采样，再和RCL单元得到$$L_4$$的输出concatenated，然后重复直到输出为$$S^{att}_2$$。

- 特征图提取部分

  - 前13层用来提取局部特征，最后层用来提取全局特征
  - 用了每一组里面的第一个conv layer来作为side output丢进RCL unit

- Self-attention module

  - 给feature maps不同的权重来高亮显著区域，并且抑制背景的干扰

  - 计算通过与同一层中所有的feature maps的position(?)来计算一个层中某个位置的weight

  - 把self-attention加入到ARN中

  - 1\*1\*C1的kernel用来生成维度为C1的attention features，这些特征整合所有channels上的不同信息可以用fx，gx来表示。 这里对应于Q，K，其中C1=1/8 ，减少计算（SAGAN[4]），而V用的是1\*1\*C的kernel，为了能够与attention map相乘得到output的shape为{W*H, C}，和input一样。最后可以通过一个线性变化，来得到y。公式如下

  - $$
    f(x)=W_Q*x,~~ g(x)=W_K*x,~~h(x)=W_V*x,\qquad (1)
    \\
    ~\\
    W_Q, W_K=\{1, 1, C_1\}, W_V=\{1, 1, C\}, C_1=C/8 \\~
    \\
    \beta=\frac {\exp(s)} {\sum_{i=1}^N\exp(s)},~~ s=f(x)^Tg(x)\qquad (2)
    \\
    ~\\
    o=\beta \otimes h(x)\qquad (3)\\
    ~\\
    y=\gamma o+x, ~\gamma=0\qquad (4)
    $$

  - 当逐渐学习时候gamma会学习如何给attention maps分配权重

- Recurrent convolutional layer[5]

  - ![model](/image/SARN/RCL.png)
  - 重复使用输入的特征图来获取更多的局部语义信息，随着时间的变化，RCL单元的state逐渐进化。看结构就能明白。At location (i, j) on kth feature maps,

  - $$
    y_{ijk} (t) = g(f(z _{ijk} (t)))\qquad(5)
    
    \\~\\
    
    z_{ijk} = (w_k^f )^T u^{(i,j)} (t)+(w_k^r )^T x^{(i,j)} (t −1)+b_k\qquad(6)
    $$

  - $z$作为input，$u^{(i,j)}$表示从上一层的前传输入，$x^{(i,j)}(t-1)$表示t-1的递归输入，$w_k^f~\&~w_k^r$分别表示前传输入和递归输出的权重，bk则是RCL的bias。公式(5)中f是RELU，g代表local response normalization 函数【忽略】

  - 其实这里公式还是有点迷糊…看结构图是可以看得懂的，但感觉公式不应该是这样子解释

#### Experiment

- Datasets
- metric用了F-measure score和MAE
- 这篇论文太水了吧，各种activation，initialization都不直接写名字还要引用论文…服了
- result上，部分是作者提供，部分是用自己生成，不一定能达到论文结果。outperform他们。

#### Conlusion

- 无

#### Reference
- 注：这里直接copy论文的ref.
- [1] Wang, L., Wang, L., Lu, H., Zhang, P., Xiang, R.: Saliency detection with recurrent fully convolutional networks. In: European Conference on Computer Vision, pp. 825–841 (2016)
- [2] Liu, N., Han, J.: DHSNet: Deep Hierarchical Saliency Network for Salient Object Detection. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 678–686 (2016). DOI 10.1109/CVPR.2016.80.
- [3] Kuen, J., Wang, Z., Wang, G.: Recurrent attentional networks for saliency detection. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), p. 36683677 (2016).
- [4] Zhang, H., Goodfellow, I., Metaxas, D., Odena, A.: Self-Attention Generative Adversarial Networks (2018).
- [5] Liang, M., Hu, X.: Recurrent convolutional neural network for object recognition. pp. 3367–3375. IEEE Computer Society (2015). DOI 10.1109/CVPR.2015.7298958.