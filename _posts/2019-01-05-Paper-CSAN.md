---
layout: post
title:  "Paper | Convolutional Self-Attention Network"
date:   2019-01-05 09:24:01 +0800
categories: ComputerVision
tag: Paper
---
<!--
 * @Description: 
 * @Author: Leesky
 * @Date: 2019-01-05 08:54:29
 * @LastEditors: Leesky
 * @LastEditTime: 2019-01-05 11:52:27
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

#### Intro&Abstract

- 作者感觉NLP挺厉害的，刚发了两篇在EMNLP，这篇也算不错的，可以得到一些启发。

- Self-attention network (SAN) has recently attracted increasing interest due to its fully parallelized computation and ﬂexibility in modeling dependencies. It can be further enhanced with multi-headed attention mechanism by allowing the model to jointly attend to information from different representation subspaces at different positions (Vaswani et al., 2017). In this work, we propose a novel convolutional self- attention network (CSAN), which offers SAN the abilities to 1) capture neighboring dependencies, and 2) model the interaction between multiple attention heads. Experimental results on WMT14 English⇒German translation task demonstrate that the proposed approach out- performs both the strong Transformer baseline and other existing works on enhancing the locality of SAN. Comparing with previous work, our model does not introduce any new parameters.

- 由于SAN的全并行计算和在模型依赖性中的灵活性，其最近吸引了很多注意力。它通过多头注意力机制让模型去获得不同表达子空间上不同位置的信息，来大大提高了SAN的性能。在这一工作中，我们提出了一个给与SAN如下能力的CSAN，1是捕捉邻居依赖性，2是模拟多注意力头之间的交互。在WMT14上的时间结果outperforms。和之前的工作相比，我们的模型不用添加新的参数。

#### Contribution

- 和neighboring information的attention，其中attention functinon能够**根据内容不同的动态权重**成为一个filter，**并且能提升特征提取的性能**
- 发掘multi-head之间的联系，其中self-attention的n-head是指将输入的channel分解成n个子空间，这样就能从不同角度来学习他们之间的联系，同时也能并行计算。单纯的concatenation会丢失机会去发掘head之间不同特征的联系
- 将好几个基于san的模型放到同一个框架加进行比较

#### Motivation

- SAN计算了输入中的全部元素，忽略了neighboring information，而CNN能提取local feature所以利用SAN模拟CNN
- Wu and He (2018) 说特征能够更好的被cpatured 通过模拟不同channels之间的依赖[1]

#### CSAN
- ![CSAN](/image/CSAN/CSAN.png)
- 1D-CSAN，其中attention weight这样算

  - 

  $$
  a^h_{ij}= \frac {\exp e^h_{ij}} {\sum^{i+m}_{t=i-m}\exp e^h_{it}}
  $$

  - 其中 m = (M-1)/2 向下取整，并且当index超出范围不pad，同理output更新如下
  - 

  $$
  y^h_i = \sum^{i+m}_{j=i-m}{a^h_{ij}(x_jW^h_V)}
  $$

  - 尽管借鉴了CNN部分思想，但是CNN使用的是global fixed parameters，忽略了信息自身语义上和句法上的丰富性。也就是说，这里的parameters是动态的。【这里是否可以借鉴到image or video上呢？】

- 2D-CSAN，模拟局部元素和周围子空间的依赖，attentive are从$$(1\times M)$$到$$(N\times M)$$，由包含的elements的数量和heads的数量组成。利用了不同head也就是不同子空间之间的weight，来找到在不同子空间内两者的关系。

  - 例如计算i-th element in h-th head and j-th element in s-th head

  - $$
    e^{hs}_{ij}=\lambda(x_iW^h_Q)(x_jW^s_K)^T
    $$

  - $$
    a^{hs}_{ij}= \frac {\exp e^{hs}_{ij}} {\sum^{h+n}_{k=h-n} \sum^{i+m}_{t=i-m}\exp e^{hk}_{it}}
    $$

  - $$
    y^h_i = \sum^{h+n}_{s=h-n} \sum^{i+m}_{j=i-m}{a^{hs}_{ij}(x_jW^s_V)}
    $$

  - 其中$$\lfloor n=(N-1)/2 \rfloor$$

#### Experiment

- 这个实验室基于NMT架构的Transformer也就是attention is all you need那个模型。为了公平比较，所有设置都一样，除了前三层的6层SAN-based encoder。
- 【1D-CSAN 11 2D-CSAN 3*11】
- 结果：outperform without additional parameters NLP就不涉及了..
- 和其他工作比较
  - 从 embedding 角度， Shaw[1]提出了一种相对位置编码的方法
  - 从 attention distribution角度，Sperber[2]和Luong[3]提出了局部高斯变量分别通过预测一个窗口大小和中心位置来修正注意力分布，Yang[4]把这两个combine变成LOCAL_SOFT。
  - 考虑hard fashion，Shen[5]把hard local scopres分配给blocks（？），Yu[6]则stack了CNN和SAN layers

#### Analysis

- Ablation Study
  - 一维下11是最好的窗口大小，再考虑另一个维度N
  - 随着N增加，translation质量下降。可能的原因是这样：越小的N，模型仍旧有能力去学习每个head之间的不同分布，而越大的N可能会假设有更多heads做出similiar contribution
- Effect of N-gram
  - N-Gram是一种基于统计语言模型的算法。它的基本思想是将文本里面的内容按照字节进行大小为N的滑动窗口操作，形成了长度是N的字节片段序列。
  - 跳过。