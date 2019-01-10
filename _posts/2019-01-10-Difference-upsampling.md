---
layout: post
title:  "What's the difference between Upsampling tech"
date:   2019-01-10 11:32:26 +0800
categories: ComputerVision
tag: 
---
<!--
 * @Description: 
 * @Author: Leesky
 * @Date: 2019-01-10 11:53:26
 * @LastEditors: Leesky
 * @LastEditTime: 2019-01-10 12:16:18
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

原文：https://www.quora.com/What-is-the-difference-between-Deconvolution-Upsampling-Unpooling-and-Convolutional-Sparse-Coding

翻译：https://blog.csdn.net/xiaoli_nu/article/details/79028528 

---

**Upsampling refers to any technique that**, well, upsamples your image to a higher resolution.

The easiest way is using resampling and interpolation. This is taking an input image, rescaling it to the desired size and then calculating the pixel values at each point using a interpolation method such as **bilinear interpolation**.

Unpooling is commonly used in the context of convolutional neural networks to denote reverse max pooling. Citing from this paper: Unpooling: In the convnet, the max pooling operation is non-invertible, however we can obtain an approximate inverse by **recording the locations of the maxima within each pooling region in a set of switch variables.** In the deconvnet, the unpooling operation uses these switches to place the reconstructions from the layer above into appropriate locations, preserving the structure of the stimulus.

Deconvolution in the context of convolutional neural networks is often used to denote a sort of reverse convolution, which importantly and confusingly **is not actually a proper mathematical deconvolution.** In contrast to unpooling, **using ‘deconvolution’ the upsampling of an image can be learned**. It is often used for **upsampling the output of a convnet to the original image resolution**. I wrote another answer on this topic here. Deconvolution is more appropriately also referred to as convolution with fractional strides, or transpose convolution.

Then there is proper deconvolution which reverses the effect of a convolution (Deconvolution - Wikipedia). I don’t think people actually use this in the context of convolutional neural networks.

I don’t know much about convolutional sparse coding but it appears to me from glancing at a few papers, that those approaches use of the former kind of ‘deconvolution’, i.e. tranpose convolution, to allow you to go from a sparse image representation obtained using convnets, back to the original image resolution. (Happy to be corrected on this.)

---

上采样是指将图像上采样到更高分辨率的任何技术。
最简单的方法是使用重新采样和插值。即取原始图像输入，将其重新缩放到所需的大小，然后使用插值方法（如双线性插值）计算每个点处的像素值。

在CNN上下文中，上池化通常指代最大池化的逆过程。在CNN中，最大池化操作是不可逆的，但是我们可以通过使用一组转换变量记录每个池化区域内最大值的位置来获得一个近似的逆操作结果。在反卷积（网络）中，上池化操作使用这些转换变量从前一层输入中安放这些复原物到（当前层）合适的位置，从而一定程度上保护了原有结构。

在CNN上下文中，反卷积通常用于指代卷积的逆过程，而非数学意义上真正的反卷积，这一点很重要，也很令人困惑。相比上池化，使用反卷积进行图像的上采样是可以被学习的。反卷积常被用于对CNN的输出进行上采样至原始图像分辨率。我在这里写下关于这个问题的另一个答案：

反卷积常被认为是空洞卷积或者转置卷积 ，这也更为恰当一些。

对于一个卷积，存在一个合适的反卷积反转/消除它的影响/作用（反卷积 - 维基百科）。我不认为人们实际上在CNN上下文中使用这个（定义）。

我对卷积稀疏编码知之甚少，但在浏览一些论文的过程中我发现，这些论文使用了前一种“反卷积”，即转置卷积，使图像从卷积生成的稀疏图像表示回到原始图像分辨率。 （很高兴这一点能够被纠正。）

---


再补几个从知乎看到的答案，言简意赅。[4]
- 上采样就是把[W,H]大小的feature map $F_{W,H}$扩大为[nW,nH]尺寸大小的${F}_{nW,nH}$，其中n为上采样倍数。那么可以很容易的想到我们可以在扩大的feature map ${F}$上每隔n个位置填补原F中对应位置的值。但是剩余的那些位置怎么办呢？deconv操作是把剩余位置填0，然后这个大feature map过一个conv。扩大+填0+conv = deconv操作。插值上采样类似，扩大+插值=插值上采样操作。还有一个unpooling操作，如果是max unpooling，那么在接受[W,H]大小的feature map之外还需要接收一个pooling的index，表示F[w,h]在${F}$中的对应位置。一般max unpooling需要和max pooling对应。 max pooling+max unpooling等价于在F上筛一遍，只保留pooling window中max位置的值。

- Upsampling是上采样的过程，caffe中实现的deconvolution是upsampling的一种方式，源码来看的话，用的是bilinear。

- 实际使用过程中，会把deconv层的卷积核设置成为双线性插值，学习率设置成为0。因为很多论文表明，学习率变化与否，对于性能没有差距。


---
**CAFFE detailed descriptions**

Convolve the input with a bank of learned filters, and (optionally) add biases, treating filters and convolution parameters in the opposite sense as ConvolutionLayer.ConvolutionLayer computes each output value by dotting an input window with a filter; DeconvolutionLayer multiplies each input value by a filter elementwise, and sums over the resulting output windows. In other words, **DeconvolutionLayer is ConvolutionLayer with the forward and backward passes reversed. DeconvolutionLayer reuses ConvolutionParameter for its parameters, but they take the opposite sense as in ConvolutionLayer (so padding is removed from the output rather than added to the input, and stride results in upsampling rather than downsampling)**.

---

太懒了，直接copy过来。

简而言之，上采样是一种过程，而上池化、反卷积等是一种技术。通过这种技术来实现上采样。其中，具体提出和实现可以看一下[1]. 

**但是，tensorflow backend 的 deconv 的计算真的反人类，比CAFFE的麻烦一百倍。**

昨天看了一篇全卷积做salient object detection的，觉得自己对卷积和全卷积网络还是很浅，补一补[2]和[3].

---

[1] Learning Deconvolution Network for Semantic Segmentation

[2] Fully Convolutional Networks for Semantic Segmentation

[3] Visualizing and Understanding Convolutional Networks

[4] https://www.zhihu.com/question/63890195/answer/214228245

[5] http://caffe.berkeleyvision.org/doxygen/classcaffe_1_1DeconvolutionLayer.html