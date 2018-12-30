---
layout: post
title:  "Deep learning with Python Part 1"
date:   2018-12-30 16:11:01 +0800
categories: DeepLearning
tag: Books
---

* content
{:toc}


## Part 1 Fundamentals of deep learning
<br><br>
主要做摘要、记录用。后续会进行翻译以及整合。
<br><br>
#### Chapter 1 What is deep learning

##### 1.1 AI > ML > DL

- 从 rules + data -> answers 到 data + answers -> rules
- representation = a different way to look at data -> represent data or encode data
  for example: RGB format and HSV format
  Learning, in the context of machine learning, describes an automatic search process for better representations.
- what machine learning is, technically: searching for useful representa- tions of some input data, within a predefined space of possibilities, using guidance from a feedback signal.
- what deep learning is, technically: a multistage way to learn data representa- tions.
  - The fundamental trick in deep learning is to use this score as a feedback signal to adjust the value of the weights a little, in a direction that will lower the loss score for the current example.

##### 1.2 a brief history of machine learning

- Probabilitistic modeling :
  - Naive Bayes algorithm -> assuming that the features in the input data are all independent (a strong, or “naive” assumption)
  - logistic regression -> a classification algorithm

- Kernel methods :
  - SVM: aim at solving classification problems by finding good decision boundaries (see figure 1.10) between two sets of points belonging to two different categories.
    - The data is mapped to a new high-dimensional representation where the decision boundary can be expressed as a hyperplane (if the data was two- dimensional, as in figure 1.10, a hyperplane would be a straight line). 
    - A good decision boundary (a separation hyperplane) is computed by trying to maximize the distance between the hyperplane and the closest data points from each class, a step called maximizing the margin. This allows the boundary to generalize well to new samples outside of the training dataset.
    - hard to scale to large datasets and not good for perceptual problems
  - kernel fuction: a computationally tractable operation that maps any two points in your initial space to the distance between these points in your target representation space, completely bypassing the explicit computation of the new rep- resentation.
- Decision trees, random forests, and gradient boosting machines
  - Decision trees are flowchart-like structures that let you classify input data points or pre- dict output values given inputs
  - Random Forest algorithm introduced a robust, practical take on decision-tree learning that involves building a large number of specialized decision trees and then ensembling their outputs.
  - A gradient boosting machine, much like a random forest, is a machine-learning technique based on ensembling weak prediction models, generally decision trees.
- deep learning automates feature engineering in a machine-learning workflow
- In practice, there are fast-diminishing returns to successive applica- tions of shallow-learning methods, because the optimal first representation layer in a three- layer model isn’t the optimal first layer in a one-layer or two-layer model.(不解)
- incremental, layer-by-layer way in which increasingly complex representations are developed, and the fact that these intermediate incremental representations are learned jointly,
- shallow problem XGBT ; perceptual problem Keras

##### 1.3 Why now Why deep learning.

- Hardware, Datasets and benchmarks, Algorithmic advances
<br><br>
#### Chapter 2 the mathematical building blocks of neural networks

##### 2.1 A first look at a neural network

##### 2.2 Data representations for neural networks

- Vector data
- Timeseries data or sequence data
- Image data: Images typically have three dimensions: height, width, and color depth. Although grayscale images (like our MNIST digits) have only a single color channel and could thus be stored in 2D tensors, by convention image tensors are always 3D, with a one- dimensional color channel for grayscale images.
- Video data

##### 2.3 The gears of neural networks: tensor operation

- Element-wise operation: operations that are applied independently to each entry in the tensors being considered. This means these operations are highly amenable to massively parallel implementations.
- Broadcasting: the smaller tensor will be broadcasted to match the shape of larger tensor.
  - Axes (called broadcast axes) are added to the smaller tensor to match the ndim of the larger tensor.
  - The smaller tensor is repeated alongside these new axes to match the full shape of the larger tensor.
- Tensor dot
- Tensor reshaping
- Geometric interpretation of tensor operations

##### 2.4 The engine of neural networks: gradient-based optimization

- derivative
- gradient
- SGD
- mini-batch stochastic gradient descent (mini- batch SGD)
  - 3.Compute the gradient of the loss with regard to the network’s parameters (a backward pass).  4.Move the parameters a little in the opposite direction from the gradient—for example W -= lr * gradient—thus reducing the loss on the batch a bit.
  - learning rate: If it’s too small, the descent down the curve will take many iterations, and it could get stuck in a local minimum. If step is too large, your updates may end up taking you to completely random locations on the curve.
  - batch太大或太小都不行
  - Momentum addresses two issues with SGD: convergence speed and local minima.
    - Momentum is implemented by moving the ball at each step based not only on the current slope value (current acceleration) but also on the current velocity (resulting from past acceleration).
    - 注意到一般梯度下降方法更新的是位置，或者说时位移，通俗的说就是在这个点还没达到最优值，那我沿着负梯度方向迈一步试试；而momentum update更新的是速度，通俗说就是这个点没达到最优值，那我沿着负梯度方向走快点试试，然后再更新位移。
    - 这里超参数mu是为了保证系统最后收敛到局部值，比如我现在快到局部最小点了，因此速度更新量越来越小（梯度接近于0），但是速度有啊，有速度就会继续走，因此加入小于1的mu使每次迭代后速度降下来，最后为0也就不走了。

##### 2.5 Looking back at our first example

##### 2.6 Summary

- Learning happens by drawing random batches of data samples and their targets, and computing the gradient of the network parameters with respect to the loss on the batch. The network parameters are then moved a bit (the magnitude of the move is defined by the learning rate) in the opposite direction from the gradient.
- The entire learning process is made possible by the fact that neural networks are chains of differentiable tensor operations, and thus it’s possible to apply the chain rule of derivation to find the gradient function mapping the current parameters and current batch of data to a gradient value.
<br><br>
#### Chapter 3 Getting started with neural networks

##### 3.1 Anatomy of a neural network

- Loss function: For instance, you’ll use binary crossentropy for a two-class classification problem, categorical crossentropy for a many-class classification problem, mean- squared error for a regression problem, connectionist temporal classification (CTC) for a sequence-learning problem, and so on.

##### 3.2 Keras

##### 3.3 Setting up a deep-learning workstation

##### 3.4 Classifying movie reviews: a binary classification example

- What are activation functions, and why are they necessary?
  Without an activation function like relu (also called a non-linearity), the Dense layer would consist of two linear operations—a dot product and an addition:
  output = dot(W, input) + b
  So the layer could only learn linear transformations (affine transformations) of the input data: the hypothesis space of the layer would be the set of all possible linear transformations of the input data into a 16-dimensional space.
  In order to get access to a much richer hypothesis space that would benefit from deep representations, you need a non-linearity, or activation function.
- But crossentropy is usually the best choice when you’re dealing with models that output probabilities. Crossentropy is a quantity from the field of Infor- mation Theory that measures the distance between probability distributions or, in this case, between the ground-truth distribution and your predictions.
- 'Keras.History' object has a mem- ber history, which is a dictionary containing data about everything that happened during training.
- Stacks of Dense layers with relu activations can solve a wide range of problems

##### 3.5 Classifying newswires: a multiclass classification example

- The importance of having sufficiently large intermediate layers. This drop is mostly due to the fact that you’re trying to compress a lot of information (enough information to recover the separation hyperplanes of 46 classes) into an intermediate space that is too low-dimensional. The network is able to cram most of the necessary information into these eight-dimensional representations, but not all of it. 中间层不能比输出层小，不然信息压缩，部分丢失。

##### 3.6 Predicting house prices: a regression example

- Again, logistic regression isn’t a regression algorithm—it’s a classification algorithm.
- feature-wise normalization: for each feature in the input data (a column in the input data matrix), you subtract the mean of the feature and divide by the standard deviation, so that the feature is centered around 0 and has a unit standard deviation.
- Note that the quantities used for normalizing the test data are computed using the training data. You should never use in your workflow any quantity computed on the test data, even for something as simple as data normalization.
- axis=0 按列计算 axis=1 按行计算
- Applying an activation function would constrain the range the out- put can take; for instance, if you applied a sigmoid activation function to the last layer, the network could only learn to predict values between 0 and 1. Here, because the last layer is purely linear, the network is free to learn to predict values in any range.
- Validating your approach using K-fold validation: in small datasets, the best practice in such situations is to use K-fold cross-validation (see figure 3.11). It consists of splitting the available data into K partitions (typically K = 4 or 5), instanti- ating K identical models, and training each one on K – 1 partitions while evaluating on the remaining partition. The validation score for the model used is then the average of the K validation scores obtained.
- 当结果波动较大的时候，可以做可视化平滑来smooth curve...保留前面下降较快的点。
  - Omit the first 10 data points, which are on a different scale than the rest of the curve. 
  - Replace each point with an exponential moving average of the previous points, to obtain a smooth curve.

- When little training data is available, it’s preferable to use a small network with few hidden layers (typically only one or two), in order to avoid severe overfitting.
<br><br>
#### Chapter 4 Fundamentals of machine learning

##### 4.1 Four branches of machine learning

- Supervised learning
  - It consists of learning to map input data to known targets (also called annotations), given a set of examples (often annotated by humans).
  - Sequence generation—Given a picture, predict a caption describing it. Sequence generation can sometimes be reformulated as a series of classification problems (such as repeatedly predicting a word or token in a sequence). 
  - Syntax tree prediction—Given a sentence, predict its decomposition into a syntax tree.
  - Object detection—Given a picture, draw a bounding box around certain objects inside the picture. This can also be expressed as a classification problem (given many candidate bounding boxes, classify the contents of each one) or as a joint classification and regression problem, where the bounding-box coordinates are predicted via vector regression. 
  - Image segmentation—Given a picture, draw a pixel-level mask on a specific object.
- Unsupervised learning
  - This branch of machine learning consists of finding interesting transformations of the input data without the help of any targets, for the purposes of data visualization, data compression, or data denoising, or to better understand the correlations present in the data at hand.
  - Dimensionality reduction and clustering are well-known categories of unsupervised learning.
- Self-supervised learning
  - There are still labels involved (because the learning has to be supervised by something), but they’re generated from the input data, typically using a heuristic algorithm. 启发式算法
  - autoencoder, trying to pre- dict the next frame in a video, given past frames, or the next word in a text, given previ- ous words, are instances of self-supervised learning (temporally supervised learning, in this case: supervision comes from future input data).
- Reinforcement learning
  - an agent receives information about its environment and learns to choose actions that will maximize some reward.
- Some definitions
  - Classes—A set of possible labels to choose from in a classification problem. For example, when classifying cat and dog pictures, “dog” and “cat” are the two classes.
  - Label—A specific instance of a class annotation in a classification problem. For instance, if picture #1234 is annotated as containing the class “dog,” then “dog” is a label of picture #1234.
  - Multilabel classification—A classification task where each input sample can be assigned multiple labels. For instance, a given image may contain both a cat and a dog and should be annotated both with the “cat” label and the “dog” label. The number of labels per image is usually variable.
  - Mini-batch or batch—A small set of samples (typically between 8 and 128) that are processed simultaneously by the model. The number of samples is often a power of 2, to facilitate memory allocation on GPU. When training, a mini-batch is used to compute a single gradient-descent update applied to the weights of the model.

##### 4.2 Evaluating machine-learning models

- generalize—that perform well on never-before-seen data
- Hyperparameters -> the number/size of layers and  Parameters -> network's weights
- why three sets? 因为验证数据集（Validation Set)用来调整模型参数从而选择最优模型，模型本身已经同时知道了输入和输出，所以从验证数据集上得出的误差（Error)会有偏差（Bias)。
  但是我们只用测试数据集(Test Set) 去评估模型的表现，并不会去调整优化模型。模型的训练和选择必须和test set无关，防止信息泄露导致模型包括了test set的信息，从而完全过拟合。
- three classic evaluation recipes: 
  - simple hold-out validation
  - K- fold validation
  - iterated K-fold validation with shuffling
    - It consists of applying K-fold validation multiple times, shuffling the data every time before splitting it K ways. The final score is the average of the scores obtained at each run of K-fold validation.
- choosing an evaluation protocol:
  - Data representativeness -- randomly shuffle
  - The arrow of time -- temporal leak
  - Redundancy in your data -- training set adn validation set are disjiont

##### 4.3 Data preprocessing, feature engineering, and feature learning

- Data preprocessing aims at making the raw data at hand more amenable to neural networks.
  - VECTORIZATION
    - All inputs and targets in a neural network must be tensors of floating-point data (or, in specific cases, tensors of integers), a step called data vectorization.
  - VALUE NORMALIZATION
    - In the digit-classification example, you started from image data encoded as integers in the 0–255 range, encoding grayscale values. Before you fed this data into your net- work, you had to cast it to float32 and divide by 255 so you’d end up with floating- point values in the 0–1 range.
    - Before you fed this data into your network, you had to normalize each feature independently so that it had a standard deviation of 1 and a mean of 0.
    - Take small values—Typically, most values should be in the 0–1 range. 
    - Be homogenous—That is, all features should take values in roughly the same range.
  - HANDLING MISSING VALUES
    - In general, with neural networks, it’s safe to input missing values as 0, with the con- dition that 0 isn’t already a meaningful value. The network will learn from exposure to the data that the value 0 means missing data and will start ignoring the value.
    - In this situation, you should artificially generate training samples with missing entries: copy some training samples several times, and drop some of the fea- tures that you expect are likely to be missing in the test data.
- Feature engineering
  - making a problem easier by expressing it in a simpler way. It usually requires understanding the problem in depth.
  - Fortunately, modern deep learning removes the need for most feature engineer- ing, because neural networks are capable of automatically extracting useful features from raw data.
  - Good features still allow you to solve problems more elegantly while using fewer resources.
  - Good features let you solve a problem with far less data. The ability of deep- learning models to learn features on their own relies on having lots of training data available; if you have only a few samples, then the information value in their features becomes critical.

##### 4.4 Overfitting and underfitting

- To prevent a model from learning misleading or irrelevant patterns found in the training data, the best solution is to get more training data. A model trained on more data will naturally generalize better. When that isn’t possible, the next-best solution is to modulate the quantity of information that your model is allowed to store or to add constraints on what information it’s allowed to store. If a network can only afford to memorize a small number of patterns, the optimization process will force it to focus on the most prominent patterns, which have a better chance of generalizing well.
- Reducing the network’s size
  - more parameters -> more memorization capacity -> low generalization power
  - 通过在验证集上调参来找到折中方案。找到合适的模型大小的一般工作流程是从相对较少的层和参数开始，并增加层的大小或添加新层，直到看到有关验证丢失的收益递减。
- Adding weight regularization
  - Simpler models are less likely to overfit than complex ones.
  - to put constraints on the complex- ity of a network by forcing its weights to take only small values, which makes the distribution of weight values more regular. This is called weight regularization, and it’s done by adding to the loss function of the network a cost associated with having large weights.
  - L1 regularization—The cost added is proportional to the absolute value of the weight coefficients (the L1 norm of the weights).
  - L2 regularization—The cost added is proportional to the square of the value of the weight coefficients (the L2 norm of the weights). L2 regularization is also called weight decay in the context of neural networks. Don’t let the different name confuse you: weight decay is mathematically the same as L2 regularization.
- Adding dropout
  - Dropout, applied to a layer, consists of randomly dropping out (setting to zero) a number of output features of the layer during training.
  - At test time, no units are dropped out; instead, the layer’s output values are scaled down by a factor equal to the dropout rate, to balance for the fact that more units are active than at training time.（？）
  - noise in the output values of a layer can break up happenstance patterns that aren’t significant, which the network will start memorizing if no noise is present.

##### 4.5 The universal workflow of machine learning（好好学、七部曲

- Defining the problem and assembling a dataset
  - What will your input data be? What are you trying to predict?
  - What type of problem are you facing? Is it binary classification? Multiclass classi- fication? Scalar regression? Vector regression? Multiclass, multilabel classifica- tion? Something else, like clustering, generation, or reinforcement learning?
  - Not all problems can be solved; just because you’ve assembled exam- ples of inputs X and targets Y doesn’t mean X contains enough information to predict Y.
  - nonstationary problems
  - Keep in mind that machine learning can only be used to memorize patterns that are present in your training data. You can only recognize what you’ve seen before. Using machine learning trained on past data to predict the future is making the assumption that the future will behave like the past. That often isn’t the case.
- Choosing a measure of success
  - To control something, you need to be able to observe it. To achieve success, you must define what you mean by success—accuracy? Precision and recall? Customer-retention rate? Your metric for success will guide the choice of a loss function: what your model will optimize.
  - it’s helpful to browse the data science competitions on Kaggle (https://kaggle.com); they showcase a wide range of problems and evaluation metrics.
- Deciding on an evaluation protocol
  - Maintaining a hold-out validation set—The way to go when you have plenty of
    data 常用。
- Preparing your data
  - The values taken by these tensors should usually be scaled to small values: for example, in the [-1, 1] range or [0, 1] range.
  - If different features take values in different ranges (heterogeneous data), then
    the data should be normalized.
- Developing a model that does better than a baseline
  - Last-layer activation—This establishes useful constraints on the network’s output.
  - Loss function—This should match the type of problem you’re trying to solve.
  - Optimization configuration—What optimizer will you use? What will its learning rate be? In most cases, it’s safe to go with rmsprop and its default learning rate.
- Scaling up: developing a model that overfits
  - Once you’ve obtained a model that has statistical power, the question becomes, is your model sufficiently powerful? Does it have enough layers and parameters to properly model the problem at hand?
  - between overfitting and underfitting/ undercapacity and overcapacity
  - Always monitor the training loss and validation loss, as well as the training and valida- tion values for any metrics you care about. When you see that the model’s performance on the validation data begins to degrade, you’ve achieved overfitting.
- Regularizing your model and tuning your hyperparameters
  - Optionally, iterate on feature engineering: add new features, or remove fea- tures that don’t seem to be informative.
  - Be mindful of the following: every time you use feedback from your validation process to tune your model, you leak information about the validation process into the model. Repeated just a few times, this is innocuous; but done systematically over many itera- tions, it will eventually cause your model to overfit to the validation process.
  - Once you’ve developed a satisfactory model configuration, you can train your final production model on all the available data (training and validation) and evaluate it one last time on the test set. If it turns out that performance on the test set is signifi- cantly worse than the performance measured on the validation data, this may mean either that your validation procedure wasn’t reliable after all, or that you began over- fitting to the validation data while tuning the parameters of the model. In this case, you may want to switch to a more reliable evaluation protocol.

