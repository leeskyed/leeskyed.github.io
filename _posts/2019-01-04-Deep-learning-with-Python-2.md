---
layout: post
title:  "Deep learning with Python Part 2 (update)"
date:   2019-1-04 16:11:01 +0800
categories: DeepLearning
tag: Books
---

* content
{:toc}

### Part 2 Deep learning in practice

#### Chapter 5 Deep learning for computer vision

##### 5.1 Introduction to ConvNets

- Dense layers learn global patterns in their input feature space, whereas convolution layers learn local patterns
- two properties:
  - The patterns they learn are translation invariant
  - They can learn spatial hierarchies of patterns
    - A first convolution layer will learn small local patterns such as edges, a second convolution layer will learn larger patterns made of the features of the first layers, and so on. 越底层越局部，深度越深学到的特征域越大
- feature maps: height, width, depth(channel)
- Its depth can be arbitrary, because the output depth is a parameter of the layer, and the different channels in that depth axis no longer stand for specific colors as in RGB input; rather, they stand for filters. Filters encode specific aspects of the input data: at a high level, a single filter could encode the concept “presence of a face in the input,” for instance.(重点)
- Output feature map means: every dimension in the depth axis is a feature (or filter), and the 2D tensor output[:, :, n] is the 2D spatial map of the response of this filter over the input.
- "same", which means “pad in such a way as to have an output with the same width and height as the input.
- "valid", which means no padding
- Pooling: to aggressively downsample feature maps, much like strided convolutions.
  - 通过downsample来提高学习不同spatial hierarchy的features 不然还仍然只是学习局部特征
  - 通过downsample来减少参数，降低过拟合
  - The high-level patterns learned by the convnet will still be very small with regard to the initial input, which may not be enough to learn to clas- sify digits (try recognizing a digit by only looking at it through windows that are 7 × 7 pixels!).
  - reduce the number of feature-map coefficients to process, as well as to induce spatial-filter hierarchies by making succes- sive convolution layers look at increasingly large windows
  - In a nutshell, the reason is that features tend to encode the spatial presence of some pattern or concept over the different tiles of the feature map (hence, the term feature map), and it’s more informative to look at the maximal presence of different features than at their average presence.（相比起average-pooling maxpooling可能更具信息性和特征性
  - So the most reasonable subsampling strategy is to first produce dense maps of features (via unstrided convolutions) and then look at the maximal activation of the features over small patches, rather than looking at sparser windows of the inputs (via strided convolutions) or averaging input patches, which could cause you to miss or dilute feature-presence information.

##### 5.2 Training a convnet from scratch on a small dataset

- 复习一下基本流程：

  - **training a small model from scratch(data augmentation)**, **doing feature extraction using a pretrained model**, and **fine-tuning a pretrained model**—will constitute your future toolbox for tackling the problem of per- forming image classification with small datasets.

- **Building your network:**

  - The depth of the feature maps progressively increases in the network (from 32 to 128), whereas the size of the feature maps decreases (from 148 × 148 to 7 × 7). This is a pattern you’ll see in almost all convnets.

- **Data preprocessing**:

  - Read the picture files. 

  - Decode the JPEG content to RGB grids of pixels. 

  - Convert these into floating-point tensors. 

  - Rescale the pixel values (between 0 and 255) to the [0, 1] interval (as you know, neural networks prefer to deal with small input values).

  - ```python
    from keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
    	train_dir,
    	target_size=(150, 150),
    	bach_size=20,
    	class_mode='binary')
    
    validation_generator = test_datagen.flow_from_directory(
    	validation_dir,
    	target_size=(150, 150),
    	bach_size=20,
    	class_mode='binary')
    
    history = model.fit_generator(
    	train_generator,
    	steps_per_epoch=100, # all / batch_size
    	epochs=30, # tell the model when to stop
    	validation_data=validation_generator,
    	validation_steps=50
    )
    ```

  - yield在python中的作用，iterator，return

- **data augmentation**

  - by augmenting the samples via a number of random transformations that yield believable-looking images. 

  - In Keras, this can be done by configuring a number of random transformations to be performed on the images read by the ImageDataGenerator instance.

  - ```
    datagen = ImageDataGenerator(
        roration=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    # more info in document
    ```

  - 

##### 5.3 Using a pretrained convnet

- A pretrained network is a saved network that was previously trained on a large dataset, typically on a large-scale image-classification task. If this original dataset is large enough and general enough, then the spatial hierarchy of fea- tures learned by the pretrained network can effectively act as a generic model of the visual world, and hence its features can prove useful for many different computer- vision problems, even though these new problems may involve completely different classes than those of the original task Such portability of learned features across different problems is a key advantage of deep learning compared to many older, shallow-learning approaches, and it makes deep learning very effective for small-data problems.
- **Feature extraction**
  - Feature extraction consists of using the representations learned by a previous network to extract interesting features from new samples. These features are then run through a new classifier, which is trained from scratch.
  - ConvNets used for image classification comprise two parts:
    - convolutional base
      - In the case of convnets, feature extraction consists of taking the convolutional base of a previously trained network, running the new data through it, and training a new clas- sifier on top of the output
      - the representations learned by the convolutional base are likely to be more generic
      - representations learned by the classifier will necessarily be specific to the set of classes on which the model was trained
      - densely connected layers no longer contain any information about where objects are located in the input image: these layers get rid of the notion of space, whereas the object location is still described by convolutional feature maps
      - Layers that come earlier in the model extract local, highly generic feature maps (such as visual edges, colors, and textures), whereas layers that are higher up extract more-abstract concepts (such as “cat ear” or “dog eye”).
      - new dataset differs a lot from the dataset on which the original model was trained, you may be bet- ter off using only the first few layers of the model to do feature extraction, rather than using the entire convolutional base.
      - But we’ll choose not to, in order to cover the more gen- eral case where the class set of the new problem doesn’t overlap the class set of the original model.
    - **two ways to proceed**
      - Running the convolutional base over your dataset, recording its output to a Numpy array on disk, and then using this data as input to a standalone, densely connected classifier similar to those you saw in part 1 of this book. Fast and cheap. But no data augmentation
        - The extracted features are currently of shape (samples, 4, 4, 512). You’ll feed them to a densely connected classifier, so first you must flatten them to (samples, 8192):
      - Extending the model you have (conv_base) by adding Dense layers on top, and running the whole thing end to end on the input data. This will allow you to use data augmentation, because every input image goes through the convolutional base every time it’s seen by the model.
        - Freezing a layer or set of layers means preventing their weights from being updated during training. If you don’t do this, then the representations that were pre- viously learned by the convolutional base will be modified during training. Because the Dense layers on top are randomly initialized, very large weight updates would be propagated through the network, effectively destroying the representations previously learned.
- **Fine-tuning**
  - 实际上就是让深度较深的层不要freeze，反而丢进数据让他们重新训练参数
  - 步骤
    - Add your custom network on top of an already-trained base network. 
    - Freeze the base network. 
    - Train the part you added.  到这里和上面部分pretrain中是一样的
    - Unfreeze some layers in the base network.  然后再解封一点高级层数，因为越高级越和任务相关
    - Jointly train both these layers and the part you added. 然后再训练一次
  - 为什么只fine-tune一点层数：
    - 低级层基于编码更一般、可复用的特征，而高基层则是更具体任务相关的特征
    - 训练越多参数则会越过拟合
    - You’ll do this with the RMSProp opti- mizer, using a very low learning rate. The reason for using a low learning rate is that you want to limit the magnitude of the modifications you make to the representations of the three layers you’re fine-tuning.
  - You may wonder, how could accuracy stay stable or improve if the loss isn’t decreasing? The answer is simple: what you display is an average of pointwise loss val- ues; but what matters for accuracy is the distribution of the loss values, not their aver- age, because accuracy is the result of a binary thresholding of the class probability predicted by the model. The model may still be improving even if this isn’t reflected in the average loss.

##### 5.4 Visualizing what convnets learn

- Visualizing intermediate activations: 生成每个filter后的feature map

  - Visualizing intermediate activations consists of displaying the feature maps that are output by various convolution and pooling layers in a network, given a certain input (the output of a layer is often called its activation, the output of the activation func- tion).

  - Each channel encodes relatively independent features, so the proper way to visualize these feature maps is by independently plot- ting the contents of every channel as a 2D image.

  - **preprocessing a single image**

    - image.load_img, img_to_array, np.expand_dims, img /= 255.

  - **extract the feature maps**

    - ```python
      from keras import models
      #Extracts the outputs of the top eight layers
      layer_outputs = (layer.output for layer in model.layers[:8])
      # Creates a model that will return these outputs, given the model inpu
      activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
      # Returns a list of five Numpy arrays: one array per layer activation
      activations = activation_model.predict(img_tensor)
      ```

    - Visualizing every channel in every intermediate activation

      ```python
      import keras
      
      # These are the names of the layers, so can have them as part of our plot
      layer_names = []
      for layer in model.layers[:8]:
          layer_names.append(layer.name)
      
      images_per_row = 16
      
      # Now let's display our feature maps
      for layer_name, layer_activation in zip(layer_names, activations):
          # This is the number of features in the feature map
          n_features = layer_activation.shape[-1]
      
          # The feature map has shape (1, size, size, n_features)
          size = layer_activation.shape[1]
      
          # We will tile the activation channels in this matrix
          n_cols = n_features // images_per_row
          display_grid = np.zeros((size * n_cols, images_per_row * size))
      
          # We'll tile each filter into this big horizontal grid
          for col in range(n_cols):
              for row in range(images_per_row):
                  channel_image = layer_activation[0,
                                                   :, :,
                                                   col * images_per_row + row]
                  # Post-process the feature to make it visually palatable
                  channel_image -= channel_image.mean()
                  channel_image /= channel_image.std()
                  channel_image *= 64
                  channel_image += 128
                  channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                  display_grid[col * size : (col + 1) * size,
                               row * size : (row + 1) * size] = channel_image
      
          # Display the grid
          scale = 1. / size
          plt.figure(figsize=(scale * display_grid.shape[1],
                              scale * display_grid.shape[0]))
          plt.title(layer_name)
          plt.grid(False)
          plt.imshow(display_grid, aspect='auto', cmap='viridis')
          
      plt.show()
      
      ```

    - The first layer acts as a collection of various edge detectors.

    - As you go higher, the activations become increasingly abstract and less visually interpretable. Higher presentations carry increasingly less information about the visual contents of the image, and increasingly more information related to the class of the image.

    - The sparsity of the activations increases with the depth of the layer: in the first layer, all filters are activated by the input image; but in the following layers, more and more filters are blank. This means the pattern encoded by the filter isn’t found in the input image.

  - A deep neural network effectively acts as an information distillation pipeline, with raw data going in (in this case, RGB pictures) and being repeatedly transformed so that irrelevant infor- mation is filtered out (for example, the specific visual appearance of the image), and useful information is magnified and refined (for example, the class of the image).

- Visualizing convnet filters： 

  - 生成filter的visual pattern，通过(gradient ascent in input space)输入空间的梯度上升可以实现：将gradient descent梯度下降应用于convnet的输入图像的值，以便从空白输入图中开始最大化特定filter的response。输入图的结果是那个filter有最大的responsive的。

  - ```python
    # Defining the loss tensor for filter visualization
    from keras.applications import VGG16 from keras import backend as K
    
    model = VGG16(weights='imagenet', include_top=False)
    
    layer_name = 'block3_conv1' filter_index = 0
    
    layer_output = model.get_layer(layer_name).output 
    loss = K.mean(layer_output[:, :, :, filter_index])
    # Obtaining the gradient of the loss with regard to the input
    # The call to gradients returns a list of tensors (of size 1 in this case). Hence, you keep only the first element— which is a tensor.
    grads = K.gradients(loss, model.input)[0]
    # Gradient-normalization trick
    # A non-obvious trick to use to help the gradient-descent process go smoothly is to nor- malize the gradient tensor by dividing it by its L2 norm (the square root of the average of the square of the values in the tensor). This ensures that the magnitude of the updates done to the input image is always within the same range.
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    # Fetching Numpy output values given Numpy input values
    iterate = K.function([model.input], [loss, grads])
    
    import numpy as np 
    loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])
    
    # Loss maximization via stochastic gradient descent
    # Starts from a gray image with some noise
    input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.
    step = 1.
    for i in range(40): 
        loss_value, grads_value = iterate([input_img_data])
    	input_img_data += grads_value * step
    ```

  - The resulting image tensor is a floating-point tensor of shape (1, 150, 150, 3), with values that may not be integers within [0, 255]. Hence, you need to postprocess this tensor to turn it into a displayable image

  - ```python
    # Utility function to convert a tensor into a valid image
    def deprocess_image(x):
        # Normalizes the tensor: centers on 0, ensures that std is 0.1
        x -= x.mean() 
        x /= (x.std() + 1e-5) 
        x *= 0.1
    	# Clips to [0, 1]
    	x += 0.5 
    	x = np.clip(x, 0, 1)
        # converts to an RGB array
        x *= 255 
        x = np.clip(x, 0, 255).astype('uint8') 
        return x
    
    # Function to generate filter visualizations
    def generate_pattern(layer_name, filter_index, size=150): 		
        layer_output = model.get_layer(layer_name).output 
        loss = K.mean(layer_output[:, :, :, filter_index])
        # Computes the gradient of the input picture with regard to this loss
        grads = K.gradients(loss, model.input)[0]
        # Normalization trick: normalizes the gradient
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        # Returns the loss and grads given the input picture
        iterate = K.function([model.input], [loss, grads])
        # Starts from a gray image with some noise
        input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
        
        step = 1.
        for i in range(40): 
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step
            
        img = input_imge_data[0]
        return deprocess_image(img)
    ```

  - These filter visualizations tell you a lot about how convnet layers see the world: each layer in a convnet learns a collection of filters such that their inputs can be expressed as a combination of the filters. This is similar to how the Fourier transform decom- poses signals onto a bank of cosine functions. The filters in these convnet filter banks get increasingly complex and refined as you go higher in the model:

    - The filters from the first layer in the model (block1_conv1) encode simple

      directional edges and colors (or colored edges, in some cases).

    - The filters from block2_conv1 encode simple textures made from combina-

      tions of edges and colors.

    - The filters in higher layers begin to resemble textures found in natural images: feathers, eyes, leaves, and so on.

- Visualizing heatmaps of class activation: 

  - which parts of a given image led a convnet to its final classification decision. This is helpful for debugging the decision process of a convnet, particularly in the case of a classification mistake.

  - class activation map (CAM) visualization

  - heatmap:  a 2D grid of scores associated with a specific output class, computed for every location in any input image, indicating how important each location is with respect to the class under consideration.

  - ```python
    # Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.
    from keras.applications.vgg16 import VGG16
    model = VGG16(weights='imagenet')
    # preprocessing iput
    from keras.preprocessing import image from keras.applications.vgg16 import preprocess_input, decode_predictions import numpy as np
    
    img_path = '/Users/fchollet/Downloads/creative_commons_elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Setting up the Grad-CAM algorithm
    african_e66lephant_output = model.output[:, 386]
    last_conv_layer = model.get_layer('block5_conv3')
    grads = K.gradients(african_elephant_output, last_conv_layer.output)[0] 
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads,last_conv_layer.output[0]])
    
    pooled_grads_value, conv_layer_output_value = iterate([x])
    
    for i in range(512):
    	conv_layer_output_value[:, :, i] *= pooled_grads_value[i] 
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    
    # Heatmap post-processing
    heatmap = np.maximum(heatmap, 0) 
    heatmap /= np.max(heatmap) 
    plt.matshow(heatmap)
    
    # Superimposing the heatmap with the original picture
    import cv2 
    img = cv2.imread(img_path)  
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    heatmap = np.uint8(255 * heatmap)  # convert heatmap to RGB
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) superimposed_img = heatmap * 0.4 + img cv2.imwrite('/Users/fchollet/Downloads/elephant_cam.jpg', superimposed_img)
    ```

  - **This visualization technique answers two important questions:** 

    - Why did the network think this image contained an African elephant? 
    - Where is the African elephant located in the picture?

#### Chapter 6 Deep learning for text and sequences

- 这一部分由于时间关系先跳过了，后面可以先看看CNN+RNN结合来解决long range dependency的问题。
- Summary：
  - You can use RNNs for timeseries regression (“predicting the future”), timeseries classification, anomaly detection in timeseries, and sequence labeling (such as identifying names or dates in sentences).
  - Similarly, you can use 1D convnets for machine translation (sequence-to- sequence convolutional models, like SliceNet a ), document classification, and spelling correction.
  - If global order matters in your sequence data, then it’s preferable to use a recurrent network to process it. This is typically the case for timeseries, where the recent past is likely to be more informative than the distant past.
  - If global ordering isn’t fundamentally meaningful, then 1D convnets will turn out to work at least as well and are cheaper. This is often the case for text data, where a keyword found at the beginning of a sentence is just as meaningful as a keyword found at the end.



#### Chapter 7 Advanced deep-learning best practices

##### 7.1 Keras functional API

- 除了sequence之外，还有一种就是functional equivalent形式的，和tensorflow有点像，不过input放在最后面做单独输入，**还有model初始化中要写明输入和输出**。

- Multi-input model：

  - feed the model a list of Numpy arrays as inputs

  - ```python
    model.fit([text, question], answers, epochs=10, batch_size=128)
    ```

  - feed it a dictionary that maps input names to Numpy arrays(available only if you give names to your inputs.)

  - ```
    model.fit({'text': text, 'question': question}, answers, epochs=10, batch_size=128)
    ```

- Multi-output model:

  - different loss functions for different heads:

  - for instance, age prediction is a scalar regression task, but gender prediction is a binary classification task, requiring a different training procedure. But because gradient descent requires you to minimize a scalar, you must combine these losses into a single value in order to train the model.

  - ```python
    model.compile(optimizer='rmsprop', 
                  loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'])
    # Equivalent (possible only if you give names to the output layers)
    # In Keras, you can use either a list or a dictionary of losses in compile to specify different objects for different outputs; the resulting loss values are summed into a global loss, which is minimized during training.
    model.compile(optimizer='rmsprop', 
                  loss={'age': 'mse', 
                        'income': 'categorical_crossentropy', 
                        'gender': 'binary_crossentropy'})
    
    # Note that very imbalanced loss contributions will cause the model representations to be optimized preferentially for the task with the largest individual loss, at the expense of the other tasks. To remedy this, you can assign different levels of importance to the loss values in their contribution to the final loss. This is useful in particular if the losses’ values use different scales.
    model.compile(optimizer='rmsprop', 
                  loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'], 
                  loss_weights=[0.25, 1., 10.])
    
    model.compile(optimizer='rmsprop', 
                  loss={'age': 'mse', 
                        'income': 'categorical_crossentropy', 
                        'gender': 'binary_crossentropy'}, 
                  loss_weights={'age': 0.25, 
                                'income': 1., 
                                'gender': 10.})
    ```

  - feeding data to a multi-output model

    - ```python
      model.fit(posts, [age_targets, income_targets, gender_targets], epochs=10, batch_size=64)
      
      model.fit(posts, {'age': age_targets, 
                        'income': income_targets, 
                        'gender': gender_targets}, 
                epochs=10, batch_size=64)
      ```

- Directed acyclic graph of layers

  - cycles are not allowed but loops.

  - **Inception modules**

    - 

    - inspired by network-in-network

    - a stack of modules that themselves look like small independent networks, split into several parallel branches.

    - The purpose of 1 * 1 convolutions

      - An edge case is when the patches extracted consist of a single tile. The convolution operation then becomes equivalent to running each tile vector through a Dense layer: it will compute features that mix together information from the channels of the input tensor, but it won’t mix information across space (because it’s looking at one tile at a time). Such 1 × 1 convolutions (also called pointwise convolutions) are featured in Inception mod- ules, where they contribute to factoring out channel-wise feature learning and space- wise feature learning—a reasonable thing to do if you assume that each channel is highly autocorrelated across space, but different channels may not be highly correlated with each other.
      - 简单来讲就是为了捕捉channel之间的correlation 只提取出channel-wise feature

    - ```python
      # keras.applications .inception_v3.InceptionV3, pretrained on ImageNet
      from keras import layers
      branch_a = layers.Conv2D(128, 1, activation='relu', strides=2)(x) 
      branch_b = layers.Conv2D(128, 1, activation='relu')(x) 
      branch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_b)
      
      branch_c = layers.AveragePooling2D(3, strides=2)(x) 
      branch_c = layers.Conv2D(128, 3, activation='relu')(branch_c)
      
      branch_d = layers.Conv2D(128, 1, activation='relu')(x) 
      branch_d = layers.Conv2D(128, 3, activation='relu')(branch_d) 
      branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)
      
      output = layers.concatenate( [branch_a, branch_b, branch_c, branch_d], axis=-1)
      ```

    - **Xception inspired by Inception**, spatial conv seperately first then channel.. **Higher accuracy than InceptionV3**