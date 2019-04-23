**<font size=4>Lec1: Introduction</font>**\
cs231n focus: image classification\
\
**<font size=4>Lec2: Image classification pipeline</font>** \
如何分类？\
1.KNN缺点：1) fast in training, slow in inference
     2) dimensionality curse\
2.linear classifier: score = f(W, x)\
\
**<font size=4>Lec3: Loss Functions and Optimization</font>**\
1.**Multiclass SVM loss** [SVM之Hinge Loss解释][1] \
\begin{aligned}
L_i &=\sum_{j\neq y_i} 
\left. \{
  \begin{array}{l}
  0 & if  s_{y_i}\ge s_j+1\\
  s_j-s_{y_i}+1 & otherwise
  \end{array}
  \right.} \\
  &=\sum_{j\neq y_i}max(0, s_j-s_{y_i}+1)
  \end{aligned}

\
2.**Regularization**\
<img src="Regularization.png" width = "400" height = "200"> \
\
3.**Softmax Classifier** (Multinomial Logistic Regression)\
<img src="softmax.png" width = "400" height = "200">
\
4.**Softmax vs. SVM** \
[Support Vector Machine vs Logistic Regression][2]
\
[Logistic Loss ](http://www.hongliangjie.com/wp-content/uploads/2011/10/logistic.pdf)

[1]: https://www.cnblogs.com/guoyaohua/p/9436237.html
[2]:https://towardsdatascience.com/support-vector-machine-vs-logistic-regression-94cc2975433f
\
**Lec4: Backpropagation and Neural Networks**\
1.Backpropagation chain rule: [local gradient] x [upstream gradient]
<img src="backprop.png" width = "400" height = "200"> \
add gate: gradient distributor\
max gate: gradient router\
mul gate: gradient switcher\
Gradients add at branches

2.Gradients for vectorized code: [Jacobian matrix][3] \
<img src="vectorized gradient.png" width = "400" height = "200"> \
!!Always check: The gradient with respect to a variable should have the same shape as the variable

[3]: https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant

\
**<font size=4>Lec5: Convolutional Neural Networks</font>**\
1.**convolution layer**: Use convolutional layers to preserve 
the spatial structure
<img src="filters.png" width = "500" height = "250"> \
ConvNet is a sequence of Convolutional Layers, interspersed with activation functions(eg.ReLu).
filter: receptive field.\

2.**pooling layer**(没有引入参数)
<img src="pooling layer.png" width = "420" height = "250">
- max pooling:
<img src="max pooling.png" width = "500" height = "200"> 

3.**Summary**
- ConvNets stack CONV,POOL,FC layers
- Trend towards smaller filters and deeper architectures
- Trend towards getting rid of POOL/FC layers (just CONV)
- Typical architectures look like [(CONV-RELU)*N-POOL?]*M-(FC-RELU)*K,SOFTMAX where N is usually up to ~5, M is large, 0 <= K <= 2.
- but recent advances such as ResNet/GoogLeNet challenge this paradigm

\
**<font size=4>Lec 6: Training Neural Networks(I)</font>**\
**Overview**\
1)One time setup\
 activation functions, preprocessing, weight initialization, regularization, gradient checking
\
2)Training dynamics \
babysitting the learning process, parameter updates, hyperparameter optimization
\
3)Evaluation\
 model ensembles

1.**Activation Functions** 
<img src="activation func.png" width = "380" height = "170"> \
(1)**sigmoid函数**\
<img src="sigmoid.png" width = "400" height = "200"> \
<img src="sigmoid1.png" width = "400" height = "200"> \
(2)**tanh函数**\
<img src="tanh.png" width = "400" height = "150"> \
(3)**ReLU函数**\
<img src="ReLU.png" width = "400" height = "180"> \
<img src="ReLU1.png" width = "400" height = "220"> \
(4)**Leaky ReLU函数**\
<img src="leaky ReLU.png" width = "480" height = "200"> \
(5)**ELU函数**\
<img src="ELU.png" width = "480" height = "200">

(6)**Maxout “Neuron”** [Goodfellow et al., 2013]\
<img src="maxout.png" width = "330" height = "130">

**In practice:**
- Use ReLU. Be careful with your learning rates
- Try out Leaky ReLU / Maxout / ELU
- Try out tanh but don’t expect much
- Don’t use sigmoid

2.**Data Preprocessing**\
<img src="image preprocess.png" width = "400" height = "200">\

3.**Weight Initialization**\
(1) ``W=0``\
All the neurons will do the same thing, have same operation on top of the inputs, output the same thing, get same the same gradient and update in the same way.\
However what we want is that neurons do different things. So initialize everything equally is no symmetry breaking.\
(2) Small random numbers (eg.gaussian with zero mean and 1e-2 standard deviation)\
``W = np.random.randn(fan_in, fan_out) * 0.01``\
Works ~okay for small networks, but problems with deeper networks: All activations become zero(因为不断乘以由小的随机数构成的W)! Think about the backward pass for a W*X gate, because X’s are very small(around zero), we get the same problem of gradients being too small and no update.\
(3) Big random numbers (eg.gaussian with zero mean and 1 standard deviation)\
``W = np.random.randn(fan_in, fan_out) * 1.0``\
Almost all neurons completely saturated, either -1 and 1. Gradients will be all zero.\
(4) Xavier initialization\
``W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)``\
Want: variance of the input to be the same as variance of the output\
(5) for ReLU neurons\
``W = np.random.randn(fan_in, fan_out) * sqrt(2.0/fan_in)``

4.**Batch Normalization**\
<img src="BN.png" width = "400" height = "190">\
<img src="BN1.png" width = "500" height = "200">\
*Note*: 
- BatchNorm layer usually inserted after Fully Connected or Convolutional layers, and before nonlinearity.
- At test time BatchNorm layer functions differently:
The mean/std are not computed based on the batch. Instead, a single fixed empirical mean of activations during training is used.
(e.g. can be estimated during training with running averages)

5.**Babysitting the Learning Process**\
**Step 1: Preprocess the data**\
**Step 2: Choose the architecture**\
-Double check that the loss is reasonable\
**Step 3: Start to train**\
-Make sure that you can overfit very small portion of the training data\
-Start with small regularization and find learning rate that makes the loss go down:\
<font color='brown'>*loss not going down*</font>: learning rate too low\
<font color='brown'>*loss exploding*</font>: learning rate too high. cost: NaN almost always means high learning rate...\
-Rough range for learning rate we should be cross-validating is somewhere [1e-3 … 1e-5]

6.**Hyperparameter Optimization**\
(1)**Cross-validation strategy**\
**coarse -> fine** cross-validation in stages\
**First stage**: only a few epochs to get rough idea of what params work\
**Second stage**: longer running time, finer search … (repeat as necessary)
*Tip for detecting explosions in the solver:
If the cost is ever > 3 * original cost, break out early*\
(2)**Random Search vs. Grid Search**\
(3)**Hyperparameters to play with**:\
  -network architecture\
  -learning rate, its decay schedule, update type\
  -regularization (L2/Dropout strength)

7.**Summary**
- Activation Functions (use ReLU)
- Data Preprocessing (images: subtract mean)
- Weight Initialization (use Xavier init)
- Batch Normalization (use)
- Babysitting the Learning process
- Hyperparameter Optimization
(random sample hyperparams, in log space when appropriate)

**<font size=4>Lec 7: Training Neural Networks(II)</font>**\
**I.Optimization**\
1.**Problems with SGD**\
1.1.**Poor Conditioning**\
<img src="SGD.png" width = "530" height = "230">\
1.2.**Local Optima**\
<img src="SGD1.png" width = "450" height = "200">\
1.3.**Gradient Noise**\
<img src="SGD2.png" width = "450" height = "200">

2.**SGD + Momentum**\
<img src="SGD+Momentum.png" width = "450" height = "200">\
**Nesterov Momentum**\
<img src="Nesterov.png" width = "450" height = "190">\
**Q**: sgd+momentum may overshoot and in some cases it will come back to the minima, but what if it's a sharp minima? will sgd+momentum miss the minima?\
**A**: the idea of some recent theoretical work is that the sharp minima is bad minima and we dont even want to land into those minima. the sharp minima may be the minima that overfits more. Imagine we double the dataset and the whole optimization landscape would change and those sharp minima may disappear. Flat minima is more robust as we change the training data, hence generalize better to test data. so it's actually a feature rather than a bug that SGD+momentum skips over the sharp minima.

3.**AdaGrad && RMSProp**\
**AdaGrad features**\
(1) When the loss func has high condition number, AdaGrad has the nice property of accelerating movement along one dimension and slowing down movement along the wiggling dimension.\
(2) Step size gets smaller since the grad_squared term gets larger, may get stuck before reaching the minima, so not commonly used in practice.\
<img src="AdaGrad+RMSProp.png" width = "500" height = "210">

4.**Adam**\
<img src="Adam.png" width = "540" height = "210">

<img src="opt methods.png" width = "450" height = "210">

5.**Learning rate**\
<img src="learning rate.png" width = "450" height = "220">

6.**Second-Order Optimization**\
second-order Taylor expansion:
$$J(\theta)\approx J(\theta_0)+(\theta - \theta_0)^T\nabla _{\theta}J(\theta_0)+\frac{1}{2}(\theta - \theta_0)^T H(\theta - \theta_0)$$
Solving for the critical point we obtain the **Newton parameter update**:
$$ \theta^*=\theta_0-H^{-1}\nabla _{\theta}J(\theta_0) $$
Pro: No hyperparameters! No learning rate!\
Con: Hessian has O(N^2) elements Inverting takes O(N^3) N = (Tens or Hundreds of) Millions

**Quasi-Newton methods** (BGFS most popular):
instead of inverting the Hessian (O(n^3)), approximate inverse Hessian with rank 1 updates over time (O(n^2) each).\
**L-BFGS** (Limited memory BFGS): Does not form/store the full inverse Hessian.\
-Does not work well for non-convex problems.\
-Usually works very well in *full batch, deterministic mode/little stochasticity* i.e. if you have a single, deterministic f(x) then L-BFGS will probably work very nicely\
-Does not transfer very well to mini-batch setting. Gives bad results. Adapting L-BFGS to large-scale, stochastic setting is an active area of research.

7.**In practice**
- Adam is a good default choice in most cases
- If you can afford to do full batch updates then try out L-BFGS (and don’t forget to disable all sources of noise)

**II.Model Ensembles**\
1.**Steps**:\
(1)Train multiple independent models. \
(2)At test time average their results\
2.**Tips and Tricks**\
(1)Instead of training independent models, use multiple snapshots of a single model during training! [SGDR](https://arxiv.org/pdf/1608.03983.pdf)\
(2)Instead of using actual parameter vector, keep a moving average of the parameter vector and use that at test time. [Polyak averaging](http://www.meyn.ece.ufl.edu/archive/spm_files/Courses/ECE555-2011/555media/poljud92.pdf)

**III.Regularization** 
*How to improve single-model performance? [check slides]*\
1.**Add term to loss** *check Lec3*\
2.**Dropout** *check slides*\
3.**Data Augmentation**\
(1)Horizontal Flips\
(2)Random crops and scales\
(3)Color Jitter: Randomize contrast and brightness\
4.**DropConnect**\
Instead of zeroing out the activations at every
forward pass, we randomly zero out some of the values of the weight matrix.\
5.**Stochastic Depth**\
randomly drop layers from the network during training\
6.**Regularization: A common pattern**\
**Training**: Add random noise\
**Testing**: Marginalize over the noise

**IV.Transfer Learning**\
1.**Transfer Learning with CNNs**\
<img src="transfer learning.png" width = "500" height = "220">
<img src="transfer learning1.png" width = "500" height = "220">\
2.**Transfer Learning with CNNs is pervasive**: use CNN pretrained on ImageNet and fine tune it or reinitialize part of it for task at hand\

3.Deep learning frameworks provide a **“Model Zoo” of pretrained models** so you don’t need to train your own\
Caffe: https://github.com/BVLC/caffe/wiki/Model-Zoo \
TensorFlow: https://github.com/tensorflow/models \
 PyTorch: https://github.com/pytorch/vision

**<font size=4>Lec 8: Deep Learning Software</font>**
- Theano / TensorFlow
- Torch / PyTorch
- Caffe / Caffe2
