# Handwritten Digit Classifier
This is an Implementation of Digit Classifier using Neural Networks.

## A bit about Machine Learning
Two definitions of Machine Learning are offered. *Arthur Samuel* described it as: "the field of study that gives computers the ability to learn without being explicitly programmed." This is an older, informal definition.
*Tom Mitchell* provides a more modern definition: "A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E."
Machine learning can be classified into two categories:
- *Supervised Learning* :In this case we are given the data set(called training_set)with input values and its corresponding outputs.Based on this set we train our model to predict new inputs. It can be classified further into two categories 
  - *Regression Problem*- Here we form a continous function and try to map inputs to the given outputs using this function.Example Given a large sample of house size with its price ,predict price with house size not in data set.
  - *Classification Problem*-Here we form a function with maps input to discrete output.Example Given a patient with a tumor, we have to predict whether the tumor is malignant or benign.This is also called called logistic regression problem.
  
 - *Unsupervised Learning*:In this case we have no labelled data by this I mean we just have inputs and no outputs associated with it. It is used to draw inferences from datasets consisting of input data without labeled responses. It is also further classified as :
   - *Clustering Problem*-Here we divide the large data set into different clusters(groups) that are somehow similar or related by different variables.
   - *Non Clustering Problem*-Here we dont have to group rather we distinguish one object from another.Example From a cocktail party we have data set of voices on microphone and we want to identify individual voices and music from a mesh of sounds at a cocktail party.

## What are Neural Networks ?
Artificial neural networks (ANNs) are biologically inspired computer programs designed to simulate the way in which the human brain processes information. ANNs gather their knowledge by detecting the patterns and relationships in data and learn (or are trained) through experience, not from programming. An ANN is formed from hundreds of single units, artificial neurons or processing elements (PE), connected with coefficients (weights), which constitute the neural structure and are organised in layers. The power of neural computations comes from connecting neurons in a network. Each PE has weighted inputs, transfer function and one output. The behavior of a neural network is determined by the transfer functions of its neurons, by the learning rule, and by the architecture itself. The weights are the adjustable parameters and, in that sense, a neural network is a parameterized system. The weighed sum of the inputs constitutes the activation of the neuron. The activation signal is passed through transfer function to produce a single output of the neuron. Transfer function introduces non-linearity to the network. During training, the inter-unit connections are optimized until the error in predictions is minimized and the network reaches the specified level of accuracy. Once the network is trained and tested it can be given new input information to predict the output. Many types of neural networks have been designed already and new ones are invented every week but all can be described by the transfer functions of their neurons, by the learning rule, and by the connection formula. ANN represents a promising modeling technique, especially for data sets having non-linear relationships which are frequently encountered in pharmaceutical processes. In terms of model specification, artificial neural networks require no knowledge of the data source but, since they often contain many weights that must be estimated, they require large training sets. In addition, ANNs can combine and incorporate both literature-based and experimental data to solve problems. The various applications of ANNs can be summarised into classification or pattern recognition, prediction and modeling. Supervised associating networks can be applied in pharmaceutical fields as an alternative to conventional response surface methodology. Unsupervised feature-extracting networks represent an alternative to principal component analysis. Non-adaptive unsupervised networks are able to reconstruct their patterns when presented with noisy samples and can be used for image recognition.

The artificial equivalent of a neuron is a node (also sometimes called neurons, but I will refer to them as nodes to avoid ambiguity) that receives a set of weighted inputs wi*ai, processes their sum with its activation function ϕ, and passes the result of the activation function to nodes further down the graph. Note that it is simpler to represent the input to our activation function as a dot product:
```sh
              ϕ(∑wiai)=ϕ(w.transpose().a)
```
Visually it looks like 
![alt text](https://deeplearning4j.org/img/perceptron_node.png "Neuron")
There are several canonical activation functions. For instance, we can use a linear activation function: 
```sh
ϕ(w,a)=w.transpose()*a
```
This is also called the identity activation function. Another example is the sigmoid activation function: 
```sh
ϕ(w,a)=1.0/(1+exp(−w.transpose()*a))
```
w is weights associated with the network and a (1D matrix) is the input fed.

We can then form a network by chaining these nodes together. Usually this is done in layers - one node layer's outputs are connected to the next layer's inputs (layers are defined below).
Our goal is to train a network using labelled data so that we can then feed it a set of inputs and it produces the appropriate outputs for unlabeled data. We can do this because we have both the input xi and the desired target output yi in the form of data pairs. Training in this case involves learning the correct edge weights to produce the target output given the input. The network and its trained weights form a function (denoted hh) that operates on input data. With the trained network, we can make predictions given any unlabeled test input.

## Why use Neural Networks ?
Neural networks, with their remarkable ability to derive meaning from complicated or imprecise data, can be used to extract patterns and detect trends that are too complex to be noticed by either humans or other computer techniques. A trained neural network can be thought of as an "expert" in the category of information it has been given to analyse. This expert can then be used to provide projections given new situations of interest and answer "what if" questions.

## Implementation 
I implemented logistic regression problem of digit classifier using neural networks.As said earlier neuron have take many canonical activation function,but for logistic regression sigmoid is an apt choice. Sigmoid function is a contionous function so any small change in weights associated with networks brings changes noticeable by using sigmoid function. Sigmoid function is shown here
![alt text](https://ml4a.github.io/images/figures/sigmoid.png "sigmoid function")

Now moving to the neuron using sigmoid function,here we have some inputs x0,x1 and x2 along with there respective weights w1,w1,w2 and a term called Bias associated with a neuron. As described above the summation of (wi*xi) is fed to the neuron and the output is calculated as sigmoid() of this summation.

![Sigmoid Neuron](http://i.imgur.com/dOkT9Y9.png)

Quick takeaway:
  - Neural networks are formed from large no. of sigmoid neurons.The name follows from a mathemaatical function sigmoid(z)=1.0/(1.0+exp(-z)).
 - B is called the bias
- w0,w1,w2...wm are the weights and x0 ,x1,x2 ... xm is is input fed to the neuron.
- Output of neuron is calculated as output a=sigmoid(z+b) where z=x0*w0+x1*w1+x2*w2+...xm*wm here m=2



### Naming and Indexing
![Imgur](http://i.imgur.com/KnhlcoN.png "Neural Network")
### Layers
The above ANN(Artifical Neuron Network) would start by computing the outputs of nodes s1 and s2 given x1,x2, and x3. Once that was complete, the ANN would next compute the outputs of nodes s3,s4 , and s5, dependent on the outputs of s1 and s2. Once that was complete, the ANN would do the final calculation of nodes y1 and y2, dependent on the outputs of nodes s3,s4 , and s5.
It is obvious from this computational flow that certain sets of nodes tend to be computed at the same time, since a different set of nodes uses their outputs as inputs. For example set {s3,s4,s5} depends on set {s1,s2}. These sets of nodes that are computed together are known as layers, and ANNs are generally thought of a series of such layers, with each layer Li dependent on previous layer L(i-1). Thus, the above graph is composed of four layers. The first layer  is called the input layer (which does not need to be computed, since it is given), while the final layer  is called the output layer. The intermediate layers are known as hidden layers, which in this case are the layers l1={s1,s2} and l2= {s3,s4,s5}, are usually numbered so that hidden layer hi corresponds to layer li. Thus, hidden layer h1={s1,s2} and hidden layer h2= {s3,s4,s5}.

### Weights
- Weights in this neural network implementation are a list of matrices (numpy.ndarrays). weights[l] is a matrix of weights entering the l+1th layer of the network (0 based indexing).
- Weight connecting Layer l of size S(l) and layer l+1 say of s(l+1) form a matrix of dimension S(l+1)*S(l) adn denoted by weight(l).
- For input layer there is no meaning of weights.
- So in the above given code weight is basically a list of numpy array where each element in the list is a matrix representing weights connecting that layer and the previous one.
- Here is the weight matrix ,mapping from first layer to second one.Similarly we can get other mappins as well.
```
weights[0] = |¯[[Wx1s1, Wx2s1,Wx3s1]  ¯|
             |_  [Wx1s2, Wx2s2,Wx3s2] _|             
```

### Bias
- A bias is associated with each neuron and so it forms a 1d array numpy.array for a particular layer.
- The input layer has no bias associated with it.
- In implementation bias(l) is the numpy.array containing bias values corresponding to each neuron in (l+1)th layer (0 based indexing).

### Activation 
- Activation of a neuron is its output.
- Activation of input layer is the inputs we feed the network with.
- Activation of layer l is fed to layer l+1 to compute its activation using corresponding weights and bias.

## Training The Network
The ANN can now calculate some function F(theta,x) that depends on the values of the individual nodes' weight vectors and biases, which together are known as the ANN's parameters,theta . The logical next step is to determine how to alter those biases and weight vectors so that the ANN computes known values of the function. That is, given a series of input-output pairs (xi,yi), how can the weight vectors and biases be altered such that f(theta,xi) is close to yi for all i ?
### Choosing an Error Function
The typical way to do this is define an error function E over the set of pairs X={xi,yi for all i } such that E(X,theta) is small when F(theta,xi) is close to yi for all i. Common choices for E are the mean squared error (MSE) in the case of regression problems and the cross entropy in the case of classification problems. Thus, training the ANN reduces to minimizing the error  with respect to the parameters (since  is fixed). For example, for the mean square error function, given two input-output pairs  X={(x1,y1),(x2,y2)} and an ANN with parameters theta that outputs F(theta,x) for input x, the error function  is E(X,theta)

![Imgur](http://i.imgur.com/jkMXStu.png)

### Gradient Descent
Since the error function E(X,theta)  defines a fairly complex function (it is a function of the output of the ANN, which is a composition of many nonlinear functions), finding the minimum analytically is generally impossible. Luckily, there exists a general method for minimizing differentiable functions called gradient descent. Basically, gradient descent finds the gradient of a function f at a particular value  (for an ANN, that value will be the parameters theta) and then updates that value by moving (or stepping) in the direction of the negative of the gradient. Generally speaking (it depends on the size of the step ), this will find a nearby value x'=x-n*(derivative of f) for which f(x') < f(x).This process repeats until a local minimum is found, or the gradient sufficiently converges (i.e. becomes smaller than some threshold). Learning for an ANN typically starts with a random initialization of the parameters (the weight vectors and biases) followed by successive updates to those parameters based on gradient descent until the error function E(X,theta) converges.

A major advantage of gradient descent is that it can be used for online learning, since the parameters are not solved in one calculation but are instead gradually improved by moving in the direction of the negative gradient. Thus, if input-output pairs are arriving in a sequential fashion, the ANN can perform gradient descent on one input-output pair for a certain number of steps, and then do the same once the next input-output pair arrives. For an appropriate choice of step size n(eta), this approach can yield results similar to gradient descent on the entire dataset X (known as batch learning).

![alt text](http://hi.csdn.net/attachment/201202/19/0_13296568341m44.gif "gradient Descent")

As it is clear from the figure that because gradient descent is a local method (the step direction is determined by the gradient at a single point), it can only find local minima.Since if start from different red buldges we end up at different minima. While this is generally a significant problem for most optimization applications, recent research has suggested that finding local minima is not actually an issue for ANNs, since the vast majority of local minima are evenly distributed and similar in magnitude for large ANNs.

### Backpropagation
For a long time, calculating the gradient for ANNs was thought to be mathematically intractable, since ANNs can have large numbers of nodes and very many layers, making the error function E(X,theta) highly nonlinear. However, in the mid-1980s, computer scientists were able to derive a method for calculating the gradient with respect to an ANN's parameters, known as backpropagation, or "backpropagation by errors". The method works for both feedforward neural networks (for which it was originally designed) as well as for recurrent neural networks, in which case it is called backpropagation through time, or BPTT. The discovery of this method brought about a renaissance in artificial neural network research, as training non-trivial ANNs had finally become feasible.

## Summing Up !
The power of neural network can be seen by executing the code network.py. I got an accuracy of about 96% by using this tool of machine learning.This high accuracy may exceed the human accuracy for certain difficult digit recognition.The code can be executed by launching the python shell in the source code directory and then following this:

1. import data_load 
2. training_data,validation_data_test_data = data_load.load_data() 
3. import network
4. net=network.Network([784,30,10])
5. net.compute(training_data, 30, 10, 3.0, test_data=test_data)

The code starts executing and gives the accuracy after each epoch(for now you may refer this as "one cycle of learning").Here's the snapshot
![Imgur](http://i.imgur.com/B0cgHWw.png "Command Prompt")
