# Handwritten Digit Classifier
This is an Implementation of Digit Classifier using Neural Networks.

## What are Neural Networks ?
An Artificial Neural Network (ANN) is an information processing paradigm that is inspired by the way biological nervous systems, such as the brain, process information. An ANN is configured for a specific application, such as pattern recognition or data classification, through a learning process. Learning in biological systems involves adjustments to the synaptic connections that exist between the neurones. This is true of ANNs as well.

Mathematically ,consider this figure
![alt text](https://camo.githubusercontent.com/03263c81130b6b49ed681422520d0fa101d30377/687474703a2f2f692e696d6775722e636f6d2f644f6b543959392e706e67 "Neuron")

  - Neural networks are formed from large no. of sigmoid neurons.The name follows from a mathemaatical function sigmoid(z)=1.0/(1.0+exp(-z)).
 - B is called the bias
 - sigmoid function is show below

![alt text](https://ml4a.github.io/images/figures/sigmoid.png "sigmoid function")

- w0,w1,w2...wm are the weights and x0 ,x1,x2 ... xm is is input fed to the neuron.
- Output of neuron is calculated as output a=sigmoid(z+b) where z=x0*w0+x1*w1+x2*w2+...xm*wm here m=2

## Why use Neural Networks ?
Neural networks, with their remarkable ability to derive meaning from complicated or imprecise data, can be used to extract patterns and detect trends that are too complex to be noticed by either humans or other computer techniques. A trained neural network can be thought of as an "expert" in the category of information it has been given to analyse. This expert can then be used to provide projections given new situations of interest and answer "what if" questions.

## Naming and Indexing
![alt text](http://mlexplore.org/images/feedfwd_nn.jpg "Neural Network")
### Layers
This network consists of three layers :
  - First Layer : Also called the input layer consists of 4 neurons to which 4 inputs are fed.This layer has no bias associated with it.
  - Hidden layer : All layers except the first and the last one come under this category.This networks contains 1 hidden layer which in turn contains 4 neurons.
  - Output Layer : Result is the output of this layer.Ouput of the hidden layer along with th e appropriate weights is fed as input toh the outer layer.

### Weights
- Weights in this neural network implementation are a list of matrices (numpy.ndarrays). weights[l] is a matrix of weights entering the l+1th layer of the network (0 based indexing).
- Weight connecting Layer l of size S(l) and layer l+1 say of s(l+1) form a matrix of dimension S(l+1)*S(l) adn denoted by weight(l).
- For input layer there is no meaning of weights.
- So in the above given code weight is basically a list of numpy array where each element in the list is a matrix representing weights connecting that layer and the previous one.

### Bias
- A bias is associated with each neuron and so it forms a 1d array numpy.array for a particular layer.
- The input layer has no bias associated with it.
- In implementation bias(l) is the numpy.array containing bias values corresponding to each neuron in (l+1)th layer (0 based indexing).

### Activation 
- Activation of a neuron is its output.
- Activation of input layer is the inputs we feed the network with.
- Activation of layer l is fed to layer l+1 to compute its activation using corresponding weights and bias.

