# Handwritten Digit Classifier
This is an Implementation of Digit Classifier using Neural Networks.

## What are Neural Networks ?
An Artificial Neural Network (ANN) is an information processing paradigm that is inspired by the way biological nervous systems, such as the brain, process information. An ANN is configured for a specific application, such as pattern recognition or data classification, through a learning process. Learning in biological systems involves adjustments to the synaptic connections that exist between the neurones. This is true of ANNs as well.
Mathematically ,consider this figure
![alt text](https://camo.githubusercontent.com/03263c81130b6b49ed681422520d0fa101d30377/687474703a2f2f692e696d6775722e636f6d2f644f6b543959392e706e67 "Neuron")
-Neural networks are formed from large no. of sigmoid neurons.The name follows from a mathemaatical function sigmoid(z)=1.0/(1.0+exp(-z)).
-B is called the bias
-sigmoid function is show below
![alt text](data:image/png;base64,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 "sigmoid function")
-w0,w1,w2...wm are the weights and x0 ,x1,x2 ... xm is is input fed to the neuron.
-Output of neuron is calculated as output a=sigmoid(z+b) where z=x0*w0+x1*w1+x2*w2+...xm*wm here m=2

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

