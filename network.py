
import random
import os
import json
import numpy as np

from definitions import sigmoid,sigmoid_prime

class Network(object):

   def __init__ (self,sizes): 
      """Intialises a neural network with num_layers defining total layers.
      Weights are randomly intialised by np.random.rand() func which produces 
      values sampled from gausain distribution with mean 0 and variance 1
      Similarly bias are initialised.Layer 0- the input layer does not have 
      any bias."""
      self.num_layers=len(sizes)
      self.sizes=sizes
      self.weights=[np.random.randn(y,x)/np.sqrt(x) for x,y in zip(sizes[:-1],sizes[1:]) ]
     # np.random.rand(y,x) gives a matrix y*x  
     
      self.bias=[np.random.randn(y,1) for y in sizes[1:]]
     

   def compute(self,training_data,epochs,mini_batch_size,eta,test_data=None):
      """Trains the neural network using Mini batch gradient descent algorithm.
      eta is the learning rate.If test_data is provided calculates accuracy after 
      each epoch"""
      if test_data: test_n=len(test_data)
    
      n=len(training_data)
      
      for i in range(epochs):         
         random.shuffle(training_data)
         batches=[training_data[m:m+mini_batch_size] for m in range(0,n,mini_batch_size)]
         # created all mini_batches from shuffled training_data
         for mini_batch in batches:
               self.update(mini_batch,eta)
         
         if test_data:
              print "Epoch {0}: {1} / {2}".format(
                    i, self.evaluate(test_data), test_n)
         else:
                print "Epoch {0} complete".format(i)  
   

   def feedforward(self, a):
      """Gives the output of the network if input is a."""
      for (b,w) in zip(self.bias, self.weights):
         a = sigmoid(np.dot(w, a)+b)
      return a  
     
  
   def update(self,mini_batch,eta):
      """Updates the weight and bias by using each tuple(x,y)
      from the given mini_batch."""     
      nabla_w=[np.zeros(w.shape) for w in self.weights]
      nabla_b=[np.zeros(b.shape) for b in self.bias]
   
      for (x,y) in mini_batch:
           delta_b,delta_w=self.back_propagation(x,y)
           nabla_b=[temp1+temp2 for temp1,temp2 in zip(nabla_b,delta_b)]   
           nabla_w=[temp1+temp2 for temp1,temp2 in zip(nabla_w,delta_w)]  

      self.weights=[weights -(eta/len(mini_batch))*nabla_w for weights, nabla_w in zip(self.weights,nabla_w)]
      self.bias=[bias-(eta/len(mini_batch))*nabla_b for bias,nabla_b in zip(self.bias,nabla_b)]
    

   def back_propagation(self,x,y):
      """Return a tuple ``(nabla_b, nabla_w)`` representing the
      gradient for the cost function C_x for each trainig example(x,y). 
      ``nabla_b`` and ``nabla_w`` are layer-by-layer lists of numpy 
      arrays, similar to ``self.biases`` and ``self.weights``."""

      nabla_b=[np.zeros(b.shape) for b in self.bias]
      nabla_w=[np.zeros(w.shape) for w in self.weights]

      activation =x
      activations =[x]
          
      zs=[]
          
      for (b,w) in zip(self.bias,self.weights):
          z=np.dot(w,activation)+b
          zs.append(z)
          activation=sigmoid(np.dot(w,activation)+b)
          activations.append(activation)
      
      delta=self.cost_derivative(activations[-1],y)*sigmoid_prime(zs[-1])
      nabla_b[-1]=delta
      nabla_w[-1]=np.dot(delta,activations[-2].transpose())

      for i in xrange(2,self.num_layers):
          z=zs[-i]
          sp=sigmoid_prime(z)
          delta=np.dot(self.weights[-i+1].transpose(),delta)*sp
          nabla_b[-i]=delta
          nabla_w[-i]=np.dot(delta,activations[-i-1].transpose())

      return (nabla_b,nabla_w)  


   def cost_derivative(self,a,b):     
      """ difference to compute delta for last layer"""
      return (a-b)


   def evaluate(self,test_data):
      """ gives the value as to how many digits our network predicited correctly"""
      result=[(np.argmax(self.feedforward(x)),y) for x,y in test_data]
      return sum(int(x==y) for x,y in result)

