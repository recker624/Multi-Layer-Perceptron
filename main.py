from sklearn.datasets import fetch_openml
import numpy as np

# as_frames=False gives us pure numpy array
mnist = fetch_openml('mnist_784', version=1 , as_frame=False)

X_raw = mnist.data
Y_raw = mnist.target

print(f'Shape of actual data : {X_raw.shape}')
print(f'Shape of the target values : {Y_raw.shape}')

def init_parameters(input_size, hidden_size, output_size):
  
  """
  Initialize weights for both the layers (First and the hidden layer)
    
  :param input_size: we have a 28x28 grid, so the no of values in grid go here
  :param hidden_size: this is the middle/hidden layer in our NN. We can decide any appropriate value here,
    this is a hyperparameter
  :output_layer: this is the final layer the NN.
  :return {W1:<val>, b1:<val>, W2:<val>, b2:<val>}
  """

  W1 = np.random.randn(input_size, hidden_size) * 0.01
  b1 = np.zeros((1,hidden_size))

  W2 = np.random.randn(hidden_size, output_size) * 0.01
  b2 = np.zeros((1, output_size))

  return {'W1' : W1, 'b1' : b1, 'W2' : W2, 'b2' : b2}
  

def relu(Z):
  """
  Computes the Rectified Linear Unit (ReLU) activation
  Role: It introduces non-linearity by zeroing out negative values
  Formula: A = max(0, Z)
  
  Args:
    Z = (numpy.ndarray) : Linear output (Z = W.X + b) 
  :param Z: Description
  
  Returns:
    A (numpy.ndarray) : Post activation values. Same shape as Z. 
  """
  return np.maximum(0, Z)

def softmax(Z):
  """
  Computes the Softmax activation function.
  Converts the raw scores (logits) into probabilities that sum to 1 for each example.
  Formula: exp(X)/Sum(exp(Z)) 
  
  Args:
        Z (numpy.ndarray): Linear output of the final layer.
                           Shape: (m_examples, output_size)
    
  Returns:
      A (numpy.ndarray): Probabilities. Shape: (m_examples, output_size).
  """
  # We subtract the max value from each row. This prevents e^1000 (Infinity).
  # Math: exp(x - C) / sum(exp(x - C)) gives the exact same result as exp(x) / sum(exp(x)).
  # axis=1 means "find max across the 10 classes for each image".
  # keepdims=True ensures the shape stays (m, 1) for broadcasting.
  Z_Shifted = Z - np.max(Z, axis=1, keepdims=True)
  
  exp_Z = np.exp(Z_Shifted)
  
  #Normalize
  # axis=1 sums across the columns 
  A = exp_Z/np.sum(exp_Z, axis=1, keepdims=True)
  
  return A
  