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

  try:
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1,hidden_size))

    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
  except RuntimeError as err:
    print(f"Initilization Error: {err}")

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
  try:
    Z_Shifted = Z - np.max(Z, axis=1, keepdims=True)
    
    exp_Z = np.exp(Z_Shifted)
    
    #Normalize
    # axis=1 sums across the columns 
    A = exp_Z/np.sum(exp_Z, axis=1, keepdims=True)
  except RuntimeError as err:
    print(f"Softmax Runtime error : {err}")
    
  return A
  
# TODO : Add forward propogation function
def forward_propogation(X, parameters):
  """
    Argument:
    X -- Input data of shape (m, input_size)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2"
    
    Returns:
    A2 -- The output of the second activation (probabilities)
    cache -- a dictionary containing "Z1", "A1", "Z2", "A2"
             (We need these stored for the backward pass!)
  """
  try:
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    # First layer
    # Linear step
    Z1 = np.dot(X, W1) + b1
    
    A1 = relu(Z1) # Actual activation
    
    # Hidden layer
    # Linear step
    Z2 = np.dot(A1, W2) + b2
    
    A2 = softmax(Z2) # Final activation function to predict the actual numbers
    
    # store everything in a dictionary (Cache)
    cache = { "Z1": Z1, "A1": A1, "Z2": Z2, "A2":A2 }  
    print(cache)
  except RuntimeError as err:
    print(f"forward propogation error : {err}")
  return A2, cache
 
  
def categorical_cross_entropy(A2, Y_true, ):
  """
   Computes the Categorical Cross-Entropy Loss.

     Role:
     This measures how well the model's probability distribution (A2)
     matches the true distribution (Y). It penalizes the model more
     heavily when it is "confident and wrong."

     Args:
         A2 (numpy.ndarray): The output of the softmax activation (probabilities).
                            Shape: (m_examples, 10)
         Y (numpy.ndarray): The ground truth labels (integers 0-9).
                           Shape: (Y_raw,)

     Returns:
         loss (float): The average cross-entropy loss across all m_examples.

     Formula: -(1/m) * sum(Y_one_hot * log(A2 + epsilon))
  
  """
  try:
    # One hot encode the true labels
    encoded_arr = np.zeros((Y_true.size, 10), dtype=int)
    encoded_arr[np.arange(Y_true.size), Y_true] = 1
    
    loss_matrix = encoded_arr * np.log(A2 + 1e-8)
    average_loss = -(1/Y_true.size) * np.sum(loss_matrix)
    
    return average_loss
    
  except Exception as e:
    print(f"Exception in categorical_cross_entropy: {e}")
    