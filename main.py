from sklearn.datasets import fetch_openml
import numpy as np

# as_frames=False gives us pure numpy array
mnist = fetch_openml('mnist_784', version=1 , as_frame=False)

X_raw = mnist.data
Y_raw = mnist.target

print(f'Shape of actual data : {X_raw.shape}')
print(f'Shape of the target values : {Y_raw.shape}')

# TODO: write a init_parameters function to initialize the W1 and W2 for both the layers. 
def init_parameters(input_size, hidden_size, output_size):
  """
  Docstring for init_parameters
  
  :param input_size: we have a 28x28 grid, so the no of values in grid go here
  :param hidden_size: this is the middle/hidden layer in our NN. We can decide any appropriate value here,
    this is a hyperparameter
  :output_layer: this is the final layer in our NN. In our case this is 10
  :return {W1:<val>, b1:<val>, W2:<val>, b2:<val>}
  """

  W1 = np.random.randn(input_size, hidden_size) * 0.01
  b1 = np.zeros(1,hidden_size)

  W2 = np.random.randn(hidden_size, output_size) * 0.01
  b2 = np.zeros(1, output_size)

  return {'W1' : W1, 'b1' : b1, 'W2' : W2, 'b2' : b2}
  