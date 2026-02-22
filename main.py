from sklearn.datasets import fetch_openml
import numpy as np

# as_frames=False gives us pure numpy array
mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X_raw = mnist.data
Y_raw = mnist.target

print(f"Shape of actual data : {X_raw.shape}")
print(f"Shape of the target values : {Y_raw.shape}")


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
        weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        biases_input_hidden = np.zeros((1, hidden_size))

        weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        biases_hidden_output = np.zeros((1, output_size))
    except RuntimeError as err:
        print(f"Initilization Error: {err}")

    return {
        "weights_input_hidden": weights_input_hidden,
        "biases_input_hidden": biases_input_hidden, 
        "weights_hidden_output": weights_hidden_output, 
        "biases_hidden_output": biases_hidden_output
        }


def relu(linear_output):
    """
    Computes the Rectified Linear Unit (ReLU) activation
    Role: It introduces non-linearity by zeroing out negative values
    Formula: A = max(0, linear_output)

    Args:
      linear_output = (numpy.ndarray) : Linear output (Z = Weights.X + b)
    :param linear_output: Description

    Returns:
      A (numpy.ndarray) : Post activation values. Same shape as linear_output.
    """
    return np.maximum(0, linear_output)


def softmax(linear_output):
    """
    Computes the Softmax activation function.
    Converts the raw scores (logits) into probabilities that sum to 1 for each example.
    Formula: exp(X)/Sum(exp(Z))

    Args:
          linear_output (numpy.ndarray): Linear output of the final layer.
                             Shape: (m_examples, output_size)

    Returns:
        probabilities (numpy.ndarray): Probabilities. Shape: (m_examples, output_size).
    """
    # We subtract the max value from each row. This prevents e^1000 (Infinity).
    # Math: exp(x - C) / sum(exp(x - C)) gives the exact same result as exp(x) / sum(exp(x)).
    # axis=1 means "find max across the 10 classes for each image".
    # keepdims=True ensures the shape stays (m, 1) for broadcasting.
    try:
        stable_linear_output = linear_output - np.max(linear_output, axis=1, keepdims=True)

        exponentiated_values = np.exp(stable_linear_output)

        # Normalize
        # axis=1 sums across the columns
        probabilities = exponentiated_values / np.sum(exponentiated_values, axis=1, keepdims=True)
    except RuntimeError as err:
        print(f"Softmax Runtime error : {err}")

    return probabilities


def forward_propagation(input_data, parameters):
    """
    Argument:
    input_data -- Input data of shape (m, input_size)
    parameters -- python dictionary containing your parameters "weights_input_hidden", "biases_input_hidden", etc.

    Returns:
    probabilities -- The output of the second activation (probabilities)
    cache -- a dictionary containing "linear_output_hidden", "activation_hidden", etc.
             (We need these stored for the backward pass!)
    """
    try:
        weights_input_hidden = parameters["weights_input_hidden"]
        biases_input_hidden = parameters["biases_input_hidden"]
        weights_hidden_output = parameters["weights_hidden_output"]
        biases_hidden_output = parameters["biases_hidden_output"]

        # First layer
        # Linear step
        linear_output_hidden = np.dot(input_data, weights_input_hidden) + biases_input_hidden

        activation_hidden = relu(linear_output_hidden)  # Actual activation

        # Hidden layer
        # Linear step
        linear_output_final = np.dot(activation_hidden, weights_hidden_output) + biases_hidden_output

        probabilities = softmax(linear_output_final)  # Final activation function to predict the actual numbers

        # store everything in a dictionary (Cache)
        cache = {
            "linear_output_hidden": linear_output_hidden, 
            "activation_hidden": activation_hidden, 
            "linear_output_final": linear_output_final, 
            "probabilities": probabilities
            }
        print(cache)
    except RuntimeError as err:
        print(f"forward propogation error : {err}")
    return probabilities, cache


def categorical_cross_entropy(
    probabilities,
    true_labels,
):
    """
    Computes the Categorical Cross-Entropy Loss.

      Role:
      This measures how well the model's probability distribution (probabilities)
      matches the true distribution (true_labels). It penalizes the model more
      heavily when it is "confident and wrong."

      Args:
          probabilities (numpy.ndarray): The output of the softmax activation (probabilities).
                             Shape: (m_examples, 10)
          true_labels (numpy.ndarray): The ground truth labels (integers 0-9).
                            Shape: (Y_raw,)

      Returns:
          loss (float): The average cross-entropy loss across all m_examples.

      Formula: -(1/m) * sum(Y_one_hot * log(probabilities + epsilon))

    """
    try:
        # One hot encode the true labels
        one_hot_labels = np.zeros((true_labels.size, 10), dtype=int)
        one_hot_labels[np.arange(true_labels.size), true_labels] = 1

        loss_matrix = one_hot_labels * np.log(probabilities + 1e-8)
        average_loss = -(1 / true_labels.size) * np.sum(loss_matrix)

        return average_loss

    except Exception as e:
        print(f"Exception in categorical_cross_entropy: {e}")

def relu_derivative(linear_output):
    """
    Computes the derivative of the ReLU function.
    
    Role: Serves as a "gatekeeper" for gradients.
    - If linear_output > 0, the slope is 1 (let the error pass through).
    - If linear_output <= 0, the slope is 0 (block the error, this neuron was dead).
    
    Args:
        linear_output (numpy.ndarray): The input used during forward propagation.
        
    Returns:
        gradient (numpy.ndarray): A mask of 1s and 0s.
    """
    return linear_output>0

def backward_propagation(input_data, true_labels, cache, parameters):
    """
    Computes the gradients (derivatives) for all parameters.
    
    Role: Calculates how much each weight contributed to the error.
    
    Arguments:
    input_data -- Input data (m, input_size)
    true_labels -- True labels (m,) - Integer list (0-9)
    cache -- The dictionary from forward_prop containing (linear_output_hidden, activation_hidden, linear_output_final, probabilities)
    parameters -- The dictionary containing (weights_input_hidden, biases_input_hidden, weights_hidden_output, biases_hidden_output)
    
    Returns:
    grads -- Dictionary containing gradients
    """
    m = input_data.shape[0]
    
    linear_output_hidden = cache["linear_output_hidden"]
    activation_hidden = cache["activation_hidden"]
    probabilities = cache["probabilities"]
    weights_hidden_output = parameters["weights_hidden_output"]
    
    one_hot_labels = np.zeros((m, 10))
    one_hot_labels[np.arange(m), true_labels] = 1
    
    error_output_layer = probabilities - one_hot_labels
    
    gradient_weights_hidden_output = (1/m) * np.dot(activation_hidden.T, error_output_layer)
    
    gradient_biases_hidden_output = (1/m) * np.sum(error_output_layer, axis=0, keepdims=True)
    
    # calculate the raw blame for the activation values. 
    error_hidden_layer = np.dot(error_output_layer, weights_hidden_output.T)
    delta_linear_hidden = error_hidden_layer * relu_derivative(linear_output_hidden)
    
    