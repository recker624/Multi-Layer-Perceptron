## Multi-Layer Perceptron on MNIST

This repository is for a **simple multi-layer perceptron (MLP)** with **one hidden layer** trained on the **MNIST** handwritten digits dataset.

The goal is to implement the core pieces of a small neural network from scratch using **NumPy**, while using **scikit-learn** only for loading the MNIST data.

---

### Project structure

- `main.py` – loads the MNIST dataset and contains the starting point for the MLP implementation (parameter initialization, etc.).
- `requirements.txt` – pinned Python package dependencies.

---

### Setup

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

### Usage

Run the main script:

```bash
python main.py
```

At this stage, the script:
- Downloads and loads the MNIST dataset using `fetch_openml`.
- Prints the shapes of the input data and labels.
- Provides a starting point for initializing the MLP parameters (weights and biases) for a 1-hidden-layer network.

You can extend `main.py` to add:
- Forward pass through the hidden and output layers
- Loss computation (e.g., cross-entropy)
- Backpropagation and parameter updates
- Training loop and evaluation on a test set

---

### Next steps / ideas

- Implement the full training pipeline for the 1-hidden-layer MLP.
- Experiment with different hidden layer sizes and activation functions.
- Plot training / validation accuracy and loss over epochs.

