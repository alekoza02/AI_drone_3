import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("\n### END MNIST IMPORT ###\n")

x_train = np.array(x_train, dtype=np.float32)
x_test  = np.array(x_test, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int64)
y_test  = np.array(y_test, dtype=np.int64)

x_train /= 255.0
x_test  /= 255.0

def initialize_structure():
    input_hidden_layer_weights = np.random.random((28 * 28, 64)) * 2 - 1
    hidden_output_layer_weights = np.random.random((64, 10)) * 2 - 1
    hidden_layer_biases = np.random.random(64) * 2 - 1
    output_layer_biases = np.random.random(10) * 2 - 1
    
    return input_hidden_layer_weights, hidden_output_layer_weights, hidden_layer_biases, output_layer_biases


def learn(x_train, y_train):
    w0, w1, b0, b1 = initialize_structure()
    
    iterations = 0
    sample_train_len = len(x_train)
    for x, y in zip(x_train, y_train):
        x = np.ravel(x)
        z0, a1, z1, a2 = forward(x, w0, w1, b0, b1)
        d_w0, d_w1, d_b0, d_b1 = backward(x, y, z0, a1, a2, w1)
        w0, w1, b0, b1 = update(0.01, w0, w1, b0, b1, d_w0, d_w1, d_b0, d_b1)
        
        iterations += 1
        if iterations % 1000 == 0:
            print(f"Progress: {100 * iterations / sample_train_len:.2f}%")

    return w0, w1, b0, b1


def test(x_test, y_test, w0, w1, b0, b1, visualize_n_errors=-1):
    
    n_correct = 0 

    for x, y in zip(x_test, y_test):
        x_ravel = np.ravel(x)
        _, _, _, ris = forward(x_ravel, w0, w1, b0, b1)
        pred = np.argmax(ris)
        if pred == y:
            n_correct += 1
        else:
            if plot_errors > 0:
                plot_errors -= 1
                plt.imshow(x)
                plt.title(f"Expected: {y}, Predicted: {pred}")
                plt.show()

    print(f"Accuracy: {100 * n_correct / len(y_test):.2f}%")


def update(eta, o_w0, o_w1, o_b0, o_b1, d_w0, d_w1, d_b0, d_b1):
    o_w0 -= eta * d_w0
    o_w1 -= eta * d_w1
    o_b0 -= eta * d_b0
    o_b1 -= eta * d_b1
    return o_w0, o_w1, o_b0, o_b1


def forward(x, w_ih, w_ho, b_h, b_o):
    z0 = x.dot(w_ih) + b_h
    a1 = activation(z0)
    z1 = a1.dot(w_ho) + b_o
    a2 = soft_max(z1)

    return z0, a1, z1, a2


def backward(x, y_true, z0, a1, a2, w_ho):
    one_hot = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    one_hot[y_true] = 1

    delta1 = a2 - one_hot                # (10,)
    d_w1 = np.outer(a1, delta1)           # (64,10)
    d_b1 = delta1                         # (10,)

    delta0 = w_ho.dot(delta1) * der_activation(z0)   # (64,)
    d_w0 = np.outer(x, delta0)            # (784,64)
    d_b0 = delta0                         # (64,)

    return d_w0, d_w1, d_b0, d_b1


def activation(vector):
    # ReLU
    return np.maximum(0, vector)
    

def der_activation(vector):
    # Derivative of ReLU (0 if x < 0 and 1 if x > 0)
    return vector > 0


def soft_max(z):
    # Avoid overflow
    z_shifted = z - np.max(z)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)


def der_cost_function(y_pred, y_true_vector):
    return y_pred - y_true_vector


def cost_function(y_pred, y_true_vector):
    # Not used, but useful for debugging
    return - np.sum(y_true_vector * np.log(y_pred))


def predict(y_pred):
    # Not used, but useful for debugging
    return f"This is {np.argmax(y_pred) + 1}!"


test(x_test, y_test, *learn(x_train, y_train), visualize_n_errors=5)