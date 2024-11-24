import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle, FancyArrowPatch
import networkx as nx

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr
        self.activation_fn = activation
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
        
        self.activations = {}
        self.gradients = {}
        
        if activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative

    def forward(self, X):
        self.activations['z1'] = np.dot(X, self.W1) + self.b1
        self.activations['a1'] = self.activation(self.activations['z1'])
        self.activations['z2'] = np.dot(self.activations['a1'], self.W2) + self.b2
        self.activations['a2'] = np.tanh(self.activations['z2'])
        return self.activations['a2']

    def backward(self, X, y):
        m = X.shape[0]
        delta2 = self.activations['a2'] - y
        self.gradients['W2'] = np.dot(self.activations['a1'].T, delta2) / m
        self.gradients['b2'] = np.sum(delta2, axis=0, keepdims=True) / m
        
        delta1 = np.dot(delta2, self.W2.T) * self.activation_derivative(self.activations['z1'])
        self.gradients['W1'] = np.dot(X.T, delta1) / m
        self.gradients['b1'] = np.sum(delta1, axis=0, keepdims=True) / m
        
        self.W2 -= self.lr * self.gradients['W2']
        self.b2 -= self.lr * self.gradients['b2']
        self.W1 -= self.lr * self.gradients['W1']
        self.b1 -= self.lr * self.gradients['b1']

def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1
    y = y.reshape(-1, 1)
    return X, y

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)
    
    # Hidden Space Plot (Left)
    hidden_features = mlp.activations['a1']
    
    # Add visualization of distorted input space
    xx = np.linspace(-3, 3, 20)
    yy = np.linspace(-3, 3, 20)
    XX, YY = np.meshgrid(xx, yy)
    grid_points = np.column_stack((XX.ravel(), YY.ravel()))
    
    # Transform grid points through the first layer
    Z1 = np.dot(grid_points, mlp.W1) + mlp.b1
    transformed_grid = mlp.activation(Z1)
    
    # Plot the transformed grid as a wireframe
    ax_hidden.plot_wireframe(transformed_grid[:, 0].reshape(20, 20),
                           transformed_grid[:, 1].reshape(20, 20),
                           transformed_grid[:, 2].reshape(20, 20),
                           alpha=0.1, color='gray')
    
    # Plot original features in hidden space
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], 
                     c=y.ravel(), cmap='bwr', alpha=0.7)
    
    # Add decision boundary surface in hidden space
    xx = np.linspace(-1, 1, 20)
    yy = np.linspace(-1, 1, 20)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = (-mlp.W2[0] * XX - mlp.W2[1] * YY - mlp.b2[0]) / mlp.W2[2]
    surf = ax_hidden.plot_surface(XX, YY, ZZ, alpha=0.2, color='tan')
    
    ax_hidden.set_title('Hidden Space at Step {}'.format(frame * 10))
    ax_hidden.set_xlim(-1, 1)
    ax_hidden.set_ylim(-1, 1)
    ax_hidden.set_zlim(-1, 1)
    
    # Input Space Plot (Middle)
    x_min, x_max = -3, 3
    y_min, y_max = -3, 3
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    Z = mlp.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax_input.contourf(xx, yy, Z, levels=[-1, 0, 1], colors=['blue', 'red'], alpha=0.3)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr')
    ax_input.set_title('Input Space at Step {}'.format(frame * 10))
    ax_input.set_xlim(x_min, x_max)
    ax_input.set_ylim(y_min, y_max)
    
    # Gradient Visualization (Right)
    # Create network layout
    pos = {
        'x1': (0.0, 0.0), 'x2': (0.0, 1.0),
        'h1': (0.5, 0.0), 'h2': (0.5, 0.5), 'h3': (0.5, 1.0),
        'y': (1.0, 0.0)
    }
    
    # Draw nodes
    node_size = 1000
    nodes = ['x1', 'x2', 'h1', 'h2', 'h3', 'y']
    for node in nodes:
        circle = Circle(pos[node], 0.05, color='blue')
        ax_gradient.add_patch(circle)
    
    # Draw edges with thickness based on weights and gradients
    for i, start in enumerate(['x1', 'x2']):
        for j, end in enumerate(['h1', 'h2', 'h3']):
            weight = mlp.W1[i, j]
            gradient = mlp.gradients['W1'][i, j]
            line_width = abs(gradient) * 50
            color = 'purple' if weight > 0 else 'pink'
            ax_gradient.plot([pos[start][0], pos[end][0]], 
                           [pos[start][1], pos[end][1]], 
                           color=color, linewidth=line_width, alpha=0.6)
    
    # Draw edges from hidden to output
    for i, start in enumerate(['h1', 'h2', 'h3']):
        weight = mlp.W2[i, 0]
        gradient = mlp.gradients['W2'][i, 0]
        line_width = abs(gradient) * 50
        color = 'purple' if weight > 0 else 'pink'
        ax_gradient.plot([pos[start][0], pos['y'][0]], 
                        [pos[start][1], pos['y'][1]], 
                        color=color, linewidth=line_width, alpha=0.6)
    
    # Add labels
    for node, position in pos.items():
        ax_gradient.text(position[0], position[1]+0.1, node)
    
    ax_gradient.set_title('Gradients at Step {}'.format(frame * 10))
    ax_gradient.set_xlim(-0.2, 1.2)
    ax_gradient.set_ylim(-0.2, 1.2)
    ax_gradient.axis('equal')

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, 
                                   ax_gradient=ax_gradient, X=X, y=y), 
                       frames=step_num//10, repeat=False)

    # Save animation
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)