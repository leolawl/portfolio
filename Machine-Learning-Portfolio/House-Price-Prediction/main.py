import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


X = np.array([2104, 1600, 2400, 1416, 3000])
y = np.array([399.9, 329.9, 369.0, 232.0, 539.9])
m = len(y)

plt.scatter(X, y, c='red', marker='x', label='Training Data')
plt.xlabel('Size (sqft)')
plt.ylabel('Price ($1000s)')
plt.title('House Prices vs. Size')
plt.grid(True)
plt.legend()
plt.show()

mean_X = np.mean(X)
std_X = np.std(X)
X_scaled = ((X - mean_X) / std_X).reshape(m, 1)
X_b = np.hstack([np.ones((m, 1)), X_scaled])

def compute_cost(X, y, theta):
    errors = X @ theta - y
    return (1 / (2 * m)) * np.sum(errors ** 2)

def gradient_descent(X, y, theta, alpha, num_iters):
    J_history = []
    for _ in range(num_iters):
        theta -= (alpha / m) * X.T @ (X @ theta - y)
        J_history.append(compute_cost(X, y, theta))
    return theta, J_history

theta = np.zeros(2)
alpha = 0.1
iterations = 100
theta, J_history = gradient_descent(X_b, y, theta, alpha, iterations)

print(f"Optimized parameters: {theta}")
print(f"Final cost: {J_history[-1]:.4f}")

plt.plot(range(iterations), J_history, 'b-')
plt.xlabel("Iterations")
plt.ylabel("Cost (J)")
plt.title("Cost Function Convergence")
plt.grid(True)
plt.show()

x_vals = np.linspace(min(X), max(X), 100)
x_vals_scaled = (x_vals - mean_X) / std_X
x_vals_b = np.c_[np.ones_like(x_vals_scaled), x_vals_scaled]
y_vals = x_vals_b @ theta

plt.scatter(X, y, c='red', marker='x', label='Training Data')
plt.plot(x_vals, y_vals, 'b-', label='Model Prediction')
plt.xlabel('Size (sqft)')
plt.ylabel('Price ($1000s)')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid(True)
plt.show()

def predict_price(size, theta, mean, std):
    size_scaled = (size - mean) / std
    x_input = np.array([1, size_scaled])
    return x_input @ theta

new_size = 2000
predicted_price = predict_price(new_size, theta, mean_X, std_X)
print(f"Predicted price for a {new_size} sqft house: ${predicted_price * 1000:.2f}")
