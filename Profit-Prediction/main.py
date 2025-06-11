import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

X = np.array([6.1101, 5.5277, 8.5186, 7.0032, 5.8598])
y = np.array([17.592, 9.1302, 13.662, 11.854, 6.8233])
m = len(y)

plt.scatter(X, y, c='red', marker='x', label='Training Data')
plt.xlabel('Population of City (10,000s)')
plt.ylabel('Profit ($10,000s)')
plt.title('Profit vs. Population')
plt.legend()
plt.grid(True)
plt.show()

X_b = np.c_[np.ones((m, 1)), X.reshape(m, 1)]

def compute_cost(X, y, theta):
    predictions = X @ theta
    errors = predictions - y
    return (1 / (2 * m)) * np.sum(errors ** 2)

def gradient_descent(X, y, theta, alpha, num_iters):
    J_history = []
    for _ in range(num_iters):
        gradient = (1 / m) * X.T @ (X @ theta - y)
        theta -= alpha * gradient
        J_history.append(compute_cost(X, y, theta))
    return theta, J_history

theta = np.zeros(2)
alpha = 0.01
iterations = 1500

theta, J_history = gradient_descent(X_b, y, theta, alpha, iterations)

print(f"Optimized parameters: {theta}")
print(f"Final cost: {J_history[-1]:.4f}")

plt.plot(range(iterations), J_history, 'b-')
plt.xlabel("Iterations")
plt.ylabel("Cost (J)")
plt.title("Cost Function Convergence")
plt.grid(True)
plt.show()

plt.scatter(X, y, c='red', marker='x', label='Training Data')
plt.plot(X, X_b @ theta, label='Linear Regression', color='blue')
plt.xlabel('Population of City (10,000s)')
plt.ylabel('Profit ($10,000s)')
plt.title('Linear Fit')
plt.legend()
plt.grid(True)
plt.show()

pop = 7
pop_input = np.array([1, pop])
predicted_profit = pop_input @ theta
print(f"Predicted profit for population {pop * 10000:.0f}: ${predicted_profit * 10000:.2f}")
