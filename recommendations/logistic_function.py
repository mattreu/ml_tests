import numpy as np
import matplotlib.pyplot as plt

def logistic_function(x):
    return 1 / (1 + np.exp(-x))

x_values = np.linspace(-5, 5, 200)
y_values = logistic_function(x_values)

plt.figure(figsize=(7, 3.75))
plt.tight_layout()
plt.plot(x_values, y_values, label=r'$f(x) = \frac{1}{1 + e^{-x}}$')
plt.title('Funkcja logistyczna')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.xlim(left=-5, right=5)
plt.yticks(np.arange(0,1.25,0.25))
plt.axhline(0, color='black', linewidth=0.75)
plt.axvline(0, color='black', linewidth=0.75)
plt.legend()
plt.grid(True)
plt.show()