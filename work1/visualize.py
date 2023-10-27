from sklearn.datasets import make_moons
import matplotlib.pyplot as plt     
    
    
x, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

plt.scatter(x[:, 0], x[:, 1], c=y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Half Moon Dataset')
plt.savefig('half_moon.png')