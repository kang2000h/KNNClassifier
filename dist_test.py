import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


f = lambda x : x
g = lambda x : np.sqrt(x)

x = np.arange(-5, 5, 0.5)
y = np.arange(-5, 5, 0.5)
xx, yy = np.meshgrid(x, y, sparse=False)
x = xx.flatten()
y = yy.flatten()
print("x", xx.shape)
print("y", yy.shape)
l1 = f(np.abs(x)+np.abs(y))
print("l1", l1.shape)
l2 = g(x**2+y**2)
print("l2", l2.shape)
# plt.xticks(())
# plt.yticks(())

# To getter a better understanding of interaction of the dimensions
# plot the first three PCA dimensions
fig = plt.figure(1, figsize=(8, 6))
# ax = Axes3D(fig, elev=-150, azim=110)
ax = Axes3D(fig)
ax.scatter(x, y, l1,
           cmap=plt.cm.Set1, edgecolor='k', s=40, label='L1')
ax.scatter(x, y, l2,
           cmap=plt.cm.Set1, edgecolor='k', s=40, label='L2')
plt.legend()
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.set_zlim(l1.min(), l1.max())

ax.set_title("L1 & L2")
ax.set_xlabel("X")
# ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Y")
# ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Z")
# ax.w_zaxis.set_ticklabels([])

plt.show()
#plt.clf()