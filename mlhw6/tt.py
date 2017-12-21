import numpy as np
from matplotlib import pyplot as plt

vis_x=[1,2,3,4]
vis_y=[1,2,3,4]
y=[1,4,8,7]
cm= plt.cm.get_cmap('RdYlBu')
sc= plt.scatter(vis_x, vis_y, c=y, cmap=cm)
plt.colorbar(sc)
plt.show()

