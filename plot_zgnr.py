import numpy as np
from time import time
import matplotlib.pyplot as plt
from HF_spinspliting import *

fig = plt.figure(dpi = 200)
ax = fig.gca()

Lx = 10
Ly = 4
scatter_list = []

for i in hopping_list(Lx, Ly):
    x1, y1 = coordinate(i[0], Lx, Ly)
    x2, y2 = coordinate(i[1], Lx, Ly)
    ax.plot([x1, x2], [y1, y2], color = 'k')

site_color = ['k', 'grey']
for i in range(Lx * Ly):
    x, y = coordinate(i, Lx, Ly)
    color_ind = np.mod(i//Lx, 2)
    ax.scatter(x, y, color = site_color[color_ind])

ax.set_aspect(1)
plt.savefig('./geometry/geometry_Lx10Ly4.png', bbox_inches='tight')
# plt.show()
