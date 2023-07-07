import numpy as np
import matplotlib.pyplot as plt
from HF_spinspliting import *


def compute_qA(state, Lx, Ly):
    '''
    Compute probability density distributed on A-carbon sites of a given 
    state.
    '''
    prob_density = state**2
    prob_Asites = np.take(prob_density, list_Asite(Lx, Ly))
    return np.sum(prob_Asites)

# 1. Symmetry protected phase
t = 1
U = 1
gamma = 0
dope = 0

Lx = 100
Ly = 8

import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure(dpi=300)
ax = fig.add_subplot(1,2,1)


# Disorder realizations
realization = 1
list_qa_up = []
list_qa_dn = []
list_val_up = []
list_val_dn = []
for i in range(realization):
    val_up, vec_up, val_dn, vec_dn = HF_vec(t, U, gamma, dope, Lx, Ly)
    list_qa_up.append([compute_qA(state, Lx, Ly) for state in vec_up.transpose()])
    list_qa_dn.append([compute_qA(state, Lx, Ly) for state in vec_dn.transpose()])
    list_val_up.append(val_up)
    list_val_dn.append(val_dn)


# because Mott-gap = 0.2 when U = 1
ax.scatter(list_qa_up, np.array(list_val_up)/0.2, s=0.5, fc='none', ec='r')
ax.scatter(list_qa_dn, np.array(list_val_dn)/0.2, s=0.5, fc='none', ec='b')

ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_xlabel(r'$q_A$', fontsize = 20)
ax.set_ylabel(r'$E/(\Delta/2)$', fontsize = 20)
ax.set_xlim(0, 1)

boudary = 1
ax.set_ylim([-boudary, boudary])
ax.set_aspect(0.4/boudary)

plt.savefig(f'qA_energy/Lx{Lx}Ly{Ly}_U{U}_gamma{gamma}_real{realization}.png',
            bbox_inches='tight')
# plt.show()


# # 2. Topological order phase regime.
# t = 1
# U = 1
# gamma = 0.01
# dope = 0

# Lx = 100
# Ly = 8

# import matplotlib.pyplot as plt
# import numpy as np


# fig = plt.figure(dpi=300)
# ax = fig.add_subplot(1,2,1)


# # Disorder realizations
# realization = 50
# list_qa_up = []
# list_qa_dn = []
# list_val_up = []
# list_val_dn = []
# for i in range(realization):
#     val_up, vec_up, val_dn, vec_dn = HF_vec(t, U, gamma, dope, Lx, Ly)
#     list_qa_up.append([compute_qA(state, Lx, Ly) for state in vec_up.transpose()])
#     list_qa_dn.append([compute_qA(state, Lx, Ly) for state in vec_dn.transpose()])
#     list_val_up.append(val_up)
#     list_val_dn.append(val_dn)


# # because Mott-gap = 0.2 when U = 1
# ax.scatter(list_qa_up, np.array(list_val_up)/0.2, s=0.5, fc='none', ec='r')
# ax.scatter(list_qa_dn, np.array(list_val_dn)/0.2, s=0.5, fc='none', ec='b')

# ax.tick_params(axis='both', which='major', labelsize=12)
# ax.set_xlabel(r'$q_A$', fontsize = 20)
# ax.set_ylabel(r'$E/(\Delta/2)$', fontsize = 20)
# ax.set_xlim(0, 1)

# boudary = 0.4
# ax.set_ylim([-boudary, boudary])
# ax.set_aspect(0.4/boudary)

# plt.savefig(f'qA_energy/Lx{Lx}Ly{Ly}_U{U}_gamma{gamma}_real{realization}.png',
#             bbox_inches='tight')
# # plt.show()



# # 3. Crossover phase II phase regime.
# t = 1
# U = 1
# gamma = 5
# dope = 0

# Lx = 100
# Ly = 8

# import matplotlib.pyplot as plt
# import numpy as np


# fig = plt.figure(dpi=300)
# ax = fig.add_subplot(1,2,1)


# # Disorder realizations
# realization = 50
# list_qa_up = []
# list_qa_dn = []
# list_val_up = []
# list_val_dn = []
# for i in range(realization):
#     val_up, vec_up, val_dn, vec_dn = HF_vec(t, U, gamma, dope, Lx, Ly)
#     list_qa_up.append([compute_qA(state, Lx, Ly) for state in vec_up.transpose()])
#     list_qa_dn.append([compute_qA(state, Lx, Ly) for state in vec_dn.transpose()])
#     list_val_up.append(val_up)
#     list_val_dn.append(val_dn)


# # because Mott-gap = 0.2 when U = 1
# ax.scatter(list_qa_up, np.array(list_val_up)/0.2, s=0.5, fc='none', ec='r')
# ax.scatter(list_qa_dn, np.array(list_val_dn)/0.2, s=0.5, fc='none', ec='b')

# ax.tick_params(axis='both', which='major', labelsize=12)
# ax.set_xlabel(r'$q_A$', fontsize = 20)
# ax.set_ylabel(r'$E/(\Delta/2)$', fontsize = 20)
# ax.set_xlim(0, 1)

# boudary = 0.4
# ax.set_ylim([-boudary, boudary])
# ax.set_aspect(0.4/boudary)

# plt.savefig(f'qA_energy/Lx{Lx}Ly{Ly}_U{U}_gamma{gamma}_real{realization}.png',
#             bbox_inches='tight')
# # plt.show()