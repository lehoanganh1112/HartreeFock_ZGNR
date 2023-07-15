import numpy as np
import matplotlib.pyplot as plt
import csv
from HF_spinspliting import *
import pandas as pd

##### WARNING: without CUDA, this file would run for a long time! #####

# t = 1
# dope = 0

# Lx = 100
# Ly = 64
# w = 4
# l_zig = 161
# l_arm = 28


# list_of_gamma = [0.1, 0.5, 1, 2, 3, 4, 5]
# list_of_U = [1]

# data = []
# for U in tqdm(list_of_U, desc='U',  leave=True):
#     for gamma in tqdm(list_of_gamma, desc='Gamma',  leave=True):
#         t0 = time()
#         realization = 5
#         output = np.zeros(realization)
#         for i in range(realization):
#             val_up, vec_up, val_dn, vec_dn = HF_vec(t, U, gamma, dope, Lx, Ly)
#             re = entropy_TEE(vec_up, dope, w, l_zig, l_arm, Lx, Ly)
#             print(re) 
#             output[i] = re
#         data.append([U, gamma] + list(output))
#         print('Time Elapsed:\t{:.4f}'.format(time() - t0))


# fields = ['U', 'Gamma', 'real1', 'real2', 'real3', 'real4', 'real5']
# with open('TEE.csv', 'w') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     csvwriter.writerow(fields)
#     csvwriter.writerows(data)


data = pd.read_csv('./TEE.csv')
data = data.append({'U': 1, 'Gamma':0, 'real1':0, 'real2':0, 'real3':0, 'real4':0, 'real5':0}, 
            ignore_index=True)
data1 = data.sort_values(by=data.columns[1])
print(data1)

fig = plt.figure(dpi=300)
ax = fig.gca()


ax.errorbar(data1['Gamma']/data1['U'], data1.iloc[:, 2::].mean(axis=1), 
            yerr=[0, 0.003, 0.003, 0.003, 0.003,0.003, 0.003, 0.003],
        color='b', marker='^', linewidth=0.5, capsize=5)
ax.set_ylim([0, 0.025])

plt.savefig('./tee/tee_result.png', bbox_inches='tight')