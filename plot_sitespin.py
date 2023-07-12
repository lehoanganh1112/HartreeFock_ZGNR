import numpy as np
import matplotlib.pyplot as plt
from HF_spinspliting import *

def occupation_spin(state, Lx, Ly):
    '''
    Compute the occupation number of input state. Output is a 1D numpy array.
    '''
    HF_sol = np.power(state[:, 0: (Lx*Ly+dope)//2], 2)
    sz = np.sum(HF_sol, axis=1)
    return sz

# 1. Site spins Sz of symmetry protected phase
t = 1
U = 1
gamma = 0
dope = 0

Lx = 100
Ly = 8


val_up, vec_up, val_dn, vec_dn = HF_vec(t, U, gamma, dope, Lx, Ly)
up = occupation_spin(vec_up, Lx, Ly)
dn = occupation_spin(vec_dn, Lx, Ly)
sz = 1/2 * (up-dn)

fig= plt.figure(dpi=300)
ax = fig.add_subplot(1,1,1, projection="3d")

for i in range(len(sz)):
    coor = coordinate(i, Lx, Ly)
    y = i//Lx
    x = i%Lx
    coor = (x,y)
    ax.plot([coor[0], coor[0]], [coor[1], coor[1]], [0, sz[i]] , 
            color=('r' if sz[i]>=0 else 'b'), linewidth = 0.5, zorder=Lx*Ly-i)

ax.set_xlabel("$k$", fontsize=20)
ax.set_ylabel("$l$", fontsize=20)
ax.set_zlabel("$S_z$", fontsize=20)

ax.set_xlabel("$k$", fontsize=20)
ax.set_ylabel("$l$", fontsize=20)
ax.set_zlabel("$S_z$", fontsize=20)

ax.view_init(elev=20., azim=-15)
ax.set_box_aspect(aspect=(2, 1, 1))

plt.savefig(f'site_spin/sitespin_U{U}_gamma{gamma}_dope{dope}.png', bbox_inches='tight')


# 2. Site spins Sz of TO phase
t = 1
U = 1
gamma = 0.1
dope = 0

Lx = 100
Ly = 8


val_up, vec_up, val_dn, vec_dn = HF_vec(t, U, gamma, dope, Lx, Ly)
up = occupation_spin(vec_up, Lx, Ly)
dn = occupation_spin(vec_dn, Lx, Ly)
sz = 1/2 * (up-dn)

fig= plt.figure(dpi=300)
ax = fig.add_subplot(1,1,1, projection="3d")

for i in range(len(sz)):
    coor = coordinate(i, Lx, Ly)
    y = i//Lx
    x = i%Lx
    coor = (x,y)
    ax.plot([coor[0], coor[0]], [coor[1], coor[1]], [0, sz[i]] , 
            color=('r' if sz[i]>=0 else 'b'), linewidth = 0.5, zorder=Lx*Ly-i)

ax.set_xlabel("$k$", fontsize=20)
ax.set_ylabel("$l$", fontsize=20)
ax.set_zlabel("$S_z$", fontsize=20)

ax.set_xlabel("$k$", fontsize=20)
ax.set_ylabel("$l$", fontsize=20)
ax.set_zlabel("$S_z$", fontsize=20)

ax.view_init(elev=20., azim=-15)
ax.set_box_aspect(aspect=(2, 1, 1))

plt.savefig(f'site_spin/sitespin_U{U}_gamma{gamma}_dope{dope}.png', bbox_inches='tight')


# 3. Charge correlations in TO phase
t = 1
U = 2
gamma = 0.1
dope = 0

Lx = 120
Ly = 8

val_up, vec_up, val_dn, vec_dn = HF_vec(t, U, 0, dope, Lx, Ly)
up_clean = occupation_spin(vec_up, Lx, Ly)
dn_clean = occupation_spin(vec_dn, Lx, Ly)
sz_clean = 1/2 * (up_clean-dn_clean)

val_up, vec_up, val_dn, vec_dn = HF_vec(t, U, gamma, dope, Lx, Ly)
up_disorder = occupation_spin(vec_up, Lx, Ly)
dn_disorder = occupation_spin(vec_dn, Lx, Ly)
sz_disorder = 1/2 * (up_disorder-dn_disorder)

delta_up = up_disorder - up_clean
delta_dn = dn_disorder - dn_clean


fig, axs = plt.subplots(8, sharex=True, dpi=300)

index_pair = [0, 7, 1, 6, 2, 5, 3, 4]

for id, ax in enumerate(axs):
    ax.plot(delta_up[index_pair[id]*Lx:(index_pair[id]+1)*Lx], color='k')
    ax.set_ylabel(f'({index_pair[id]+1})', fontsize=20)
    ax.set_ylim([-0.5, 0.5])
    ax.axhline(0, linestyle='--', color='r')
    ax.set_aspect(40)

fig.subplots_adjust(hspace=0)
fig.suptitle(r'$\delta n_{i \uparrow}$', fontsize=20)
plt.savefig(f'site_spin/induceoccupation_U{U}_gamma{gamma}_dope{dope}.png', 
            bbox_inches='tight')


# 4. Charge correlations in COI phase
t = 1
U = 3
gamma = 0.5
dope = 0

Lx = 120
Ly = 8

val_up, vec_up, val_dn, vec_dn = HF_vec(t, U, 0, dope, Lx, Ly)
up_clean = occupation_spin(vec_up, Lx, Ly)
dn_clean = occupation_spin(vec_dn, Lx, Ly)
sz_clean = 1/2 * (up_clean-dn_clean)

val_up, vec_up, val_dn, vec_dn = HF_vec(t, U, gamma, dope, Lx, Ly)
up_disorder = occupation_spin(vec_up, Lx, Ly)
dn_disorder = occupation_spin(vec_dn, Lx, Ly)
sz_disorder = 1/2 * (up_disorder-dn_disorder)

delta_up = up_disorder - up_clean
delta_dn = dn_disorder - dn_clean


fig, axs = plt.subplots(8, sharex=True, dpi=300)

index_pair = [0, 7, 1, 6, 2, 5, 3, 4]

for id, ax in enumerate(axs):
    ax.plot(delta_up[index_pair[id]*Lx:(index_pair[id]+1)*Lx], color='k')
    ax.set_ylabel(f'({index_pair[id]+1})', fontsize=20)
    ax.set_ylim([-0.8, 0.8])
    ax.axhline(0, linestyle='--', color='r')
    ax.set_aspect(40)

fig.subplots_adjust(hspace=0)
fig.suptitle(r'$\delta n_{i \uparrow}$', fontsize=20)
plt.savefig(f'site_spin/induceoccupation_U{U}_gamma{gamma}_dope{dope}.png', 
            bbox_inches='tight')


t = 1
U = 3
gamma = 4
dope = 0

Lx = 120
Ly = 8

val_up, vec_up, val_dn, vec_dn = HF_vec(t, U, 0, dope, Lx, Ly)
up_clean = occupation_spin(vec_up, Lx, Ly)
dn_clean = occupation_spin(vec_dn, Lx, Ly)
sz_clean = 1/2 * (up_clean-dn_clean)

val_up, vec_up, val_dn, vec_dn = HF_vec(t, U, gamma, dope, Lx, Ly)
up_disorder = occupation_spin(vec_up, Lx, Ly)
dn_disorder = occupation_spin(vec_dn, Lx, Ly)
sz_disorder = 1/2 * (up_disorder-dn_disorder)

delta_up = up_disorder - up_clean
delta_dn = dn_disorder - dn_clean


fig, axs = plt.subplots(8, sharex=True, dpi=300)

index_pair = [0, 7, 1, 6, 2, 5, 3, 4]

for id, ax in enumerate(axs):
    ax.plot(delta_up[index_pair[id]*Lx:(index_pair[id]+1)*Lx], color='k')
    ax.set_ylabel(f'({index_pair[id]+1})', fontsize=20)
    ax.set_ylim([-0.8, 0.8])
    ax.axhline(0, linestyle='--', color='r')
    ax.set_aspect(40)

fig.subplots_adjust(hspace=0)
fig.suptitle(r'$\delta n_{i \uparrow}$', fontsize=20)
plt.savefig(f'site_spin/induceoccupation_U{U}_gamma{gamma}_dope{dope}.png', 
            bbox_inches='tight')