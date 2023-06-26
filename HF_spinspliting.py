# <---------- Import necessary library ---------->
# import cupy as cp
import numpy as cp
import numpy as np
from time import time
import random
from tqdm import tqdm


# These code lines prevent error of Allocated Memory of Cuda/11.3
pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
cp.cuda.set_allocator(pool.malloc)



# <---------- 1. Module to solve Hartree-Fock eigensystem ---------->
def hopping_list(Lx, Ly):
    '''
    Generate the list of nearest-neighbor hopping of an (Lx, Ly) ZGNR with 
    periodic boundary condition.
    < Input >
    Lx: (int) length of ribbon.
    Ly: (int) width of ribbon.
    < Output >
    the_hopping_list: (list) list of n.n. hoppings on ZGNR.
    '''
    the_hopping_list = []        
    for i in range(Lx*Ly):
        if cp.floor_divide(i, Lx) % 4 == 0:
            if cp.remainder(i, Lx) == 0:
                the_hopping_list.append((i,i + Lx))
                the_hopping_list.append((i,i + 2*Lx-1))
                if i-Lx>0:
                    the_hopping_list.append((i,i-Lx))        
            else:
                the_hopping_list.append((i,i + Lx-1))
                the_hopping_list.append((i,i + Lx))
                if i-Lx>0:
                    the_hopping_list.append((i,i-Lx))
        if (cp.floor_divide(i, Lx) + 2) % 4 == 0:
            if cp.remainder(i, Lx) == Lx-1:
                the_hopping_list.append((i,i+Lx))
                the_hopping_list.append((i,i+1))
                the_hopping_list.append((i,i-Lx))
            else:
                the_hopping_list.append((i,i+Lx+1))
                the_hopping_list.append((i,i+Lx))
                the_hopping_list.append((i,i-Lx))
    return the_hopping_list

def coordinate(j, Lx, Ly):
    '''
    Return Cartesian coordinates of site j in an (Lx, Ly) ribbon. This function
    helps to check whether hopping terms are correctly generated.
    < Input > 
    j: (int) (0 <= j < Lx*Ly) Order of site j
    Lx: (int) length of ribbon.
    Ly: (int) width of ribbon.
    < Output >
    (x, y) : (tuples) coordinates of site j
    < Notice >
    With this notation the most upper-left site is labeled 0
    '''
    x = j % Lx + 1
    y = Ly - (j // Lx + 1) + 1
    if y % 2 == 1 and ((y + 1)/2) % 2 == 1:
        return (x, ((y + 1)/2) * cp.sqrt(3)/2)
    if y % 2 == 1 and ((y + 1)/2) % 2 == 0:
        return (x + 1/2, ((y + 1)/2) * cp.sqrt(3)/2)
    if y % 2 == 0 and (y/2) % 2 == 0:
        return (x, (y/2) * cp.sqrt(3)/2 + 1/(2*cp.sqrt(3)))
    if y % 2 == 0 and (y/2) % 2 == 1:
        return (x + 1/2, (y/2) * cp.sqrt(3)/2 + 1/(2*cp.sqrt(3)))
    
def hopping_matrix(t, Lx, Ly):
    '''
    Return hopping matrix for mean-field Hamiltonian of ZGNR.
    < Input >
    Lx: (int) length of ribbon.
    Ly: (int) width of ribbon.
    < Output >
    (ndarrays) (Lx*Ly) by (Lx*Ly) hopping matrix.
    '''
    draft_hopping = cp.zeros((Lx*Ly, Lx*Ly))
    for element in hopping_list(Lx, Ly):
        draft_hopping[element] = -t
    return draft_hopping + cp.transpose(draft_hopping)

def initial_condition(gamma, U, dope, Lx, Ly, ferro = False):
    '''
    Return initial condition for occupation number. There are 3 types of 
    initial conditions considered here: ferromagnetic, anti-ferromagnetic 
    and paramagnetic.

    If gamma is not zero, PM initial condition is used: it generates initial
    condition for spin-up matrix, and matrix of spin-down one will be 
    multiplied by -1 in creating Hamiltonian.

    NOTICE: interaction strength U is absorbed into the initial condition.
    < Input >
    Lx: (int) length of ribbon.
    Ly: (int) width of ribbon.
    < Output >
    (ndarrays) (Lx*Ly) by (Lx*Ly) initial condition matrix.
    '''
    nf = 1 + dope/(Lx * Ly)
    if ferro == True:
        print("Using F initial condition")
        lst = []
        for i in range(Lx * Ly):
            lst.append(U * (nf/2))
        return cp.diag(lst)
    else:
        if gamma < 1e-3 and dope == 0:
            print("Using AF initial condition")
            lst = []
            for i in range(Lx * Ly):
                yi = Ly - (i // Lx + 1) + 1
                if yi % 2 == 0:
                    lst.append(U * (nf/2))
                else:
                    lst.append(U * (-nf/2))
            return cp.diag(lst)
        else:
            print("Using PM initial condition")
            pm_matrix = cp.random.choice((U * 0.001, -U * 0.001), Lx * Ly)
            return cp.diag(pm_matrix)
        
def impurity_matrix(gamma, Lx, Ly):
    '''
    Return impurity matrix. The percentage of sites with impurity is 10%.
    < Input >
    Lx: (int) length of ribbon.
    Ly: (int) width of ribbon.
    < Output >
    (ndarrays) (Lx*Ly) by (Lx*Ly) impurity matrix.
    '''
    imp = 0.1
    num_imp = Lx*Ly*imp
    if num_imp - int(num_imp) >= 0.5:
        num_imp_site = int(cp.ceil(num_imp))
    else:
        num_imp_site = int(num_imp)
    print("Number of impurity sites: " + str(num_imp_site))
    draft_impurity = cp.zeros(Lx*Ly)
    for impurity_site in random.sample(range(Lx*Ly), num_imp_site):
        draft_impurity[impurity_site] = cp.random.uniform(-gamma, gamma)
    return cp.diag(draft_impurity)
        
def self_consistent_solution(t, U, gamma, dope, Lx, Ly, number_iteration=14):
    nf = 1 + dope/(Lx * Ly)
    ini_hamil = hopping_matrix(Lx, Ly, t) + impurity_matrix(Lx, Ly, gamma)
    ini_cond = initial_condition(Lx, Ly, U, gamma)
    val_up, vec_up = cp.linalg.eigh(ini_hamil + ini_cond)
    val_dn, vec_dn = cp.linalg.eigh(ini_hamil - ini_cond)    
    
    def average_n_matrix(vec):
        average_n = 0
        for i in range(int((Lx*Ly+dope)/2)):
            average_n = average_n + abs(vec[:, i])**2
        average_n = U * (average_n - nf/2)
        return cp.diag(average_n)
    
    # The HF loop
    hamil_up = ini_hamil + average_n_matrix(vec_dn)
    hamil_dn = ini_hamil + average_n_matrix(vec_up)    
    for i in range(number_iteration):
        print("At iteration: " + str(i))
        val_up, vec_up = cp.linalg.eigh(hamil_up)
        val_dn, vec_dn = cp.linalg.eigh(hamil_dn)
        hamil_up = ini_hamil + average_n_matrix(vec_dn)
        hamil_dn = ini_hamil + average_n_matrix(vec_up)
    return hamil_up, hamil_dn

def HF_vec(t, U, gamma, dope, Lx, Ly):
    '''
    Return Hartree-Fock eigensystem.
    < Input >
    Lx: (int) length of ribbon.
    Ly: (int) width of ribbon.
    < Output >
    val, vec
    val: (2*Lx*Ly) column vector of eigenvalues.
    vec[:, i]: (2*Lx*Ly) column-eigenvector corresponding to val[i].
    '''
    the_system = self_consistent_solution(t, U, gamma, dope, Lx, Ly)
    val_up, vec_up = cp.linalg.eigh(the_system[0])
    val_dn, vec_dn = cp.linalg.eigh(the_system[1])
    return val_up, vec_up, val_dn, vec_dn

# <---------- 2. Module to compute TEE ---------->
def partition_geometry(w, l_zig, l_arm, Lx, Ly):
    ## l_zig: number of zigzag sites in partitioned region (f.e.: l_zig = 161 L' = 200)
    ## must be an odd number!
    ## l_arm + 2*w: number of armchair sites in partitioned *ucpacking* region
    ## must be an even number
    '''
    Determine the coordinates of ....
    *******************************************************
    *********|---------------------------(xmax, ymax)******
    *********|*********************************|***********
    *********|**********|------(xmax1, ymmax1)*|***********
    *********|**********|*************|********|***********
    *********|****(xmin1, ymin1)------|********|***********
    *********|*********************************|***********
    ****(xmin, ymin)---------------------------|***********
    *******************************************************
    '''
    
    zig_cutoff = int((Lx - (l_zig+1)/2)/2)
    arm_cutoff = int((Ly - (l_arm + 2*w))/2)
    lt = arm_cutoff * Lx + zig_cutoff
    rt = arm_cutoff * Lx + zig_cutoff + (l_zig + 1)/2 - 1
    lb = (arm_cutoff + (l_arm + 2*w)-1) * Lx + zig_cutoff
    delta = 0.05
    xmin, ymin = coordinate(lb, Lx, Ly)
    xmax, ymax = coordinate(rt, Lx, Ly)
    xmin1 = xmin + (w-1)*1/2 + delta
    xmax1 = xmax - (w-1)*1/2 - delta
    if cp.mod(w, 2) == 1:
        ymin1 = ymin + (w-1)/2*cp.sqrt(3)/6 + (w-1)/2*cp.sqrt(3)/3 + delta
        ymax1 = ymax - (w-1)/2*cp.sqrt(3)/6 - (w-1)/2*cp.sqrt(3)/3 - delta
    elif cp.mod(w-1, 2) == 1 and cp.mod(lb // Lx + 1, 2) == 1:
        ymin1 = ymin + w/2*cp.sqrt(3)/6 + (w/2-1)*cp.sqrt(3)/3 + delta
        ymax1 = ymax - w/2*cp.sqrt(3)/6 - (w/2-1)*cp.sqrt(3)/3 - delta
    elif cp.mod(w-1, 2) == 1 and cp.mod(lb // Lx + 1, 2) == 0:
        ymin1 = ymin + w/2*cp.sqrt(3)/3 + (w/2-1)*cp.sqrt(3)/6 + delta
        ymax1 = ymax - w/2*cp.sqrt(3)/3 - (w/2-1)*cp.sqrt(3)/6 - delta
    return xmin, xmax, ymin, ymax, xmin1, xmax1, ymin1, ymax1

def listA(w, l_zig, l_arm, Lx, Ly):
    xmin, xmax, ymin, ymax, xmin1, xmax1, ymin1, ymax1 = partition_geometry(w, l_zig, l_arm, Lx, Ly)
    lst = []
    for j in range(Lx*Ly):
        x = coordinate(j, Lx, Ly)[0]
        y = coordinate(j, Lx, Ly)[1]
        condition_out = (xmin <= x <= xmax and ymin <= y <= ymax)
        condition_in = (xmin1 < x < xmax1 and ymin1 < y < ymax1)
        if condition_out and not condition_in:
            lst.append(j)
    return lst

def listB(w, l_zig, l_arm, Lx, Ly):
    xmin, xmax, ymin, ymax, xmin1, xmax1, ymin1, ymax1 = partition_geometry(w, l_zig, l_arm, Lx, Ly)
    lst = []
    for j in range(Lx*Ly):
        x = coordinate(j, Lx, Ly)[0]
        y = coordinate(j, Lx, Ly)[1]
        condition_out = (xmin <= x <= xmax and ymin <= y <= ymax)
        condition_in = (xmin1 < x < xmax1 and ymin1 < y <= ymax)
        if condition_out and not condition_in:
            lst.append(j)
    return lst

def listC(w, l_zig, l_arm, Lx, Ly):
    xmin, xmax, ymin, ymax, xmin1, xmax1, ymin1, ymax1 = partition_geometry(w, l_zig, l_arm, Lx, Ly)
    lst = []
    for j in range(Lx*Ly):
        x = coordinate(j, Lx, Ly)[0]
        y = coordinate(j, Lx, Ly)[1]
        condition_out = (xmin <= x <= xmax and ymin <= y <= ymax)
        condition_in = (xmin1 < x < xmax1 and ymin <= y < ymax1)
        if condition_out and not condition_in:
            lst.append(j)
    return lst

def listD(w, l_zig, l_arm, Lx, Ly):
    xmin, xmax, ymin, ymax, xmin1, xmax1, ymin1, ymax1 = partition_geometry(w, l_zig, l_arm, Lx, Ly)
    lst = []
    for j in range(Lx*Ly):
        x = coordinate(j, Lx, Ly)[0]
        y = coordinate(j, Lx, Ly)[1]
        condition_out = (xmin <= x <= xmax and ymin <= y <= ymax)
        condition_in = (xmin1 < x < xmax1 and ymin <= y <= ymax)
        if condition_out and not condition_in:
            lst.append(j)
    return lst

def test_lt(w, l_zig, l_arm, Lx, Ly):
    zig_cutoff = int((Lx - (l_zig+1)/2)/2)
    arm_cutoff = int((Ly - (l_arm + 2*w))/2)
    lt = arm_cutoff * Lx + zig_cutoff ## must fall on odd row 
    if cp.mod(lt // Lx + 1, 2) == 1:
        return print("Top-left site test: PASS")
    else:
        return print("Top-left site test: FAIL")
    
def test_difference(w, l_zig, l_arm, Lx, Ly):
    a = set(listA(w, l_zig, l_arm, Lx, Ly))
    b = set(listB(w, l_zig, l_arm, Lx, Ly))
    c = set(listC(w, l_zig, l_arm, Lx, Ly))
    d = set(listD(w, l_zig, l_arm, Lx, Ly))
    if a.difference(b) == c.difference(d):
        print("Levin-Wen test: PASS")
    else:
        print("Levin-Wen test: FAIL")