import cupy as cp
# import numpy as cp
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
    the_hopping_list: (list) list of n.n. hopping.
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
    Give Cartesian coordinates of site j in an (Lx, Ly) ribbon. This function
    helps to check whether hopping terms are correctly generated.
    < Input > 
    j: (int) (0 <= j < Lx*Ly) Order of site j
    Lx: (int) length of ribbon.
    Ly: (int) width of ribbon.
    < Output >
    (x, y) : (tuples) coordinates of site j
    '''
    x = j % Lx + 1
    ## With this notation the top left site is 0
    y = Ly - (j // Lx + 1) + 1
    if y % 2 == 1 and ((y + 1)/2) % 2 == 1:
        return (x, ((y + 1)/2) * cp.sqrt(3)/2)
    if y % 2 == 1 and ((y + 1)/2) % 2 == 0:
        return (x + 1/2, ((y + 1)/2) * cp.sqrt(3)/2)
    if y % 2 == 0 and (y/2) % 2 == 0:
        return (x, (y/2) * cp.sqrt(3)/2 + 1/(2*cp.sqrt(3)))
    if y % 2 == 0 and (y/2) % 2 == 1:
        return (x + 1/2, (y/2) * cp.sqrt(3)/2 + 1/(2*cp.sqrt(3)))
    

def hopping_matrix(Lx, Ly):
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


def initial_condition(Lx, Ly, ferro = False):
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
    if ferro == True:
        print("Using F initial condition")
        nf = 1 + dope/(Lx * Ly)
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


def initial_condition_sxy(Lx, Ly):
    '''
    Return initial condtion for spin-x and spin-y operators.

    NOTICE: interaction strength U is absorbed into the initial condition.
    This term is equivalent to -2U <s_{i,x}> or -2U <s_{i,y}> 
    < Input >
    Lx: (int) length of ribbon.
    Ly: (int) width of ribbon.
    < Output >
    (ndarrays) (Lx*Ly) by (Lx*Ly) initial condition matrix.
    Notice:
    March 3rd: change 'amplitude' as a variable
    (amplitude = 0.00002)
    '''
    draft_initial = cp.zeros(Lx*Ly)
    amplitude = 0.00002
    for i in range(Lx*Ly):
        draft_initial[i] = -2 * U * cp.random.uniform(-amplitude, amplitude)
    return np.diag(draft_initial)


def impurity_matrix(Lx, Ly):
    '''
    Return impurity matrix.
    < Input >
    Lx: (int) length of ribbon.
    Ly: (int) width of ribbon.
    < Output >
    (ndarrays) (Lx*Ly) by (Lx*Ly) impurity matrix.
    '''
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
    
def self_consistent_solution(Lx, Ly, number_iteration=14):
    '''
    Solution for the Hartree-Fock algorithm
    
    Revision: 
    '''
    # 1. Construct initial Hamiltonian:
    # Generate hopping, impurity and initial matrices
    ini_hamil = hopping_matrix(Lx, Ly) + impurity_matrix(Lx, Ly)
    ini_cond = initial_condition(Lx, Ly)
    if dope == 0:
        print('Off-diagonal of Hamiltonian is ZERO')
        ini_hx = cp.zeros((Lx*Ly, Lx*Ly))
        ini_hy = cp.zeros((Lx*Ly, Lx*Ly))
    else:
        ini_hx = initial_condition_sxy(Lx, Ly)
        ini_hy = initial_condition_sxy(Lx, Ly)

    # Construct Hamiltonian's blocks
    h11 = ini_hamil + ini_cond
    h22 = ini_hamil - ini_cond
    h12 = 0.5 * (ini_hx - 1j * ini_hy)
    h21 = 0.5 * (ini_hx + 1j * ini_hy)

    h_upper = cp.hstack((h11, h12))
    h_lower = cp.hstack((h21, h22))
    hamiltonian = cp.vstack((h_upper, h_lower))
    hamiltonian = 0.5 * (hamiltonian + cp.conj(cp.transpose(hamiltonian)))

    val, vec = cp.linalg.eigh(hamiltonian)

    # 2. Interation process
    def average_n_spin_up(vec):
        '''
        < Input >
        vec: eigenvectors system of length 2*Lx*Ly
        < Output >
        n_spin_up: Lx*Ly matrix of expectation value of spin-up
        ''' 
        average_n_up = 0
        # alpha stands for occupied index
        for alpha in range(int(Lx*Ly+dope)):
            average_n_up += abs(vec[0:int(Lx*Ly), alpha])**2 # take the upper part
        average_n_up = U * (average_n_up - nf/2) # a column matrix
        return cp.diag(average_n_up) #return 2D diagonal matrix

    def average_n_spin_dn(vec):
        '''
        < Input >
        vec: eigenvectors system of length 2*Lx*Ly
        < Output >
        n_spin_dn: Lx*Ly matrix of expectation value of spin-down
        ''' 
        average_n_dn = 0
        # alpha stands for occupied index
        for alpha in range(int(Lx*Ly+dope)):
            average_n_dn += abs(vec[int(Lx*Ly):int(2*Lx*Ly), alpha])**2 # take the lower part
        average_n_dn = U * (average_n_dn - nf/2) # a column matrix
        return cp.diag(average_n_dn) #return 2D diagonal matrix

    def average_sx_matrix(vec):
        '''
        < Input >
        vec: eigenvectors system of length 2*Lx*Ly
        < Output >
        sx: Lx*Ly matrix of expectation value of sx operator.

        NOTICE: interaction strength U is absorbed. This term is equivalent to
        -2U <s_{i,x}> 
        = -U \sum_{n_k} (A^{k*}_{i,up}A^{k}_{i,dn} + A^{k*}_{i,dn}A^k_{i,up})
        '''
        average_sx = 0
        # alpha stands for occupied index
        for alpha in range(int(Lx*Ly+dope)):
            upper = vec[0:int(Lx*Ly), alpha]
            lower = vec[int(Lx*Ly):int(2*Lx*Ly), alpha]
            # Multiply element-wise
            average_sx += cp.multiply(cp.conj(upper), lower) + cp.multiply(upper, cp.conj(lower))
        average_sx = -U * average_sx
        return cp.diag(average_sx)

    
    def average_sy_matrix(vec):
        '''
        < Input >
        vec: eigenvectors system of length 2*Lx*Ly
        < Output >
        sy: Lx*Ly matrix of expectation value of sy operator.

        NOTICE: interaction strength U is absorbed. This term is equivalent to
        -2U <s_{i,y}> 
        = -iU \sum_{n_k} (A^{k*}_{i,up}A^{k}_{i,dn} - A^{k*}_{i,dn}A^k_{i,up})
        '''
        average_sy = 0
        # alpha stands for occupied index
        for alpha in range(int(Lx*Ly+dope)):
            upper = vec[0:int(Lx*Ly), alpha]
            lower = vec[int(Lx*Ly):int(2*Lx*Ly), alpha]
            # Multiply element-wise
            average_sy += cp.multiply(cp.conj(upper), lower) - cp.multiply(upper, cp.conj(lower))
        average_sy = -U * 1j * average_sy
        return cp.diag(average_sy)


    h11 = ini_hamil + average_n_spin_dn(vec)
    h22 = ini_hamil + average_n_spin_up(vec)
    h12 = 0.5 * (average_sx_matrix(vec) - 1j * average_sy_matrix(vec))
    h21 = 0.5 * (average_sx_matrix(vec) + 1j * average_sy_matrix(vec))
    hamiltonian = cp.vstack((cp.hstack((h11, h12)), cp.hstack((h21, h22))))
    hamiltonian = 0.5 * (hamiltonian + cp.conj(cp.transpose(hamiltonian)))

    for i in range(number_iteration):
        print("At iteration: " + str(i+1))
        val, vec = cp.linalg.eigh(hamiltonian)
        h11 = ini_hamil + average_n_spin_dn(vec)
        h12 = 0.5 * (average_sx_matrix(vec) - 1j * average_sy_matrix(vec))
        h21 = 0.5 * (average_sx_matrix(vec) + 1j * average_sy_matrix(vec))
        print(cp.linalg.norm(h12))
        h22 = ini_hamil + average_n_spin_up(vec)
        hamiltonian = cp.vstack((cp.hstack((h11, h12)), cp.hstack((h21, h22))))
        hamiltonian = 0.5 * (hamiltonian + cp.conj(cp.transpose(hamiltonian)))
    return hamiltonian


def HF_vec(Lx, Ly):
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
    the_system = self_consistent_solution(Lx, Ly)
    val, vec = cp.linalg.eigh(the_system)
    return val, vec


# <---------- 2. Module to compute TEE ---------->
def matrixC(Lx, Ly, spin_up = True):
    thevec = cp.linalg.inv(cp.transpose(eigen_vec))
    # Occupied index k runs from 0 to Lx*Ly+dope
    # Spin-up correlation
    if spin_up == True:
        print('Spin-up correlation is used')
        Bjk = thevec[0:int(Lx*Ly), 0: int(Lx*Ly+dope)]
        Bki_star = cp.conjugate(cp.transpose(thevec))[0: int(Lx*Ly + dope), 0:int(Lx*Ly)]
    # Spin-down correlation
    else:
        print('Spin-down correlation is used')
        Bjk = thevec[int(Lx*Ly):int(2*Lx*Ly), 0: int(Lx*Ly+dope)]
        Bki_star = cp.conjugate(cp.transpose(thevec))[0: int(Lx*Ly + dope), int(Lx*Ly):int(2*Lx*Ly)]
    print(Bjk.shape)
    return cp.einsum('ki, jk -> ij', Bki_star, Bjk)


def entropy(C, partition_list):
    C = cp.take(C, partition_list, axis = 1)
    C = cp.take(C, partition_list, axis = 0)
    lamda = cp.linalg.eigvalsh(C)
    tol = 1e-100
    lamda.real[abs(lamda.real) < tol] = 0.0
    entropy = 0
    for i in lamda.real:
        if i>=1 or i<=0:
            entropy = entropy
        else:
            entropy = entropy - (i * cp.log(i) + (1-i) * cp.log(1-i))
    return entropy


def test_lt(w, l_zig, l_arm, Lx, Ly):
    zig_cutoff = int((Lx - (l_zig+1)/2)/2)
    arm_cutoff = int((Ly - (l_arm + 2*w))/2)
    lt = arm_cutoff * Lx + zig_cutoff ## must fall on odd row 
    if cp.mod(lt // Lx + 1, 2) == 1:
        return print("The condition of lt is satisfied")
    else:
        return print("The condition of lt is **NOT** satisfied")
        
def partition_geometry(w, l_zig, l_arm, Lx, Ly):
    ## l_zig: number of zigzag sites in partitioned region (f.e.: l_zig = 161 L' = 200)
    ## must be an odd number!
    ## l_arm + 2*w: number of armchair sites in partitioned *ucpacking* region
    ## must be an even number
    
    zig_cutoff = int((Lx - (l_zig+1)/2)/2)
    arm_cutoff = int((Ly - (l_arm + 2*w))/2)
    
    lt = arm_cutoff * Lx + zig_cutoff ## must fall on odd row 
    rt = arm_cutoff * Lx + zig_cutoff + (l_zig + 1)/2 - 1
    lb = (arm_cutoff + (l_arm + 2*w)-1) * Lx + zig_cutoff
    delta = 0.05
    xmin = coordinate(lb, Lx, Ly)[0]
    ymin = coordinate(lb, Lx, Ly)[1]
    xmax = coordinate(rt, Lx, Ly)[0]
    ymax = coordinate(lt, Lx, Ly)[1]
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
    return ((xmin, xmax, ymin, ymax), (xmin1, xmax1, ymin1, ymax1))


def listA(w, l_zig, l_arm, Lx, Ly):
    ((xmin, xmax, ymin, ymax), (xmin1, xmax1, ymin1, ymax1)) = partition_geometry(w, l_zig, l_arm, Lx, Ly)
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
    ((xmin, xmax, ymin, ymax), (xmin1, xmax1, ymin1, ymax1)) = partition_geometry(w, l_zig, l_arm, Lx, Ly)
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
    ((xmin, xmax, ymin, ymax), (xmin1, xmax1, ymin1, ymax1)) = partition_geometry(w, l_zig, l_arm, Lx, Ly)
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
    ((xmin, xmax, ymin, ymax), (xmin1, xmax1, ymin1, ymax1)) = partition_geometry(w, l_zig, l_arm, Lx, Ly)
    lst = []
    for j in range(Lx*Ly):
        x = coordinate(j, Lx, Ly)[0]
        y = coordinate(j, Lx, Ly)[1]
        condition_out = (xmin <= x <= xmax and ymin <= y <= ymax)
        condition_in = (xmin1 < x < xmax1 and ymin <= y <= ymax)
        if condition_out and not condition_in:
            lst.append(j)
    return lst


###### Test from Levin-Wen's paper
def test_difference(w, l_zig, l_arm, Lx, Ly):
    a = set(listA(w, l_zig, l_arm, Lx, Ly))
    b = set(listB(w, l_zig, l_arm, Lx, Ly))
    c = set(listC(w, l_zig, l_arm, Lx, Ly))
    d = set(listD(w, l_zig, l_arm, Lx, Ly))
    # print(a.difference(b))
    # print(c.difference(d))
    if a.difference(b) == c.difference(d):
        print("Levin-Wen test is passed")
    else:
        print("Levin-Wen test is NOT passed")

def entropy_TEE(w, l_zig, l_arm, Lx, Ly, spin_up=True):
    matC = matrixC(Lx, Ly, spin_up)
    sa = entropy(matC, listA(w, l_zig, l_arm, Lx, Ly))
    sb = entropy(matC, listB(w, l_zig, l_arm, Lx, Ly))
    sc = entropy(matC, listC(w, l_zig, l_arm, Lx, Ly))
    sd = entropy(matC, listD(w, l_zig, l_arm, Lx, Ly))  
    return 0.5 * ((sc - sd) - (sa - sb))

# <---------- 3. Module to compute site spins, occupation numbers ---------->
def hf_sx(Lx, Ly):
    '''
    < Input >
    vec: eigenvectors system of length 2*Lx*Ly
    < Output >
    
    '''
    average_sx = 0
    # alpha stands for occupied index
    for alpha in range(int(Lx*Ly+dope)):
        upper = eigen_vec[0:int(Lx*Ly), alpha]
        lower = eigen_vec[int(Lx*Ly):int(2*Lx*Ly), alpha]
        # Multiply element-wise
        average_sx += cp.multiply(cp.conj(upper), lower) + cp.multiply(upper, cp.conj(lower))
    return average_sx.real


def hf_sz(Lx, Ly):
    '''
    < Input >
    vec: eigenvectors system of length 2*Lx*Ly
    < Output >
    
    '''
    average_sz = 0
    # alpha stands for occupied index
    for alpha in range(int(Lx*Ly+dope)):
        upper = eigen_vec[0:int(Lx*Ly), alpha]
        lower = eigen_vec[int(Lx*Ly):int(2*Lx*Ly), alpha]
        # Multiply element-wise
        average_sz += cp.multiply(cp.conj(upper), upper) - cp.multiply(lower, cp.conj(lower))
    return average_sz.real


# EXAMPLE 1: Compute qA-energy diagram

# Set up parameters
t = 1
imp = 0.1

Lx = 100
Ly = 8
gamma = 0.01
dope = 3
U = 1
nf = 1 + dope/(Lx * Ly)

def prob_density(vector_solution, order, Lx, Ly):
    density = np.power(vector_solution[:, order], 2)
    density = density/np.sum(density)
    return np.ndarray.flatten(density)


listA = []
listB = []
list_overlap = []
dope = 0

for i in range(Lx * Ly):
    yi = Ly - (i // Lx + 1) + 1
    if yi % 2 == 0:
        listA.append(i)
    else:
        listB.append(i)

for i in range(Lx * Ly):
    yi = Ly - (i // Lx + 1) + 1
    if Ly/2 -1 <= yi <= Ly/2:
        list_overlap.append(i)


list_qa_up = []
list_qa_dn = []
list_energy_up = []
list_energy_dn = []

realization = 300


for real in range(realization):
    print('In realization: ' + str(real+1))
    # Call the vectors and normalize them
    val, vec = HF_vec(Lx, Ly)
    vec_norm = np.zeros((Lx*Ly*2, Lx*Ly*2))
    for i in range(Lx*Ly*2):
        vec_ele = vec[:, i]
        vec_ele = vec_ele/np.linalg.norm(vec_ele)
        vec_norm[:, i] = vec_ele

    vec_up = vec_norm[0:(Lx*Ly), :]
    vec_dn = vec_norm[(Lx*Ly):(2*Lx*Ly), :]

    for i in range(Lx*Ly*2):
        ORDER_up = i
        ORDER_dn = i

        density_up = prob_density(vec_up, ORDER_up, Lx, Ly)
        qa_up = np.sum(np.take(prob_density(vec_up, ORDER_up, Lx, Ly), listA))
        energy_up = val[ORDER_up]

        if qa_up > 1e-4:
            list_qa_up.append(qa_up)
            list_energy_up.append(energy_up)


        density_dn = prob_density(vec_dn, ORDER_dn, Lx, Ly)
        qa_dn = np.sum(np.take(prob_density(vec_dn, ORDER_dn, Lx, Ly), listA))
        energy_dn = val[ORDER_dn]


        if qa_dn > 1e-4:
            list_qa_dn.append(qa_dn)
            list_energy_dn.append(energy_dn)

list_qa_up = np.array(list_qa_up)
list_qa_dn = np.array(list_qa_dn)
list_energy_up = np.array(list_energy_up)
list_energy_dn = np.array(list_energy_dn)

np.savetxt('list_qa_up_L100W8_U1G001_dope3.txt', list_qa_up)
np.savetxt('list_qa_dn_L100W8_U1G001_dope3.txt', list_qa_dn)
np.savetxt('list_energy_up_L100W8_U1G001_dope3.txt', list_energy_up)
np.savetxt('list_energy_dn_L100W8_U1G001_dope3.txt', list_energy_dn)


# # Declare parameters
# Lx = 100
# Ly = 64
# w = 4
# l_zig = 161
# l_arm = 28

# t = 1
# imp = 0.1
# dope = 0
# nf = 1 + dope/(Lx * Ly)

# eigen_val, eigen_vec = HF_vec(Lx, Ly)

# #### Large parameters
# Lx = 200
# Ly = 172
# w = 12
# l_zig = 341
# l_arm = 68



# #### Large parameters
# Lx = 170
# Ly = 144
# w = 10
# l_zig = 289
# l_arm = 60

# # #### Large parameters
# # # Lx = 150
# Lx = 130
# Ly = 112
# w = 7
# l_zig = 221
# l_arm = 46



# # ##### Real parameters
# Lx = 100
# Ly = 64
# w = 4
# l_zig = 161
# l_arm = 28



# ##### Real parameters
# Lx = 80
# Ly = 68
# w = 4
# l_zig = 137
# l_arm = 24


# # ##### Real parameters
# Lx = 50
# Ly = 32
# w = 2
# l_zig = 81
# l_arm = 12


# # ##### Real parameters
# Lx = 301
# Ly = 4
# w = 2
# l_zig = 81
# l_arm = 12

# test_lt(w, l_zig, l_arm, Lx, Ly)
# test_difference(w, l_zig, l_arm, Lx, Ly)

# # list_of_gamma = [0.0, 0.1, 0.25, 0.5, 0.75, 1, 2, 3]
# # list_of_gamma = [4, 6]
# # list_of_gamma = [0.5, 1, 2, 3, 4, 5, 6]
# # list_of_gamma = [0.01]
# list_of_gamma = [1]
# list_of_U = [1, 0.5]
# list_of_dope = [0, 2]

# for ele in list_of_dope:
#     dope = ele
#     nf = 1 + dope/(Lx * Ly)
#     for U_j in tqdm(list_of_U, desc='U',  leave=True):
#         U = U_j
#         # for gamma_j in tqdm(list_of_gamma, desc='Gamma',  leave=True):
#         for gamma_i in tqdm(list_of_gamma, desc='Gamma',  leave=True):
#             # gamma = gamma_j*U
#             gamma = gamma_i
#             gamma_j = gamma/U
#             t0 = time()
#             realization = 5
#             output = cp.zeros(realization)
#             for i in range(realization):
#                 eigen_val, eigen_vec = HF_vec(Lx, Ly)
#                 re = entropy_TEE(w, l_zig, l_arm, Lx, Ly, spin_up=False)
#                 print(re) 
#                 output[i] = re
#             print(np.mean(output))
#             cp.savetxt('dope'+str(dope)+'_U'+str(U)+'_gamma'+str(gamma_j)+ '_g' +str(gamma) +'.txt', output)
#             print(output)
#             print('Time Elapsed:\t{:.4f}'.format(time() - t0))





