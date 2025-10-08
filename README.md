# HartreeFock-ZGNR
Hartree-Fock (HF) calculations for zigzag graphene nanoribbons (ZGNR). This is the Hartree-Fock approximation treatment for the Mott-Anderson Hamiltonian 

$$
H = -t \sum_{\left< ij \right>, \sigma} 
\left( c^\dagger_{i, \sigma} c_{j, \sigma} + c^\dagger_{j, \sigma} c_{i, \sigma}   \right) +
U \sum_{i} n_{i, \uparrow} n_{i, \downarrow} +
\sum_{i} V_i n_{i, \sigma}.
$$

If you use this code, please cite the following work:
1) Phase diagram and crossover phases of topologically ordered graphene zigzag nanoribbons: role of localization effects
Hoang-Anh Le, In-Hwan Lee, Young-Heon Kim and S.-R. Eric Yang, J. Phys.: Condens. Matter 36 265604 (2024)
2) Exploring Topological Order of Zigzag Graphene Nanoribbons: Phase Diagram and Crossover Phases, Hoang Anh Le, PhD Thesis, Korea University (2024)
http://dcollection.korea.ac.kr/srch/srchDetail/000000278801

# Plan 

In terms of writing the code, consider to write it the style of **class** in Python.

Before doing so, just write it simple.

<!-- Rename it? HartreeFock_ZGNR. -->


Rememeber to add the citation to my thesis.


For each calculation type, make it in an example folder.


`HF_spinspliting.py` returns the eigensystems only

Folders:
1. ~~Geometric of the zigzag ribbon: plot the ribbon, ribbon with partitioned region (`geometry` folder: Done)~~

2. ~~**qA - energy plot** (currently building)~~

Better save in csv file for easy saving and viewing.

3. Site spins (remember to include the edge-charge correlation analysis)

4. TEE 

5. ~~Fractional charge~~

# HF_spinsplitting.py

Return eigensystems of the ribbon: `val_up, vec_up, val_dn, vec_dn`

```
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
    val_up, vec_up = cp.linalg.eigh(the_system[0])
    val_dn, vec_dn = cp.linalg.eigh(the_system[1])
    return val_up, vec_up, val_dn, vec_dn
```

**Things to do** 

~~1. Include variables (t, gamma, U, dope) to the corresponding function.~~

~~2. Doing: the geometry folder....~~

~~3. Notice user of using CUDA and not using CUDA.~~

~~4. Viewing fractional charge.~~




