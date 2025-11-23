def bin(x,N):
    '''
    Return the binary representation of an integer with fixed length.

    Parameters
    ----------
    x : int
        Integer to convert to binary.
    N : int
        Total number of bits in the output string.

    Returns
    -------
    str
        Binary string of length `N` representing `x` in two-state basis.
    '''
    return f'{x:0{N}b}'

def spin_x(state,index):
    '''
    Flip the spin at a given site in a bitstring representation.

    Parameters
    ----------
    state : int
        Integer whose bits encode a spin configuration.
    index : int
        Position of the bit (spin) to flip. Bit indexing is LSB = site 0.

    Returns
    -------
    int
        New integer representing the configuration with the bit at `index`
        flipped. Equivalent to applying σˣ on that site.
    '''
    return state ^ (1<<index)

def spin_z(state,index):
    '''
    Find the spin at a given site in a bitstring representation.

    Parameters
    ----------
    state : int
        Integer whose bits encode a spin configuration.
    index : int
        Site index (bit position) to read. LSB = site 0.
    
    Returns
    -------
    int
        +1 if the bit at `index` is 1, otherwise −1.
        (Interpreting 1 ≡ spin up, 0 ≡ spin down.)
    '''
    return 1 if (state & (1<<index)) != 0 else -1

def gosper_basis(N,N_ones):
    '''
    Generate the bases with N sites and N_ones number of particles with 
    local dimension 1.

    Parameters
    ----------
    N : int
        Total number of sites.
    N_ones : int
        Number of up-spins (or set bits).

    Returns
    -------
    list of int
        Sorted list of integers whose binary form has exactly N_ones ones.
        These correspond to the fixed Hamming-weight subspace.
    '''
    states = []
    # --- Gosper's hack ---
    x = (1<<(N_ones)) - 1

    while x < 2**N:
        states.append(x)
        # Gosper's hack to get next integer with same number of 1s
        c = x & -x
        r = x + c
        x = (((r ^ x) >> 2) // c) | r
    
    states.sort()

    return states

def vec_norm(x):
    '''
    Compute the norm of a vector.

    Parameters
    ----------
    x : array_like
        Input vector.

    Returns
    -------
    float
        Euclidean norm of the vector, i.e. sqrt(sum(x_i^2)).
    '''
    return np.sqrt(np.sum(np.square(x)))


def lanczos(H,L = 20,phi = None,return_basis = False):
    '''
    Perform the Lanczos iteration to tridiagonalize a Hermitian matrix.

    Parameters
    ----------
    H : ndarray or sparse matrix
        Hermitian (or real symmetric) matrix representing the Hamiltonian.
        Its shape determines the Hilbert-space dimension used in the algorithm.
    L : int, optional
        Maximum number of Lanczos iterations (size of Krylov subspace).
        The actual number used is ``min(2**H.shape[0], L)``.
    phi : ndarray, optional
        Initial vector for the Lanczos procedure. If None, a random
        normalized vector of size ``2**N`` is generated. Must be a column
        vector.
    return_basis : bool, optional
        If True, also return the list of Lanczos basis vectors.

    Returns
    -------
    T : scipy.sparse.spmatrix
        Tridiagonal matrix in the Lanczos basis, with diagonal
        `diag` and off-diagonal `offdiag` elements.
    diag : ndarray
        Array of diagonal elements of the tridiagonal matrix.
    offdiag : list of float
        List of off-diagonal elements (the Lanczos βₖ coefficients).
    basis : list of ndarray, optional
        Only returned if ``return_basis=True``. Contains the Lanczos
        orthonormal basis vectors (Krylov vectors).
    '''
    
    #L should not be more than basis size
    L = min(2**H.shape[0],L)
    
    #Generating a Random Normalised Vector - psi will be our Krylov Space
    if(phi is None):
        phi = np.array([np.random.uniform(size = 2**N)])
    #psi = np.array(np.ones(2**N))
    phi = [phi.T/vec_norm(phi)]
    
    #List of Diagonal and Off-diagonal terms
    diag,offdiag = [0],[0]

    #First Iteration
    
    phi.append(H@phi[0]) #Applying Hamiltonian
    
    diag.append(phi[0].T@phi[1]) #Finding Diagonal Term
    
    phi[1] -= diag[1]*phi[0] #New vector must be Orthogonal
    # offdiag.append(norm(psi[1])) #Offdiagonal term is the Norm of the next Vector
    
    for i in range(2,L+1):
        
        offdiag.append(vec_norm(phi[i-1])) #Offdiagonal term is the Norm of the current Vector
        assert offdiag[i-1] != 0 ,"Offdiagonal term is zero"
        phi[i-1] /= offdiag[i-1] #Normalising the Orthogonal Vector
        
        phi.append(H@phi[i-1]) #New vector which is orthogonal but not normal

        diag.append(phi[i-1].T @ phi[i]) 

        phi[i] = phi[i] - diag[i-1]*phi[i-1] - offdiag[i-2]*phi[i-2] 

        phi[i] /= vec_norm(phi[i])

    diag = np.array(diag[1:]).flatten()
    offdiag = offdiag[1:]
    
    #print(diag)
    if(return_basis):
        return sp.sparse.diags([offdiag,diag,offdiag],offsets = [-1,0,1]),diag, offdiag, phi[:-1]
    return sp.sparse.diags([offdiag,diag,offdiag],offsets = [-1,0,1]),diag, offdiag