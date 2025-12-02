export M_Fermion

@doc raw"""
    struct M_Fermion <: AbstractHEOMLSMatrix
HEOM Liouvillian superoperator matrix for fermionic bath

# Fields
- `data<:AbstractSciMLOperator` : the matrix of HEOM Liouvillian superoperator
- `tier` : the tier (cutoff level) for the fermionic hierarchy
- `dimensions` : the dimension list of the coupling operator (should be equal to the system dimensions).
- `N` : the number of total ADOs
- `sup_dim` : the dimension of system superoperator
- `parity` : the parity label of the operator which HEOMLS is acting on (usually `EVEN`, only set as `ODD` for calculating spectrum of fermionic system).
- `bath::Vector{FermionBath}` : the vector which stores all `FermionBath` objects
- `hierarchy::HierarchyDict`: the object which contains all dictionaries for fermion-bath-ADOs hierarchy.

!!! note "`dims` property"
    For a given `M::M_Fermion`, `M.dims` or `getproperty(M, :dims)` returns its `dimensions` in the type of integer-vector.
"""
struct M_Fermion{T<:AbstractSciMLOperator} <: AbstractHEOMLSMatrix{T}
    data::T
    tier::Int
    dimensions::Dimensions
    N::Int
    sup_dim::Int
    parity::AbstractParity
    bath::Vector{FermionBath}
    hierarchy::HierarchyDict
end

function M_Fermion(
    Hsys::QuantumObject,
    tier::Int,
    Bath::FermionBath,
    parity::AbstractParity = EVEN;
    threshold::Real = 0.0,
    verbose::Bool = true,
)
    return M_Fermion(Hsys, tier, [Bath], parity, threshold = threshold, verbose = verbose)
end

@doc raw"""
    M_Fermion(Hsys, tier, Bath, parity=EVEN; threshold=0.0, verbose=true)
Generate the fermion-type HEOM Liouvillian superoperator matrix

# Parameters
- `Hsys` : The time-independent system Hamiltonian or Liouvillian
- `tier::Int` : the tier (cutoff level) for the fermionic bath
- `Bath::Vector{FermionBath}` : objects for different fermionic baths
- `parity::AbstractParity` : the parity label of the operator which HEOMLS is acting on (usually `EVEN`, only set as `ODD` for calculating spectrum of fermionic system).
- `threshold::Real` : The threshold of the importance value (see Ref. [1]). Defaults to `0.0`.
- `verbose::Bool` : To display verbose output and progress bar during the process or not. Defaults to `true`.

[1] [Phys. Rev. B 88, 235426 (2013)](https://doi.org/10.1103/PhysRevB.88.235426)
"""
@noinline function M_Fermion(
    Hsys::QuantumObject,
    tier::Int,
    Bath::Vector{FermionBath},
    parity::AbstractParity = EVEN;
    threshold::Real = 0.0,
    verbose::Bool = true,
)

    # check for system dimension
    _Hsys = HandleMatrixType(Hsys, "Hsys (system Hamiltonian or Liouvillian)")
    sup_dim = prod(_Hsys.dimensions)^2
    I_sup = sparse(one(ComplexF64) * I, sup_dim, sup_dim)

    # the Liouvillian operator for free Hamiltonian term
    Lsys = minus_i_L_op(_Hsys)

    # fermionic bath
    if verbose && (threshold > 0.0)
        print("Checking the importance value for each ADOs...")
        flush(stdout)
    end
    Nado, baths, hierarchy = genBathHierarchy(Bath, tier, _Hsys.dimensions, threshold = threshold)
    idx2nvec = hierarchy.idx2nvec
    nvec2idx = hierarchy.nvec2idx
    if verbose && (threshold > 0.0)
        println("[DONE]")
        flush(stdout)
    end

    # start to construct the matrix
    Nthread = nthreads()
    L_row = [Int[] for _ in 1:Nthread]
    L_col = [Int[] for _ in 1:Nthread]
    L_val = [ComplexF64[] for _ in 1:Nthread]
    chnl = Channel{Tuple{Vector{Int},Vector{Int},Vector{ComplexF64}}}(Nthread)
    foreach(i -> put!(chnl, (L_row[i], L_col[i], L_val[i])), 1:Nthread)
    if verbose
        println("Preparing block matrices for HEOM Liouvillian superoperator (using $(Nthread) threads)...")
        flush(stdout)
        progr = Progress(Nado; enabled = verbose, desc = "[M_Fermion] ", QuantumToolbox.settings.ProgressMeterKWARGS...)
    end
    @threads for idx in 1:Nado

        # fermion (current level) superoperator
        nvec = idx2nvec[idx]
        if nvec.level >= 1
            sum_γ = bath_sum_γ(nvec, baths)
            op = Lsys - sum_γ * I_sup
        else
            op = Lsys
        end
        L_tuple = take!(chnl)
        add_operator!(op, L_tuple[1], L_tuple[2], L_tuple[3], Nado, idx, idx)
        put!(chnl, L_tuple)

        # connect to fermionic (n+1)th- & (n-1)th- level superoperator
        mode = 0
        nvec_neigh = copy(nvec)
        for fB in baths
            for k in 1:fB.Nterm
                mode += 1
                n_k = nvec[mode]

                # connect to fermionic (n-1)th-level superoperator
                if n_k > 0
                    Nvec_minus!(nvec_neigh, mode)
                    if (threshold == 0.0) || haskey(nvec2idx, nvec_neigh)
                        idx_neigh = nvec2idx[nvec_neigh]
                        op = minus_i_C_op(fB, k, nvec.level, sum(nvec_neigh[1:(mode-1)]), parity)
                        L_tuple = take!(chnl)
                        add_operator!(op, L_tuple[1], L_tuple[2], L_tuple[3], Nado, idx, idx_neigh)
                        put!(chnl, L_tuple)
                    end
                    Nvec_plus!(nvec_neigh, mode)

                    # connect to fermionic (n+1)th-level superoperator
                elseif nvec.level < tier
                    Nvec_plus!(nvec_neigh, mode)
                    if (threshold == 0.0) || haskey(nvec2idx, nvec_neigh)
                        idx_neigh = nvec2idx[nvec_neigh]
                        op = minus_i_A_op(fB, nvec.level, sum(nvec_neigh[1:(mode-1)]), parity)
                        L_tuple = take!(chnl)
                        add_operator!(op, L_tuple[1], L_tuple[2], L_tuple[3], Nado, idx, idx_neigh)
                        put!(chnl, L_tuple)
                    end
                    Nvec_minus!(nvec_neigh, mode)
                end
            end
        end
        verbose && next!(progr) # trigger a progress bar update
    end
    if verbose
        print("Constructing matrix...")
        flush(stdout)
    end
    L_he = MatrixOperator(
        sparse(reduce(vcat, L_row), reduce(vcat, L_col), reduce(vcat, L_val), Nado * sup_dim, Nado * sup_dim),
    )
    if verbose
        println("[DONE]")
        flush(stdout)
    end
    return M_Fermion(L_he, tier, _Hsys.dimensions, Nado, sup_dim, parity, Bath, hierarchy)
end

@doc raw"""
    M_Fermion(Hsys, tier, Bath, parity, use_bsr; threshold=0.0, verbose=true)
Generate the fermion-type HEOM Liouvillian superoperator matrix with optional BSR format

# Parameters
- `Hsys` : The time-independent system Hamiltonian or Liouvillian
- `tier::Int` : the tier (cutoff level) for the fermionic bath
- `Bath::Vector{FermionBath}` : objects for different fermionic baths
- `parity::AbstractParity` : the parity label of the operator which HEOMLS is acting on
- `use_bsr::Bool` : Use Block Sparse Row format for memory efficiency
- `threshold::Real` : The threshold of the importance value. Defaults to `0.0`.
- `verbose::Bool` : To display verbose output and progress bar during the process or not. Defaults to `true`.

The BSR format stores the matrix as blocks and automatically deduplicates identical blocks.
"""
@noinline function M_Fermion(
    Hsys::QuantumObject,
    tier::Int,
    Bath::Vector{FermionBath},
    parity::AbstractParity,
    use_bsr::Bool;
    threshold::Real = 0.0,
    verbose::Bool = true,
)
    if !use_bsr
        return M_Fermion(Hsys, tier, Bath, parity, threshold = threshold, verbose = verbose)
    end

    # check for system dimension
    _Hsys = HandleMatrixType(Hsys, "Hsys (system Hamiltonian or Liouvillian)")
    sup_dim = prod(_Hsys.dimensions)^2
    I_sup = sparse(one(ComplexF64) * I, sup_dim, sup_dim)

    # the Liouvillian operator for free Hamiltonian term
    Lsys = minus_i_L_op(_Hsys)

    # fermionic bath
    if verbose && (threshold > 0.0)
        print("Checking the importance value for each ADOs...")
        flush(stdout)
    end
    Nado, baths, hierarchy = genBathHierarchy(Bath, tier, _Hsys.dimensions, threshold = threshold)
    idx2nvec = hierarchy.idx2nvec
    nvec2idx = hierarchy.nvec2idx
    if verbose && (threshold > 0.0)
        println("[DONE]")
        flush(stdout)
    end

    # Create BSR builder
    builder = BSRBuilder(sup_dim, Nado, Nado)

    # start to construct the matrix
    if verbose
        println("Preparing block matrices for HEOM Liouvillian superoperator (BSR format)...")
        flush(stdout)
        progr = Progress(Nado; enabled = verbose, desc = "[M_Fermion BSR] ", QuantumToolbox.settings.ProgressMeterKWARGS...)
    end

    # Pre-compute unique blocks
    diagonal_blocks = Dict{Float64,SparseMatrixCSC{ComplexF64,Int64}}()
    diagonal_blocks[0.0] = Lsys
    for idx in 1:Nado
        nvec = idx2nvec[idx]
        if nvec.level >= 1
            sum_γ = bath_sum_γ(nvec, baths)
            if !haskey(diagonal_blocks, sum_γ)
                diagonal_blocks[sum_γ] = Lsys - sum_γ * I_sup
            end
        end
    end

    # Off-diagonal blocks for fermions (more complex due to parity dependence)
    # We pre-compute based on unique parameter combinations
    C_blocks = Dict{Tuple,SparseMatrixCSC{ComplexF64,Int64}}()
    A_blocks = Dict{Tuple,SparseMatrixCSC{ComplexF64,Int64}}()
    
    # Pre-compute A and C blocks for all possible parameter combinations
    for idx in 1:Nado
        nvec = idx2nvec[idx]
        mode = 0
        for fB in baths
            for k in 1:fB.Nterm
                mode += 1
                n_k = nvec[mode]
                
                if n_k > 0
                    # C blocks (n-1 level)
                    n_exc = nvec.level
                    n_exc_before = sum(nvec[1:(mode-1)])
                    block_key = (:C, mode, n_exc, n_exc_before)
                    if !haskey(C_blocks, block_key)
                        C_blocks[block_key] = minus_i_C_op(fB, k, n_exc, n_exc_before, parity)
                    end
                elseif nvec.level < tier
                    # A blocks (n+1 level)
                    n_exc = nvec.level
                    n_exc_before = sum(nvec[1:(mode-1)])
                    block_key = (:A, mode, n_exc, n_exc_before)
                    if !haskey(A_blocks, block_key)
                        A_blocks[block_key] = minus_i_A_op(fB, n_exc, n_exc_before, parity)
                    end
                end
            end
        end
    end

    # Now add blocks in parallel
    @threads for idx in 1:Nado
        # fermion (current level) superoperator
        nvec = idx2nvec[idx]
        if nvec.level >= 1
            sum_γ = bath_sum_γ(nvec, baths)
            op = diagonal_blocks[sum_γ]
        else
            op = diagonal_blocks[0.0]
        end
        add_block!(builder, op, idx, idx)

        # connect to fermionic (n+1)th- & (n-1)th- level superoperator
        mode = 0
        nvec_neigh = copy(nvec)
        for fB in baths
            for k in 1:fB.Nterm
                mode += 1
                n_k = nvec[mode]

                # connect to fermionic (n-1)th-level superoperator
                if n_k > 0
                    Nvec_minus!(nvec_neigh, mode)
                    if (threshold == 0.0) || haskey(nvec2idx, nvec_neigh)
                        idx_neigh = nvec2idx[nvec_neigh]
                        n_exc = nvec.level
                        n_exc_before = sum(nvec_neigh[1:(mode-1)])
                        block_key = (:C, mode, n_exc, n_exc_before)
                        op = C_blocks[block_key]
                        add_block!(builder, op, idx, idx_neigh)
                    end
                    Nvec_plus!(nvec_neigh, mode)

                    # connect to fermionic (n+1)th-level superoperator
                elseif nvec.level < tier
                    Nvec_plus!(nvec_neigh, mode)
                    if (threshold == 0.0) || haskey(nvec2idx, nvec_neigh)
                        idx_neigh = nvec2idx[nvec_neigh]
                        n_exc = nvec.level
                        n_exc_before = sum(nvec_neigh[1:(mode-1)])
                        block_key = (:A, mode, n_exc, n_exc_before)
                        op = A_blocks[block_key]
                        add_block!(builder, op, idx, idx_neigh)
                    end
                    Nvec_minus!(nvec_neigh, mode)
                end
            end
        end
        verbose && next!(progr)
    end

    # Build the BSR matrix
    bsr_matrix = build_bsr_matrix(builder; verbose = verbose)
    
    if verbose
        println("BSR construction complete:")
        println("  Total blocks: $(nnz_blocks(bsr_matrix))")
        println("  Unique blocks stored: $(n_unique_blocks(bsr_matrix))")
        println("  Memory savings: $(round((1 - memory_savings(bsr_matrix)) * 100, digits=2))% reduction")
        flush(stdout)
    end

    # Wrap in BSROperator for SciML compatibility
    L_he = BSROperator(bsr_matrix; isconstant = true)

    return M_Fermion(L_he, tier, _Hsys.dimensions, Nado, sup_dim, parity, Bath, hierarchy)
end

_getBtier(M::M_Fermion) = 0
_getFtier(M::M_Fermion) = M.tier
