export M_Boson

@doc raw"""
    struct M_Boson <: AbstractHEOMLSMatrix
HEOM Liouvillian superoperator matrix for bosonic bath

# Fields
- `data<:AbstractSciMLOperator` : the matrix of HEOM Liouvillian superoperator
- `tier` : the tier (cutoff level) for the bosonic hierarchy
- `dimensions` : the dimension list of the coupling operator (should be equal to the system dimensions).
- `N` : the number of total ADOs
- `sup_dim` : the dimension of system superoperator
- `parity` : the parity label of the operator which HEOMLS is acting on (usually `EVEN`, only set as `ODD` for calculating spectrum of fermionic system).
- `bath::Vector{BosonBath}` : the vector which stores all `BosonBath` objects
- `hierarchy::HierarchyDict`: the object which contains all dictionaries for boson-bath-ADOs hierarchy.

!!! note "`dims` property"
    For a given `M::M_Boson`, `M.dims` or `getproperty(M, :dims)` returns its `dimensions` in the type of integer-vector.
"""
struct M_Boson{T<:AbstractSciMLOperator} <: AbstractHEOMLSMatrix{T}
    data::T
    tier::Int
    dimensions::Dimensions
    N::Int
    sup_dim::Int
    parity::AbstractParity
    bath::Vector{BosonBath}
    hierarchy::HierarchyDict
end

function M_Boson(
    Hsys::QuantumObject,
    tier::Int,
    Bath::BosonBath,
    parity::AbstractParity = EVEN;
    threshold::Real = 0.0,
    verbose::Bool = true,
)
    return M_Boson(Hsys, tier, [Bath], parity, threshold = threshold, verbose = verbose)
end

@doc raw"""
    M_Boson(Hsys, tier, Bath, parity=EVEN; threshold=0.0, verbose=true)
Generate the boson-type HEOM Liouvillian superoperator matrix

# Parameters
- `Hsys` : The time-independent system Hamiltonian or Liouvillian
- `tier::Int` : the tier (cutoff level) for the bosonic bath
- `Bath::Vector{BosonBath}` : objects for different bosonic baths
- `parity::AbstractParity` : the parity label of the operator which HEOMLS is acting on (usually `EVEN`, only set as `ODD` for calculating spectrum of fermionic system).
- `threshold::Real` : The threshold of the importance value (see Ref. [1]). Defaults to `0.0`.
- `verbose::Bool` : To display verbose output and progress bar during the process or not. Defaults to `true`.

Note that the parity only need to be set as `ODD` when the system contains fermionic systems and you need to calculate the spectrum (density of states) of it.

[1] [Phys. Rev. B 88, 235426 (2013)](https://doi.org/10.1103/PhysRevB.88.235426)
"""
@noinline function M_Boson(
    Hsys::QuantumObject,
    tier::Int,
    Bath::Vector{BosonBath},
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

    # bosonic bath
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
        progr = Progress(Nado; enabled = verbose, desc = "[M_Boson] ", QuantumToolbox.settings.ProgressMeterKWARGS...)
    end
    @threads for idx in 1:Nado

        # boson (current level) superoperator
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

        # connect to bosonic (n+1)th- & (n-1)th- level superoperator
        mode = 0
        nvec_neigh = copy(nvec)
        for bB in baths
            for k in 1:bB.Nterm
                mode += 1
                n_k = nvec[mode]

                # connect to bosonic (n-1)th-level superoperator
                if n_k > 0
                    Nvec_minus!(nvec_neigh, mode)
                    if (threshold == 0.0) || haskey(nvec2idx, nvec_neigh)
                        idx_neigh = nvec2idx[nvec_neigh]
                        op = minus_i_D_op(bB, k, n_k)
                        L_tuple = take!(chnl)
                        add_operator!(op, L_tuple[1], L_tuple[2], L_tuple[3], Nado, idx, idx_neigh)
                        put!(chnl, L_tuple)
                    end
                    Nvec_plus!(nvec_neigh, mode)
                end

                # connect to bosonic (n+1)th-level superoperator
                if nvec.level < tier
                    Nvec_plus!(nvec_neigh, mode)
                    if (threshold == 0.0) || haskey(nvec2idx, nvec_neigh)
                        idx_neigh = nvec2idx[nvec_neigh]
                        op = minus_i_B_op(bB)
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
    return M_Boson(L_he, tier, _Hsys.dimensions, Nado, sup_dim, parity, Bath, hierarchy)
end

@doc raw"""
    M_Boson(Hsys, tier, Bath, parity=EVEN; threshold=0.0, verbose=true, use_bsr=false)
Generate the boson-type HEOM Liouvillian superoperator matrix with optional BSR format

# Parameters
- `Hsys` : The time-independent system Hamiltonian or Liouvillian
- `tier::Int` : the tier (cutoff level) for the bosonic bath
- `Bath::Vector{BosonBath}` : objects for different bosonic baths
- `parity::AbstractParity` : the parity label of the operator which HEOMLS is acting on (usually `EVEN`, only set as `ODD` for calculating spectrum of fermionic system).
- `threshold::Real` : The threshold of the importance value (see Ref. [1]). Defaults to `0.0`.
- `verbose::Bool` : To display verbose output and progress bar during the process or not. Defaults to `true`.
- `use_bsr::Bool` : Use Block Sparse Row format for memory efficiency. Defaults to `false`.

Note that the parity only need to be set as `ODD` when the system contains fermionic systems and you need to calculate the spectrum (density of states) of it.

The BSR format stores the matrix as blocks and automatically deduplicates identical blocks, which can significantly reduce memory usage for large hierarchies.

[1] [Phys. Rev. B 88, 235426 (2013)](https://doi.org/10.1103/PhysRevB.88.235426)
"""
@noinline function M_Boson(
    Hsys::QuantumObject,
    tier::Int,
    Bath::Vector{BosonBath},
    parity::AbstractParity,
    use_bsr::Bool;
    threshold::Real = 0.0,
    verbose::Bool = true,
)
    if !use_bsr
        # Use the standard CSC implementation
        return M_Boson(Hsys, tier, Bath, parity, threshold = threshold, verbose = verbose)
    end

    # check for system dimension
    _Hsys = HandleMatrixType(Hsys, "Hsys (system Hamiltonian or Liouvillian)")
    sup_dim = prod(_Hsys.dimensions)^2
    I_sup = sparse(one(ComplexF64) * I, sup_dim, sup_dim)

    # the Liouvillian operator for free Hamiltonian term
    Lsys = minus_i_L_op(_Hsys)

    # bosonic bath
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
        progr = Progress(Nado; enabled = verbose, desc = "[M_Boson BSR] ", QuantumToolbox.settings.ProgressMeterKWARGS...)
    end

    # Pre-compute unique blocks before parallel loop to avoid race conditions
    # Diagonal blocks: one for each unique sum_γ value
    diagonal_blocks = Dict{Float64,SparseMatrixCSC{ComplexF64,Int64}}()
    diagonal_blocks[0.0] = Lsys  # For level 0
    
    # Pre-compute all possible sum_γ values
    for idx in 1:Nado
        nvec = idx2nvec[idx]
        if nvec.level >= 1
            sum_γ = bath_sum_γ(nvec, baths)
            if !haskey(diagonal_blocks, sum_γ)
                diagonal_blocks[sum_γ] = Lsys - sum_γ * I_sup
            end
        end
    end

    # Off-diagonal blocks: pre-compute all unique ones
    off_diagonal_blocks = Dict{Tuple,SparseMatrixCSC{ComplexF64,Int64}}()
    mode = 0
    for bB in baths
        for k in 1:bB.Nterm
            mode += 1
            # D blocks (n-1 level connections)
            for n_k in 1:tier
                block_key = (:D, mode, n_k)
                if !haskey(off_diagonal_blocks, block_key)
                    off_diagonal_blocks[block_key] = minus_i_D_op(bB, k, n_k)
                end
            end
            # B blocks (n+1 level connections)
            block_key = (:B, mode)
            if !haskey(off_diagonal_blocks, block_key)
                off_diagonal_blocks[block_key] = minus_i_B_op(bB)
            end
        end
    end

    # Now add blocks in parallel
    @threads for idx in 1:Nado
        # boson (current level) superoperator
        nvec = idx2nvec[idx]
        if nvec.level >= 1
            sum_γ = bath_sum_γ(nvec, baths)
            op = diagonal_blocks[sum_γ]
        else
            op = diagonal_blocks[0.0]
        end
        add_block!(builder, op, idx, idx)

        # connect to bosonic (n+1)th- & (n-1)th- level superoperator
        mode = 0
        nvec_neigh = copy(nvec)
        for bB in baths
            for k in 1:bB.Nterm
                mode += 1
                n_k = nvec[mode]

                # connect to bosonic (n-1)th-level superoperator
                if n_k > 0
                    Nvec_minus!(nvec_neigh, mode)
                    if (threshold == 0.0) || haskey(nvec2idx, nvec_neigh)
                        idx_neigh = nvec2idx[nvec_neigh]
                        block_key = (:D, mode, n_k)
                        op = off_diagonal_blocks[block_key]
                        add_block!(builder, op, idx, idx_neigh)
                    end
                    Nvec_plus!(nvec_neigh, mode)
                end

                # connect to bosonic (n+1)th-level superoperator
                if nvec.level < tier
                    Nvec_plus!(nvec_neigh, mode)
                    if (threshold == 0.0) || haskey(nvec2idx, nvec_neigh)
                        idx_neigh = nvec2idx[nvec_neigh]
                        block_key = (:B, mode)
                        op = off_diagonal_blocks[block_key]
                        add_block!(builder, op, idx, idx_neigh)
                    end
                    Nvec_minus!(nvec_neigh, mode)
                end
            end
        end
        verbose && next!(progr) # trigger a progress bar update
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

    return M_Boson(L_he, tier, _Hsys.dimensions, Nado, sup_dim, parity, Bath, hierarchy)
end

_getBtier(M::M_Boson) = M.tier
_getFtier(M::M_Boson) = 0
