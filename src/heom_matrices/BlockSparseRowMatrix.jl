export BlockSparseRowMatrix

"""
    struct BlockSparseRowMatrix

Block Sparse Row (BSR) matrix format optimized for HEOMLS matrices.

This format stores the matrix as blocks where:
- Each block is `block_size × block_size` (typically `sup_dim × sup_dim`)
- Only non-zero blocks are stored
- Identical blocks are deduplicated - stored once and referenced multiple times
- Compatible with SciML operators for autodiff and linear solve

# Fields
- `block_size::Int` : size of each square block
- `nrows::Int` : number of block rows
- `ncols::Int` : number of block columns
- `unique_blocks::Vector{SparseMatrixCSC{ComplexF64, Int64}}` : unique block matrices
- `block_rowval::Vector{Int}` : block column indices (like CSC rowval but for blocks)
- `block_colptr::Vector{Int}` : block column pointers (like CSC colptr but for blocks)
- `block_indices::Vector{Int}` : indices into unique_blocks for each non-zero block
- `full_size::Tuple{Int, Int}` : total matrix dimensions (nrows*block_size, ncols*block_size)

# Structure
The matrix is stored in a CSC-like format but at the block level:
- `block_colptr[j]:block_colptr[j+1]-1` gives the range of non-zero blocks in block-column j
- `block_rowval[k]` gives the block-row index of the k-th non-zero block
- `block_indices[k]` gives the index into unique_blocks for the k-th non-zero block
- `unique_blocks[block_indices[k]]` is the actual sparse block matrix
"""
struct BlockSparseRowMatrix
    block_size::Int
    nrows::Int  # number of block rows
    ncols::Int  # number of block columns
    unique_blocks::Vector{SparseMatrixCSC{ComplexF64,Int64}}
    block_rowval::Vector{Int}
    block_colptr::Vector{Int}
    block_indices::Vector{Int}
    full_size::Tuple{Int,Int}
    
    function BlockSparseRowMatrix(
        block_size::Int,
        nrows::Int,
        ncols::Int,
        unique_blocks::Vector{SparseMatrixCSC{ComplexF64,Int64}},
        block_rowval::Vector{Int},
        block_colptr::Vector{Int},
        block_indices::Vector{Int},
    )
        # Validate inputs
        @assert block_size > 0 "block_size must be positive"
        @assert nrows > 0 "nrows must be positive"
        @assert ncols > 0 "ncols must be positive"
        @assert length(block_colptr) == ncols + 1 "block_colptr length must be ncols + 1"
        @assert length(block_rowval) == length(block_indices) "block_rowval and block_indices must have same length"
        @assert all(1 .<= block_indices .<= length(unique_blocks)) "block_indices must be valid indices into unique_blocks"
        
        # Validate that all unique blocks have the correct size
        for (i, blk) in enumerate(unique_blocks)
            @assert size(blk) == (block_size, block_size) "unique_blocks[$i] has wrong size: $(size(blk)) != ($block_size, $block_size)"
        end
        
        full_size = (nrows * block_size, ncols * block_size)
        new(block_size, nrows, ncols, unique_blocks, block_rowval, block_colptr, block_indices, full_size)
    end
end

# Basic interface
Base.size(B::BlockSparseRowMatrix) = B.full_size
Base.size(B::BlockSparseRowMatrix, dim::Int) = B.full_size[dim]
Base.eltype(::BlockSparseRowMatrix) = ComplexF64

"""
    nnz_blocks(B::BlockSparseRowMatrix)

Return the number of non-zero blocks in the BSR matrix.
"""
nnz_blocks(B::BlockSparseRowMatrix) = length(B.block_rowval)

"""
    n_unique_blocks(B::BlockSparseRowMatrix)

Return the number of unique blocks stored in the BSR matrix.
"""
n_unique_blocks(B::BlockSparseRowMatrix) = length(B.unique_blocks)

"""
    memory_savings(B::BlockSparseRowMatrix)

Return the memory savings ratio from block deduplication.
Returns the ratio: (blocks_stored / total_nonzero_blocks)
"""
function memory_savings(B::BlockSparseRowMatrix)
    total_blocks = nnz_blocks(B)
    unique_blocks = n_unique_blocks(B)
    return unique_blocks / total_blocks
end

"""
    getblock(B::BlockSparseRowMatrix, i::Int, j::Int)

Get the block at block position (i, j).
Returns the sparse block matrix or nothing if the block is zero.
"""
function getblock(B::BlockSparseRowMatrix, i::Int, j::Int)
    @boundscheck (1 <= i <= B.nrows && 1 <= j <= B.ncols) || throw(BoundsError(B, (i, j)))
    
    # Search for block in column j
    for k in B.block_colptr[j]:(B.block_colptr[j+1]-1)
        if B.block_rowval[k] == i
            return B.unique_blocks[B.block_indices[k]]
        end
    end
    return nothing
end

"""
    Base.getindex(B::BlockSparseRowMatrix, i::Int, j::Int)

Get the element at position (i, j) in the full matrix.
"""
function Base.getindex(B::BlockSparseRowMatrix, i::Int, j::Int)
    @boundscheck checkbounds(B, i, j)
    
    # Convert to block coordinates
    block_i, local_i = divrem(i - 1, B.block_size)
    block_j, local_j = divrem(j - 1, B.block_size)
    block_i += 1
    block_j += 1
    local_i += 1
    local_j += 1
    
    block = getblock(B, block_i, block_j)
    if block === nothing
        return zero(ComplexF64)
    else
        return block[local_i, local_j]
    end
end

Base.checkbounds(::Type{Bool}, B::BlockSparseRowMatrix, i::Int, j::Int) = 
    (1 <= i <= B.full_size[1]) && (1 <= j <= B.full_size[2])

function Base.checkbounds(B::BlockSparseRowMatrix, i::Int, j::Int)
    checkbounds(Bool, B, i, j) || throw(BoundsError(B, (i, j)))
end

"""
    to_sparse(B::BlockSparseRowMatrix)

Convert a BlockSparseRowMatrix to a standard SparseMatrixCSC.
"""
function to_sparse(B::BlockSparseRowMatrix)
    rows = Int[]
    cols = Int[]
    vals = ComplexF64[]
    
    # Iterate through all non-zero blocks
    for block_col in 1:B.ncols
        for k in B.block_colptr[block_col]:(B.block_colptr[block_col+1]-1)
            block_row = B.block_rowval[k]
            block = B.unique_blocks[B.block_indices[k]]
            
            # Add all non-zero entries from this block
            block_rows, block_cols, block_vals = findnz(block)
            for (local_row, local_col, val) in zip(block_rows, block_cols, block_vals)
                push!(rows, (block_row - 1) * B.block_size + local_row)
                push!(cols, (block_col - 1) * B.block_size + local_col)
                push!(vals, val)
            end
        end
    end
    
    return sparse(rows, cols, vals, B.full_size[1], B.full_size[2])
end

"""
    Base.show(io::IO, B::BlockSparseRowMatrix)

Display information about a BlockSparseRowMatrix.
"""
function Base.show(io::IO, B::BlockSparseRowMatrix)
    println(io, "BlockSparseRowMatrix:")
    println(io, "  Full size: $(B.full_size[1]) × $(B.full_size[2])")
    println(io, "  Block size: $(B.block_size) × $(B.block_size)")
    println(io, "  Block dimensions: $(B.nrows) × $(B.ncols)")
    println(io, "  Non-zero blocks: $(nnz_blocks(B))")
    println(io, "  Unique blocks: $(n_unique_blocks(B))")
    println(io, "  Memory savings: $(round(memory_savings(B) * 100, digits=2))% of blocks are unique")
end
