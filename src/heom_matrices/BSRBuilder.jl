export BSRBuilder, add_block!, build_bsr_matrix, build_bsr_from_coo_blocks

"""
    struct BSRBuilder

A builder for efficiently constructing BlockSparseRowMatrix with block deduplication.

This builder collects blocks during construction and automatically deduplicates 
identical blocks to minimize memory usage.

# Fields
- `block_size::Int` : size of each square block
- `nrows::Int` : number of block rows
- `ncols::Int` : number of block columns
- `blocks::Dict{Tuple{Int,Int}, SparseMatrixCSC{ComplexF64,Int64}}` : temporary storage for blocks by (row, col)
- `block_cache::Dict{UInt64, Int}` : maps block hash to unique_blocks index for deduplication
"""
mutable struct BSRBuilder
    block_size::Int
    nrows::Int
    ncols::Int
    blocks::Dict{Tuple{Int,Int},SparseMatrixCSC{ComplexF64,Int64}}
    block_cache::Dict{UInt64,Tuple{Int,SparseMatrixCSC{ComplexF64,Int64}}}  # hash -> (count, block)
    
    function BSRBuilder(block_size::Int, nrows::Int, ncols::Int)
        @assert block_size > 0 "block_size must be positive"
        @assert nrows > 0 "nrows must be positive"
        @assert ncols > 0 "ncols must be positive"
        new(block_size, nrows, ncols, Dict{Tuple{Int,Int},SparseMatrixCSC{ComplexF64,Int64}}(), 
            Dict{UInt64,Tuple{Int,SparseMatrixCSC{ComplexF64,Int64}}}())
    end
end

"""
    block_hash(block::SparseMatrixCSC{ComplexF64,Int64})

Compute a hash for a sparse block matrix for deduplication.
"""
function block_hash(block::SparseMatrixCSC{ComplexF64,Int64})
    # Hash based on the structure and values
    h = hash(size(block))
    h = hash(block.colptr, h)
    h = hash(block.rowval, h)
    h = hash(block.nzval, h)
    return h
end

"""
    blocks_equal(a::SparseMatrixCSC{ComplexF64,Int64}, b::SparseMatrixCSC{ComplexF64,Int64})

Check if two sparse blocks are exactly equal.
"""
function blocks_equal(a::SparseMatrixCSC{ComplexF64,Int64}, b::SparseMatrixCSC{ComplexF64,Int64})
    size(a) == size(b) || return false
    a.colptr == b.colptr || return false
    a.rowval == b.rowval || return false
    
    # Check values with tolerance for floating point comparison
    length(a.nzval) == length(b.nzval) || return false
    for i in 1:length(a.nzval)
        if abs(a.nzval[i] - b.nzval[i]) > 1e-14
            return false
        end
    end
    return true
end

"""
    add_block!(builder::BSRBuilder, row::Int, col::Int, block::SparseMatrixCSC{ComplexF64,Int64})

Add a block to the builder at block position (row, col).
Note: This function is not thread-safe. Use only in single-threaded context or ensure external synchronization.
"""
function add_block!(builder::BSRBuilder, row::Int, col::Int, block::SparseMatrixCSC{ComplexF64,Int64})
    @assert 1 <= row <= builder.nrows "row index out of bounds"
    @assert 1 <= col <= builder.ncols "col index out of bounds"
    @assert size(block) == (builder.block_size, builder.block_size) "block size mismatch"
    
    # Store the block (not thread-safe - blocks Dict can have race conditions)
    # In practice, each ADO writes to a unique set of blocks, so this should be safe
    builder.blocks[(row, col)] = block
    
    return nothing
end

"""
    add_block!(builder::BSRBuilder, row::Int, col::Int, block::AbstractMatrix)

Add a dense block to the builder at block position (row, col), converting to sparse.
"""
function add_block!(builder::BSRBuilder, row::Int, col::Int, block::AbstractMatrix)
    add_block!(builder, row, col, sparse(ComplexF64.(block)))
end

"""
    build_bsr_matrix(builder::BSRBuilder; verbose::Bool=false)

Build the final BlockSparseRowMatrix from the builder, performing block deduplication.
"""
function build_bsr_matrix(builder::BSRBuilder; verbose::Bool = false)
    if verbose
        println("Building BSR matrix with deduplication...")
    end
    
    # First pass: deduplicate blocks
    unique_blocks = SparseMatrixCSC{ComplexF64,Int64}[]
    block_to_index = Dict{UInt64,Vector{Tuple{Int,SparseMatrixCSC{ComplexF64,Int64}}}}()
    
    # Group blocks by hash
    for ((row, col), block) in builder.blocks
        h = block_hash(block)
        if !haskey(block_to_index, h)
            block_to_index[h] = Tuple{Int,SparseMatrixCSC{ComplexF64,Int64}}[]
        end
        push!(block_to_index[h], (length(unique_blocks) + 1, block))
    end
    
    # For each hash group, find truly unique blocks
    hash_to_unique_idx = Dict{UInt64,Dict{Int,Int}}()  # hash -> (temp_idx -> unique_idx)
    
    for (h, blocks_list) in block_to_index
        hash_to_unique_idx[h] = Dict{Int,Int}()
        processed = Set{Int}()
        
        for (temp_idx, block) in blocks_list
            temp_idx in processed && continue
            
            # Check if this block matches any already-unique block with same hash
            matched = false
            for (other_idx, other_block) in blocks_list
                if other_idx < temp_idx && other_idx in processed
                    if blocks_equal(block, other_block)
                        # Use the same unique index as the other block
                        hash_to_unique_idx[h][temp_idx] = hash_to_unique_idx[h][other_idx]
                        matched = true
                        break
                    end
                end
            end
            
            if !matched
                # This is a new unique block
                push!(unique_blocks, block)
                hash_to_unique_idx[h][temp_idx] = length(unique_blocks)
            end
            
            push!(processed, temp_idx)
        end
    end
    
    if verbose
        n_total = length(builder.blocks)
        n_unique = length(unique_blocks)
        println("  Total blocks: $n_total")
        println("  Unique blocks: $n_unique")
        println("  Deduplication ratio: $(round(100 * (1 - n_unique / n_total), digits=2))%")
    end
    
    # Build the CSC-style structure for blocks
    # Sort blocks by column then row for CSC format
    sorted_blocks = sort(collect(builder.blocks), by = x -> (x[1][2], x[1][1]))
    
    block_rowval = Int[]
    block_colptr = ones(Int, builder.ncols + 1)
    block_indices = Int[]
    
    current_col = 1
    block_colptr[1] = 1
    
    for ((row, col), block) in sorted_blocks
        # When we move to a new column, update pointers for all columns in between
        while current_col < col
            block_colptr[current_col + 1] = length(block_rowval) + 1
            current_col += 1
        end
        
        # Find the unique index for this block
        h = block_hash(block)
        # We need to find which temp_idx this block corresponds to
        # by checking all blocks with this hash
        unique_idx = -1
        for (temp_idx, stored_block) in block_to_index[h]
            if blocks_equal(block, stored_block)
                unique_idx = hash_to_unique_idx[h][temp_idx]
                break
            end
        end
        
        @assert unique_idx > 0 "Failed to find unique index for block at ($row, $col)"
        
        push!(block_rowval, row)
        push!(block_indices, unique_idx)
    end
    
    # Finish colptr for remaining columns
    while current_col <= builder.ncols
        block_colptr[current_col + 1] = length(block_rowval) + 1
        current_col += 1
    end
    
    return BlockSparseRowMatrix(
        builder.block_size,
        builder.nrows,
        builder.ncols,
        unique_blocks,
        block_rowval,
        block_colptr,
        block_indices,
    )
end

"""
    build_bsr_from_coo_blocks(
        block_size::Int,
        nrows::Int, 
        ncols::Int,
        block_rows::Vector{Int},
        block_cols::Vector{Int},
        blocks::Vector{SparseMatrixCSC{ComplexF64,Int64}};
        verbose::Bool=false
    )

Build a BSR matrix directly from COO-format block data.
"""
function build_bsr_from_coo_blocks(
    block_size::Int,
    nrows::Int,
    ncols::Int,
    block_rows::Vector{Int},
    block_cols::Vector{Int},
    blocks::Vector{SparseMatrixCSC{ComplexF64,Int64}};
    verbose::Bool = false,
)
    @assert length(block_rows) == length(block_cols) == length(blocks) "Input vectors must have same length"
    
    builder = BSRBuilder(block_size, nrows, ncols)
    
    for i in 1:length(block_rows)
        add_block!(builder, block_rows[i], block_cols[i], blocks[i])
    end
    
    return build_bsr_matrix(builder; verbose = verbose)
end
