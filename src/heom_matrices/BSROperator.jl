export BSROperator

"""
    struct BSROperator <: AbstractSciMLOperator

A SciML-compatible operator wrapper for BlockSparseRowMatrix.

This allows BSR matrices to be used with SciML's ODE solvers, linear solvers,
and automatic differentiation capabilities.

# Fields
- `bsr::BlockSparseRowMatrix` : the underlying BSR matrix
- `cache::Union{Nothing, SparseMatrixCSC{ComplexF64,Int64}}` : cached full sparse matrix for operations
- `iscached::Bool` : whether the cache is populated
- `isconstant::Bool` : whether the operator is time-independent
"""
mutable struct BSROperator <: AbstractSciMLOperator
    bsr::BlockSparseRowMatrix
    cache::Union{Nothing,SparseMatrixCSC{ComplexF64,Int64}}
    iscached::Bool
    isconstant::Bool
    
    function BSROperator(bsr::BlockSparseRowMatrix; isconstant::Bool = true)
        new(bsr, nothing, false, isconstant)
    end
end

# Required SciMLOperators interface
Base.size(op::BSROperator) = size(op.bsr)
Base.size(op::BSROperator, dim::Int) = size(op.bsr, dim)
Base.eltype(::BSROperator) = ComplexF64
SciMLOperators.iscached(op::BSROperator) = op.iscached
SciMLOperators.isconstant(op::BSROperator) = op.isconstant

"""
    cache_operator!(op::BSROperator)

Populate the cache with a full sparse matrix representation.
"""
function cache_operator!(op::BSROperator)
    if !op.iscached
        op.cache = to_sparse(op.bsr)
        op.iscached = true
    end
    return op
end

"""
    get_cached(op::BSROperator)

Get the cached sparse matrix, creating it if necessary.
"""
function get_cached(op::BSROperator)
    cache_operator!(op)
    return op.cache
end

# Matrix-vector multiplication
function LinearAlgebra.mul!(y::AbstractVector, op::BSROperator, x::AbstractVector)
    # Use block-level multiplication for efficiency
    B = op.bsr
    fill!(y, zero(ComplexF64))
    
    @inbounds for block_col in 1:B.ncols
        # Get the range of elements in x for this block column
        x_start = (block_col - 1) * B.block_size + 1
        x_end = block_col * B.block_size
        x_block = view(x, x_start:x_end)
        
        # Multiply each block in this column with x_block
        for k in B.block_colptr[block_col]:(B.block_colptr[block_col+1]-1)
            block_row = B.block_rowval[k]
            block = B.unique_blocks[B.block_indices[k]]
            
            # Get the range of elements in y for this block row
            y_start = (block_row - 1) * B.block_size + 1
            y_end = block_row * B.block_size
            y_block = view(y, y_start:y_end)
            
            # y_block += block * x_block
            mul!(y_block, block, x_block, one(ComplexF64), one(ComplexF64))
        end
    end
    
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, op::BSROperator, x::AbstractVector, α::Number, β::Number)
    if β == 0
        fill!(y, zero(ComplexF64))
    elseif β != 1
        y .*= β
    end
    
    B = op.bsr
    
    @inbounds for block_col in 1:B.ncols
        x_start = (block_col - 1) * B.block_size + 1
        x_end = block_col * B.block_size
        x_block = view(x, x_start:x_end)
        
        for k in B.block_colptr[block_col]:(B.block_colptr[block_col+1]-1)
            block_row = B.block_rowval[k]
            block = B.unique_blocks[B.block_indices[k]]
            
            y_start = (block_row - 1) * B.block_size + 1
            y_end = block_row * B.block_size
            y_block = view(y, y_start:y_end)
            
            # y_block += α * block * x_block
            mul!(y_block, block, x_block, α, one(ComplexF64))
        end
    end
    
    return y
end

# Standard matrix-vector multiplication
Base.:*(op::BSROperator, x::AbstractVector) = mul!(similar(x), op, x)

# Matrix-matrix operations (less efficient, uses cached version)
function Base.:*(op::BSROperator, A::AbstractMatrix)
    return get_cached(op) * A
end

function Base.:*(A::AbstractMatrix, op::BSROperator)
    return A * get_cached(op)
end

# Addition and subtraction with other operators
function Base.:+(op1::BSROperator, op2::BSROperator)
    # For now, convert to sparse and add
    # TODO: Could implement block-level addition for efficiency
    return MatrixOperator(get_cached(op1) + get_cached(op2))
end

function Base.:+(op::BSROperator, A::AbstractMatrix)
    return MatrixOperator(get_cached(op) + A)
end

Base.:+(A::AbstractMatrix, op::BSROperator) = op + A

function Base.:-(op1::BSROperator, op2::BSROperator)
    return MatrixOperator(get_cached(op1) - get_cached(op2))
end

function Base.:-(op::BSROperator, A::AbstractMatrix)
    return MatrixOperator(get_cached(op) - A)
end

function Base.:-(A::AbstractMatrix, op::BSROperator)
    return MatrixOperator(A - get_cached(op))
end

# Scalar multiplication
Base.:*(α::Number, op::BSROperator) = MatrixOperator(α * get_cached(op))
Base.:*(op::BSROperator, α::Number) = α * op

# Element access (delegates to BSR matrix)
Base.getindex(op::BSROperator, i::Int, j::Int) = op.bsr[i, j]

# Convert to different formats
function SciMLOperators.concretize(op::BSROperator)
    return get_cached(op)
end

# For linear solves, provide the sparse matrix
function LinearAlgebra.factorize(op::BSROperator)
    return factorize(get_cached(op))
end

function Base.show(io::IO, op::BSROperator)
    println(io, "BSROperator wrapping:")
    show(io, op.bsr)
end

"""
    MatrixOperator(bsr::BlockSparseRowMatrix; kwargs...)

Create a MatrixOperator from a BlockSparseRowMatrix by converting to sparse format.
"""
function SciMLOperators.MatrixOperator(bsr::BlockSparseRowMatrix; kwargs...)
    return MatrixOperator(to_sparse(bsr); kwargs...)
end

"""
    BSROperator(bsr::BlockSparseRowMatrix; isconstant=true)

Create a BSROperator from a BlockSparseRowMatrix.
"""
BSROperator(bsr::BlockSparseRowMatrix; isconstant::Bool = true) = BSROperator(bsr, isconstant = isconstant)
