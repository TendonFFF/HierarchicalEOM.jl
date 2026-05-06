export HEOMLSOperator

@doc raw"""
    struct HEOMLSOperator{T, TLsys, TOps, TC} <: AbstractSciMLOperator{T}

Lazy HEOM Liouvillian superoperator, stored explicitly as structured tensor-product terms.

The full operator is:
```math
\mathcal{L} = (I_{N_{\rm ado}} \otimes L_{\rm sys})
            + (\mathrm{Diag}(\gamma) \otimes I_{d^2})
            + \sum_i (A_i \otimes B_i)
```

This representation shares a single cache buffer across all coupling terms,
replacing the cache-sharing hack previously required for `AddedOperator` of
`TensorProductOperator`s.

# Fields
- `L_sys` : ``d^2 \times d^2`` system Liouvillian (von Neumann term); parametric to support time-dependent operators
- `γ_diag` : length-``N_{\rm ado}`` diagonal for the ``\mathrm{Diag}(\gamma) \otimes I_{d^2}`` damping term
- `ops` : coupling terms as a heterogeneous `Tuple` of `(A_i, B_i)` pairs — outer ``N_{\rm ado} \times N_{\rm ado}``, inner ``d^2 \times d^2``; `B_i` may be any concrete operator type
- `Nado` : number of auxiliary density operators
- `sup_dim` : system superoperator dimension ``d^2``
- `cache` : preallocated buffer of length `Nado * sup_dim`, or `nothing` before caching
"""
struct HEOMLSOperator{T, TLsys, TOps <: Tuple, TC} <: AbstractSciMLOperator{T}
    L_sys::TLsys
    γ_diag::Vector{T}
    ops::TOps
    Nado::Int
    sup_dim::Int
    cache::TC

    function HEOMLSOperator(
            L_sys::TLsys,
            γ_diag::Vector{T},
            ops::TOps,
            Nado::Int,
            sup_dim::Int,
            cache::TC = nothing,
        ) where {T, TLsys, TOps <: Tuple, TC}
        new{T, TLsys, TOps, TC}(L_sys, γ_diag, ops, Nado, sup_dim, cache)
    end
end

# Convenience constructor: convert a Vector of (A_i, B_i) pairs to a heterogeneous Tuple
function HEOMLSOperator(
        L_sys::TLsys,
        γ_diag::Vector{T},
        ops_vec::AbstractVector,
        Nado::Int,
        sup_dim::Int,
        cache::TC = nothing,
    ) where {T, TLsys, TC}
    ops_tuple = Tuple((sparse(A_i)::SparseMatrixCSC{T,Int64}, _heomls_to_sparse(B_i)) for (A_i, B_i) in ops_vec)
    return HEOMLSOperator(L_sys, γ_diag, ops_tuple, Nado, sup_dim, cache)
end

# Extract the concrete sparse matrix from whatever wraps it
_heomls_to_sparse(B::MatrixOperator) = B.A
_heomls_to_sparse(B::AbstractMatrix) = sparse(B)

Base.size(L::HEOMLSOperator) = (L.Nado * L.sup_dim, L.Nado * L.sup_dim)

function Base.copy(L::HEOMLSOperator)
    return HEOMLSOperator(
        copy(L.L_sys),
        copy(L.γ_diag),
        L.ops,
        L.Nado,
        L.sup_dim,
        isnothing(L.cache) ? nothing : copy(L.cache),
    )
end

function Base.show(io::IO, L::HEOMLSOperator)
    n = L.Nado * L.sup_dim
    return print(io, "HEOMLSOperator($n × $n), $(length(L.ops)) coupling terms")
end
Base.show(io::IO, ::MIME"text/plain", L::HEOMLSOperator) = show(io, L)

# --- SciMLOperators traits ---

SciMLOperators.islinear(::HEOMLSOperator) = true
SciMLOperators.has_concretization(::HEOMLSOperator) = true
SciMLOperators.isconvertible(::HEOMLSOperator) = true
SciMLOperators.has_mul(::HEOMLSOperator) = true
SciMLOperators.has_mul!(::HEOMLSOperator) = true
# B_i are constant; only L_sys may be time-dependent or non-adjoint
SciMLOperators.has_adjoint(L::HEOMLSOperator) = has_adjoint(L.L_sys)
SciMLOperators.isconstant(L::HEOMLSOperator) = isconstant(L.L_sys)

function SciMLOperators.iscached(L::HEOMLSOperator)
    return !isnothing(L.cache) && iscached(L.L_sys)
end

# --- cache ---

function SciMLOperators.cache_self(L::HEOMLSOperator, v::AbstractVector)
    return HEOMLSOperator(L.L_sys, L.γ_diag, L.ops, L.Nado, L.sup_dim, similar(v))
end

function SciMLOperators.cache_internals(L::HEOMLSOperator, v::AbstractVector)
    V = reshape(v, L.sup_dim, L.Nado)
    new_L_sys = cache_operator(L.L_sys, V)
    return HEOMLSOperator(new_L_sys, L.γ_diag, L.ops, L.Nado, L.sup_dim, L.cache)
end

# --- mul! ---
# @generated unrolls the ops Tuple at compile time → static dispatch per (A_i, B_i) pair

@generated function LinearAlgebra.mul!(w::AbstractVector, L::HEOMLSOperator, v::AbstractVector)
    N = length(L.parameters[3].parameters)
    ops_stmts = Expr(:block)
    for i in 1:N
        push!(ops_stmts.args, :(mul!(C, L.ops[$i][2], V)))
        push!(ops_stmts.args, :(mul!(transpose(W), L.ops[$i][1], transpose(C), true, true)))
    end
    return quote
        @assert iscached(L) "cache needs to be set up for HEOMLSOperator. Call cache_operator(L, u) first."
        W = reshape(w, L.sup_dim, L.Nado)
        V = reshape(v, L.sup_dim, L.Nado)
        mul!(W, L.L_sys, V)
        W .+= transpose(L.γ_diag) .* V
        C = reshape(L.cache, L.sup_dim, L.Nado)
        $ops_stmts
        return w
    end
end

@generated function LinearAlgebra.mul!(w::AbstractVector, L::HEOMLSOperator, v::AbstractVector, α, β)
    N = length(L.parameters[3].parameters)
    ops_stmts = Expr(:block)
    for i in 1:N
        push!(ops_stmts.args, :(mul!(C, L.ops[$i][2], V)))
        push!(ops_stmts.args, :(mul!(transpose(W), L.ops[$i][1], transpose(C), α, true)))
    end
    return quote
        @assert iscached(L) "cache needs to be set up for HEOMLSOperator. Call cache_operator(L, u) first."
        lmul!(β, w)
        W = reshape(w, L.sup_dim, L.Nado)
        V = reshape(v, L.sup_dim, L.Nado)
        mul!(W, L.L_sys, V, α, true)
        W .+= α .* transpose(L.γ_diag) .* V
        C = reshape(L.cache, L.sup_dim, L.Nado)
        $ops_stmts
        return w
    end
end

# --- adjoint ---

function Base.adjoint(L::HEOMLSOperator{T}) where {T}
    adj_ops = map(((A_i, B_i),) -> (adjoint(A_i), adjoint(B_i)), L.ops)
    return HEOMLSOperator(adjoint(L.L_sys), conj.(L.γ_diag), adj_ops, L.Nado, L.sup_dim, L.cache)
end

# --- concretize ---

function Base.convert(::Type{AbstractMatrix}, L::HEOMLSOperator{T}) where {T}
    N, d2 = L.Nado, L.sup_dim
    I_N = spdiagm(0 => ones(T, N))::SparseMatrixCSC{T,Int64}
    L_sys_mat = sparse(convert(AbstractMatrix, L.L_sys))::SparseMatrixCSC{T,Int64}
    mat = kron(I_N, L_sys_mat) + spdiagm(repeat(L.γ_diag; inner = d2))
    for (A_i, B_i) in L.ops
        mat = mat + kron(A_i, B_i)
    end
    return mat
end

# --- update_coefficients! ---

function SciMLOperators.update_coefficients!(L::HEOMLSOperator, u, p, t; kwargs...)
    update_coefficients!(L.L_sys, u, p, t; kwargs...)
    return nothing
end
