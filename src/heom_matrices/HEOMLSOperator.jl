export HEOMLSOperator

@doc raw"""
    struct HEOMLSOperator{T, TLsys, TC} <: AbstractSciMLOperator{T}

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
- `L_sys` : ``d^2 \times d^2`` system Liouvillian (von Neumann term)
- `γ_diag` : length-``N_{\rm ado}`` diagonal for the ``\mathrm{Diag}(\gamma) \otimes I_{d^2}`` damping term
- `ops` : coupling terms as `(A_i, B_i)` pairs (sparse ``N_{\rm ado} \times N_{\rm ado}`` outer, ``d^2 \times d^2`` inner)
- `Nado` : number of auxiliary density operators
- `sup_dim` : system superoperator dimension ``d^2``
- `cache` : preallocated buffer of length `Nado * sup_dim`, or `nothing` before caching
"""
struct HEOMLSOperator{T, TLsys, TC} <: AbstractSciMLOperator{T}
    L_sys::TLsys
    γ_diag::Vector{T}
    ops::Vector{Tuple{SparseMatrixCSC{T, Int64}, AbstractSciMLOperator{T}}}
    Nado::Int
    sup_dim::Int
    cache::TC
end

Base.size(L::HEOMLSOperator) = (L.Nado * L.sup_dim, L.Nado * L.sup_dim)

function Base.copy(L::HEOMLSOperator)
    return HEOMLSOperator(
        copy(L.L_sys),
        copy(L.γ_diag),
        copy(L.ops),
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
SciMLOperators.has_adjoint(L::HEOMLSOperator) =
    has_adjoint(L.L_sys) && all(t -> has_adjoint(last(t)), L.ops)
SciMLOperators.isconstant(L::HEOMLSOperator) =
    isconstant(L.L_sys) && all(t -> isconstant(last(t)), L.ops)

function SciMLOperators.iscached(L::HEOMLSOperator)
    return !isnothing(L.cache) && iscached(L.L_sys) && all(t -> iscached(last(t)), L.ops)
end

# --- cache ---

function SciMLOperators.cache_self(L::HEOMLSOperator, v::AbstractVector)
    return HEOMLSOperator(L.L_sys, L.γ_diag, L.ops, L.Nado, L.sup_dim, similar(v))
end

function SciMLOperators.cache_internals(L::HEOMLSOperator, v::AbstractVector)
    V = reshape(v, L.sup_dim, L.Nado)
    new_L_sys = cache_operator(L.L_sys, V)
    new_ops = map(L.ops) do (A_i, B_i)
        (A_i, cache_operator(B_i, V))
    end
    return HEOMLSOperator(new_L_sys, L.γ_diag, new_ops, L.Nado, L.sup_dim, L.cache)
end

# --- mul! ---

function LinearAlgebra.mul!(w::AbstractVector, L::HEOMLSOperator, v::AbstractVector)
    @assert iscached(L) "cache needs to be set up for HEOMLSOperator. Call cache_operator(L, u) first."
    W = reshape(w, L.sup_dim, L.Nado)
    V = reshape(v, L.sup_dim, L.Nado)
    mul!(W, L.L_sys, V)
    W .+= transpose(L.γ_diag) .* V
    C = reshape(L.cache, L.sup_dim, L.Nado)
    for (A_i, B_i) in L.ops
        mul!(C, B_i, V)
        mul!(transpose(W), A_i, transpose(C), true, true)
    end
    return w
end

function LinearAlgebra.mul!(w::AbstractVector, L::HEOMLSOperator, v::AbstractVector, α, β)
    @assert iscached(L) "cache needs to be set up for HEOMLSOperator. Call cache_operator(L, u) first."
    lmul!(β, w)
    W = reshape(w, L.sup_dim, L.Nado)
    V = reshape(v, L.sup_dim, L.Nado)
    mul!(W, L.L_sys, V, α, true)
    W .+= α .* transpose(L.γ_diag) .* V
    C = reshape(L.cache, L.sup_dim, L.Nado)
    for (A_i, B_i) in L.ops
        mul!(C, B_i, V)
        mul!(transpose(W), A_i, transpose(C), α, true)
    end
    return w
end

# --- adjoint ---

function Base.adjoint(L::HEOMLSOperator{T}) where {T}
    adj_ops = map(L.ops) do (A_i, B_i)
        (copy(adjoint(A_i)), adjoint(B_i))
    end
    return HEOMLSOperator(adjoint(L.L_sys), conj.(L.γ_diag), adj_ops, L.Nado, L.sup_dim, L.cache)
end

# --- concretize ---

function Base.convert(::Type{AbstractMatrix}, L::HEOMLSOperator{T}) where {T}
    N, d2 = L.Nado, L.sup_dim
    I_N = spdiagm(0 => ones(T, N))::SparseMatrixCSC{T,Int64}
    L_sys_mat = sparse(convert(AbstractMatrix, L.L_sys))::SparseMatrixCSC{T,Int64}
    mat = kron(I_N, L_sys_mat) + spdiagm(repeat(L.γ_diag; inner = d2))
    for (A_i, B_i) in L.ops
        B_mat = sparse(convert(AbstractMatrix, B_i))::SparseMatrixCSC{T,Int64}
        mat = mat + kron(A_i, B_mat)
    end
    return mat
end

# --- update_coefficients! ---

function SciMLOperators.update_coefficients!(L::HEOMLSOperator, u, p, t; kwargs...)
    update_coefficients!(L.L_sys, u, p, t; kwargs...)
    foreach(L.ops) do (_, B_i)
        update_coefficients!(B_i, u, p, t; kwargs...)
    end
    return nothing
end
