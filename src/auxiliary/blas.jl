# BLAS
import LinearAlgebra: dot
import LinearAlgebra.BLAS: nrm2, blascopy!, axpy!, scal!

export dot, nrm2, blascopy!, axpy!, scal!, copyaxpy!

nrm2(n :: Int, x :: Vector{T}) where T <: BLAS.BlasReal = BLAS.nrm2(n, x, 1)
nrm2(n :: Int, x :: AbstractVector{T}) where T <: Number = norm(x)

dot(n :: Int, x :: Vector{T}, y :: Vector{T}) where T <: BLAS.BlasReal = BLAS.dot(n, x, 1, y, 1)
dot(n :: Int, x :: AbstractVector{T}, y :: AbstractVector{T}) where T <: Number = dot(x, y)

axpy!(n :: Int, t :: T, d :: Vector{T}, x :: Vector{T}) where T <: BLAS.BlasReal = BLAS.axpy!(n, t, d, 1, x, 1)
axpy!(n :: Int, t :: T, d :: AbstractVector{T}, x :: AbstractVector{T}) where T <: Number = (x .+= t .* d)

axpby!(n :: Int, t :: T, d :: Vector{T}, s :: T, x :: Vector{T}) where T <: BLAS.BlasReal = BLAS.axpby!(n, t, d, 1, s, x, 1)
axpby!(n :: Int, t :: T, d :: AbstractVector{T}, s :: T, x :: AbstractVector{T}) where T <: Number = (x .= t .* d .+ s .* x)

scal!(n :: Int, t :: T, x :: Vector{T}) where T <: BLAS.BlasReal = BLAS.scal!(n, t, x, 1)
scal!(n :: Int, t :: T, x :: AbstractVector{T}) where T <: Number = (x .*= t)

function copyaxpy!(n :: Int, t :: T, d :: Vector{T}, x :: Vector{T}, xt :: Vector{T}) where T <: BLAS.BlasReal
  BLAS.blascopy!(n, x, 1, xt, 1)
  BLAS.axpy!(n, t, d, 1, xt, 1)
end
function copyaxpy!(n :: Int, t :: T, d :: AbstractVector{T}, x :: AbstractVector{T}, xt :: AbstractVector{T}) where T <: Number
  xt .= x .+ t .* d
end

