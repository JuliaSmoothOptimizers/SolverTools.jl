export steihaug

"""Steihaug method for finding an approximate solution for

    min q(d) = ½dᵀBd + dᵀg
    s.t ‖d‖ ≦ Δ

where B is a symmetric matrix, not necessarily positive definite.

Use with

    z = steihaug(B, g, Δ)
"""
function steihaug(B ::
  Union{Matrix,SparseMatrixCSC,AbstractLinearOperator,LBFGSOperator},
    g :: Vector, Δ :: Real; I = [], kmax = 1000, ϵ = 1e-5)
  n = length(g)
  restricted = (length(I) == 0 || length(I) == n) ? false : true
  z = zeros(n)
  z₊ = zeros(n)
  r = copy(g)
  restricted && (r[I] = 0)
  d = -r
  rtr = dot(r,r)
  k = 0
  while (k < kmax)
    Bd = B*d
    restricted && (Bd[I] = 0)
    dtd = dot(d, d)
    dtBd = dot(d, Bd)
    if dtBd <= 1e-12dtd
      dtz = dot(d, z)
      ztz = dot(z, z)
      τ = (-dtz + sqrt(dtz^2 - dtd*(ztz-Δ^2)))/dtd
      return z + τ*d, :negative
    end
    α = rtr/dtBd
    copy!(z₊, z)
    BLAS.axpy!(α, d, z₊)
    if dot(z₊,z₊) > Δ^2
      dtz = dot(d, z)
      ztz = dot(z, z)
      τ = (-dtz + sqrt(dtz^2 - dtd*(ztz-Δ^2)))/dtd
      return z + τ*d, :outside
    end
    copy!(z, z₊)
    BLAS.axpy!(α, Bd, r)
    rtr₊ = dot(r,r)
    if norm(r)/norm(g) < ϵ
      return z, :optimal
    end
    β = rtr₊/rtr
    scale!(d, β)
    BLAS.axpy!(-1.0, r, d)
    rtr = rtr₊
    k += 1
  end
  return z, :maxiter
end

function steihaug(Hv :: Function, g :: Vector, x :: Vector, Δ :: Real; I ::
    Array{Int} = [], kmax = 1000, ϵ = 1e-5)
  n = length(g)
  restricted = (length(I) == 0 || length(I) == n) ? false : true
  z = zeros(n)
  r = copy(g)
  restricted && (r[I] = 0)
  d = -r
  rtr = dot(r,r)
  k = 0
  while (k < kmax)
    Bd = Hv(x, d)
    restricted && (Bd[I] = 0)
    dtd = dot(d, d)
    dtBd = dot(d, Bd)
    if dtBd <= 1e-12dtd
      dtz = dot(d, z)
      ztz = dot(z, z)
      τ = (-dtz + sqrt(dtz^2 - dtd*(ztz-Δ^2)))/dtd
      return z + τ*d, :negative
    end
    α = rtr/dtBd
    z₊ = z + α*d
    if dot(z₊,z₊) > Δ^2
      dtz = dot(d, z)
      ztz = dot(z, z)
      τ = (-dtz + sqrt(dtz^2 - dtd*(ztz-Δ^2)))/dtd
      return z + τ*d, :outside
    end
    copy!(z, z₊)
    r = r + α*Bd
    rtr₊ = dot(r,r)
    if norm(r)/norm(g) < ϵ
      return z, :optimal
    end
    β = rtr₊/rtr
    d = β*d - r
    rtr = rtr₊
    k += 1
  end
  return z, :maxiter
end
