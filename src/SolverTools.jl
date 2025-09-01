module SolverTools

using LinearAlgebra: LinearAlgebra, /, BLAS, mul!, norm, tr
using LinearOperators: LinearOperators, AbstractLinearOperator
using NLPModels:
  NLPModels, AbstractNLPModel, AbstractNLSModel, Counters, NLPModelMeta, hprod, hprod!, residual!
using Printf: Printf, @printf

# Auxiliary.
include("auxiliary/blas.jl")
include("auxiliary/slope.jl")
include("auxiliary/bounds.jl")

# Algorithmic components.
include("linesearch/linesearch.jl")
include("trust-region/trust-region.jl")

end
