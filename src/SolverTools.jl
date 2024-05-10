module SolverTools

# stdlib
using LinearAlgebra, Printf

# our packages
using LinearOperators, NLPModels

# Auxiliary.
include("auxiliary/blas.jl")
include("auxiliary/slope.jl")
include("auxiliary/bounds.jl")

# Algorithmic components.
include("linesearch/linesearch.jl")
include("trust-region/trust-region.jl")

end
