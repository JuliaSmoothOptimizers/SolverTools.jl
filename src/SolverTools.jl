module SolverTools

# stdlib
using LinearAlgebra, Logging, Printf

# our packages
using LinearOperators, NLPModels

# auxiliary packages
using DataFrames

# Auxiliary.
include("auxiliary/blas.jl")
include("auxiliary/bounds.jl")
include("stats/stats.jl")

# Algorithmic components.
include("linesearch/linesearch.jl")
include("trust-region/trust-region.jl")

# Utilities.
include("bmark/run_solver.jl")
include("bmark/bmark_solvers.jl")

end
