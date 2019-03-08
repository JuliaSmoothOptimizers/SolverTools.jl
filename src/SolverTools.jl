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
include("auxiliary/logger.jl")
include("stats/stats.jl")

# Algorithmic components.
include("linesearch/linesearch.jl")
include("trust-region/trust-region.jl")

# Utilities.
include("bmark/bmark_solvers.jl")
include("bmark/run_solver.jl")

end
