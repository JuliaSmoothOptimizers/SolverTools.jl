# This package
using SolverTools

# Auxiliary packages
using NLPModels

# stdlib
using LinearAlgebra, Logging, Test

include("dummy_solver.jl")
include("dummy_linesearch_solver.jl")

include("test_auxiliary.jl")
include("test_linesearch.jl")
include("test_logging.jl")
include("merit.jl")
include("test_stats.jl")
include("test_trust_region.jl")