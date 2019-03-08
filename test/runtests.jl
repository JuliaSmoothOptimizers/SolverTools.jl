# This package
using SolverTools

# Auxiliary packages
using NLPModels

# stdlib
using LinearAlgebra, Logging, Test

include("dummy_solver.jl")

include("test_stats.jl")
include("test_logging.jl")
