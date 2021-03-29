# This package
using SolverTools

# Auxiliary packages
using ADNLPModels, NLPModels

# stdlib
using LinearAlgebra, Logging, Test

include("dummy_solver.jl")

include("test_auxiliary.jl")
include("test_linesearch.jl")
include("test_trust_region.jl")
