using Compat.Test, NLPModels, NLPModelsJuMP, OptimizationProblems, Optimize
using LinearAlgebra
using SparseArrays

include("simple_dixmaanj.jl")

models = [simple_dixmaanj(),
          MathProgNLPModel(dixmaanj(), name="dixmaanj"),
          ADNLSModel(x->[x[1] - 1; 10 * (x[2] - x[1]^2)], [-1.2; 1.0], 2)
         ]
@static if Sys.isunix()
  using CUTEst
  push!(models, CUTEstModel("DIXMAANJ", "-param", "M=30"))
end
solvers = Dict{Symbol,Function}(:trunk => trunk, :lbfgs => lbfgs, :tron => tron)

for model in models
  for (name, solver) in solvers
    println(solver)
    stats = solve_problem(solver, model, colstats=uncstats)
    println(stats)
    @assert stats.status == :first_order
    reset!(model)
  end
  finalize(model)
end

# test benchmark helpers, skip constrained problems (hs7 has constraints)
solve_problem(trunk, simple_dixmaanj(), monotone=false, colstats=uncstats)
probs = [dixmaane, dixmaanf, dixmaang, dixmaanh, dixmaani, dixmaanj, hs7]

models = (MathProgNLPModel(p(99), name=string(p)) for p in probs)
stats = bmark_solvers(solvers, models, skipif=m -> m.meta.ncon > 0, colstats=uncstats)
@assert(size(stats[:trunk], 1) == length(probs) - 1)
stats = bmark_solvers(solvers, models, skipif=m -> m.meta.ncon > 0, prune=false)
@assert(size(stats[:trunk], 1) == length(probs))

# test bmark_solvers with CUTEst
@static if Sys.isunix()
  models = (isa(p, String) ? CUTEstModel(p) : CUTEstModel(p...) for p in ["ROSENBR", ("DIXMAANJ", "-param", "M=30")])
  stats = bmark_solvers(solvers, models)
  println(stats)
end

# Test TRON
include("solvers/tron.jl")

# Test ExecutionStats
include("test_stats.jl")
