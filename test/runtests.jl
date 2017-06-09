using Base.Test, Compat, NLPModels, OptimizationProblems, Optimize
import Compat.String

include("simple_dixmaanj.jl")

models = [simple_dixmaanj(),
          MathProgNLPModel(dixmaanj(), name="dixmaanj")]
@static if is_unix()
  using CUTEst
  push!(models, CUTEstModel("DIXMAANJ", "-param", "M=30"))
end
solvers = [trunk, lbfgs, tron]

for model in models
  for solver in solvers
    stats = solve_problem(solver, model, verbose=false)
    assert(stats.solved)
    reset!(model)
  end
  finalize(model)
end

# test benchmark helpers, skip constrained problems (hs7 has constraints)
solve_problem(trunk, simple_dixmaanj(), verbose=true, monotone=false)
probs = [dixmaane, dixmaanf, dixmaang, dixmaanh, dixmaani, dixmaanj, hs7]

models = (MathProgNLPModel(p(99), name=string(p)) for p in probs)
stats = bmark_solvers(solvers, models, skipif=m -> m.meta.ncon > 0)
p = profile_solvers(stats)
stats, p = bmark_and_profile(solvers, models, bmark_args=Dict{Symbol,Any}(:skipif=>m -> m.meta.ncon > 0))
println(stats)
println(size(stats[Symbol(solvers[1])], 1))
println(length(probs))
assert(size(stats[Symbol(solvers[1])], 1) == length(probs) - 1)
stats = bmark_solvers(solvers, models, skipif=m -> m.meta.ncon > 0, prune=false)
p = profile_solvers(stats)
stats, p = bmark_and_profile(solvers, models, bmark_args=Dict{Symbol,Any}(:skipif=>m -> m.meta.ncon > 0, :prune=>false))
assert(size(stats[Symbol(solvers[1])], 1) == length(probs))

# test bmark_solvers with CUTEst
@static if is_unix()
  models = (isa(p, String) ? CUTEstModel(p) : CUTEstModel(p...) for p in ["ROSENBR", ("DIXMAANJ", "-param", "M=30")])
  stats = bmark_solvers(solvers, models)
  p = profile_solvers(stats)
  stats, p = bmark_and_profile(solvers, models)
  println(stats)
end

# Test TRON
include("solvers/tron.jl")

