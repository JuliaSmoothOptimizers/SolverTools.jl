using Optimize
using NLPModels
using AmplNLReader
using OptimizationProblems

dixmaanjfile = joinpath(dirname(@__FILE__), "dixmaanj.nl")
models = [AmplModel(dixmaanjfile), JuMPNLPModel(dixmaanj())]
@unix_only begin
  using CUTEst
  push!(models, CUTEstModel("DIXMAANJ", "-param", "M=30"))
end
solvers = [:trunk, :lbfgs]

for model in models
  for solver in solvers
    stats = run_solver(solver, model, verbose=true)
    assert(all([stats...] .>= 0))
    reset!(model)
  end
end
