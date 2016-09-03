using Optimize
using NLPModels
using AmplNLReader
using OptimizationProblems

models = [AmplModel("dixmaanj.nl"), JuMPNLPModel(dixmaanj())]
solvers = [:trunk, :lbfgs]

for model in models
  for solver in solvers
    stats = run_solver(solver, model, verbose=true)
    assert(all([stats...] .>= 0))
    reset!(model)
  end
end
