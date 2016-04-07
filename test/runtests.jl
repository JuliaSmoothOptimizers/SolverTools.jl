using Optimize
using NLPModels
using OptimizationProblems

model = JuMPNLPModel(dixmaanj())
stats = run_solver(:trunk, model, verbose=true)
stats = run_solver(:lbfgs, model, verbose=true)
