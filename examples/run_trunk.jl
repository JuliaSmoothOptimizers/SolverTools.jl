using Optimize
using NLPModels
using OptimizationProblems

model = MathProgNLPModel(dixmaanj())
stats = run_solver(:trunk, model, verbose=true)
