using Optimize
using NLPModels
using OptimizationProblems

model = MathProgNLPModel(dixmaanj())
stats = solve_problem(trunk, model, verbose=true)
