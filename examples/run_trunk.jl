using Optimize
using NLPModels
using NLPModelsJuMP
using OptimizationProblems

model = MathProgNLPModel(dixmaanj())
stats = solve_problem(trunk, model)
