using Optimize
using NLPModels
using OptimizationProblems
using MiniLogging

MiniLogging.basic_config(MiniLogging.INFO; date_format="%Y-%m-%d %H:%M:%S")
trunklogger = get_logger("optimize.trunk")

model = MathProgNLPModel(dixmaanj())
stats = solve_problem(trunk, model)
