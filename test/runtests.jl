using Test, NLPModels, NLPModelsJuMP, OptimizationProblems, Optimize, LinearAlgebra,
      SparseArrays, Logging, NLSProblems

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

@info("Testing various solvers in different models")
for model in models
  for (name, solver) in solvers
    stats = solver(model)
    @test stats.status == :first_order
    reset!(model)
  end
  finalize(model)
end
@info("Done")

@info("Logging of calling trunk directly")
nlp = ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2, [-1.2; 1.0], name="rosen")
trunk(nlp, logger=ConsoleLogger())
@info("Logging of calling solve_problems with trunk")
nlp2 = ADNLPModel(x -> x[1]^4 + x[2]^4, ones(2), name="sumquartic")
solve_problems(trunk, [nlp, nlp2], logger=ConsoleLogger())
@info("Logging of calling solve_problems with trunk and colstats and solver logger")
solve_problems(trunk, [nlp, nlp2], logger=ConsoleLogger(),
               solver_logger=ConsoleLogger(), colstats=[:name, :objective])
@info("Done")

# test benchmark helpers, skip constrained problems (hs7 has constraints)
trunk(simple_dixmaanj(), monotone=false)
probs = [dixmaane, dixmaanf, dixmaang, dixmaanh, dixmaani, dixmaanj, hs7]

models = [[MathProgNLPModel(p(99), name=string(p)) for p in probs];
          [eval(Symbol("mgh0$p"))() for p in 1:3]]
stats = bmark_solvers(solvers, models, skipif=m -> m.meta.ncon > 0)
@test size(stats[:trunk], 1) == length(models) - 1
stats = bmark_solvers(solvers, models, skipif=m -> m.meta.ncon > 0, prune=false)
@test size(stats[:trunk], 1) == length(models)
@test count(stats[:trunk].extrainfo .== "skipped") == 1

# test bmark_solvers with CUTEst
@static if Sys.isunix()
  using CUTEst
  models = (isa(p, String) ? CUTEstModel(p) : CUTEstModel(p...) for p in ["ROSENBR", ("DIXMAANJ", "-param", "M=30")])
  stats = bmark_solvers(solvers, models, logger=ConsoleLogger())
  for s in keys(solvers)
    @info "Example of Dataframe output: Solver $s"
    @info stats[s][[:name, :status, :objective, :elapsed_time, :neval_obj]]
  end
end

# Test TRON
include("solvers/tron.jl")

# Test ExecutionStats
include("test_stats.jl")
