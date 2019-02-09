using Test, NLPModels, NLPModelsJuMP, OptimizationProblems, Optimize, LinearAlgebra,
      SparseArrays, Logging, NLSProblems, Printf

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
    stats = with_logger(NullLogger()) do
      solver(model)
    end
    @test stats.status == :first_order
    reset!(model)
  end
  finalize(model)
end
@info("Done")

@info("Logging of calling trunk directly")
nlp = ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2, [-1.2; 1.0], name="rosen")
with_logger(ConsoleLogger()) do
  trunk(nlp)
end
@info("Logging of calling solve_problems with trunk")
nlp2 = ADNLPModel(x -> x[1]^4 + x[2]^4, ones(2), name="sumquartic")
solve_problems(trunk, [nlp, nlp2])
@info("Logging of calling solve_problems with trunk and colstats and solver logger")
df = solve_problems(trunk, [nlp, nlp2], solver_logger=ConsoleLogger(), colstats=[:name, :objective])
@info("Done")
@info("Showing latex_tabular_results example")
latex_tabular_results(stdout, df, cols=[:name, :status, :objective, :neval_obj],
                      fmt_override = Dict{Symbol,Function}(:name=>x->@sprintf("\\textbf{%s}", x) |>
                                         safe_latex_AbstractString),
                      hdr_override = Dict(:neval_obj=>"\\#F")
                     )

@info("Done")

# test benchmark helpers, skip constrained problems (hs7 has constraints)
with_logger(NullLogger()) do
  trunk(simple_dixmaanj(), monotone=false)
end
probs = [dixmaane, dixmaanf, dixmaang, dixmaanh, dixmaani, dixmaanj, hs7]

models = [[MathProgNLPModel(p(99), name=string(p)) for p in probs];
          [eval(Symbol("mgh0$p"))() for p in 1:3]]
with_logger(NullLogger()) do
  stats = bmark_solvers(solvers, models, skipif=m -> m.meta.ncon > 0)
  @test size(stats[:trunk], 1) == length(models) - 1
  stats = bmark_solvers(solvers, models, skipif=m -> m.meta.ncon > 0, prune=false)
  @test size(stats[:trunk], 1) == length(models)
  @test count(stats[:trunk].extrainfo .== "skipped") == 1
end

# test bmark_solvers with CUTEst
@static if Sys.isunix()
  using CUTEst
  models = (isa(p, String) ? CUTEstModel(p) : CUTEstModel(p...) for p in ["ROSENBR", ("DIXMAANJ", "-param", "M=30")])
  stats = with_logger(ConsoleLogger()) do
    bmark_solvers(solvers, models)
  end
  for s in keys(solvers)
    @info "Example of Dataframe output: Solver $s"
    @info stats[s][[:name, :status, :objective, :elapsed_time, :neval_obj]]
  end
end

# Test Trunk
include("solvers/trunkls.jl")

# Test TRON
include("solvers/tron.jl")

# Test ExecutionStats
include("test_stats.jl")
