using Optimize
using NLPModels
using AmplNLReader
using OptimizationProblems

models = [AmplModel("dixmaanj.nl"), JuMPNLPModel(dixmaanj(), name="dixmaanj")]
@unix_only begin
  using CUTEst
  push!(models, CUTEstModel("DIXMAANJ", "-param", "M=30"))
end
solvers = [:trunk, :lbfgs]

for model in models
  for solver in solvers
    stats = run_solver(solver, model, verbose=false)
    assert(all([stats...] .>= 0))
    reset!(model)
  end
end

# clean up the test directory
@unix_only begin
  here = dirname(@__FILE__)
  so_files = filter(x -> (ismatch(r".so$", x) || ismatch(r".dylib$", x)), readdir(here))

  for so_file in so_files
    rm(joinpath(here, so_file))
  end

  rm(joinpath(here, "AUTOMAT.d"))
  rm(joinpath(here, "OUTSDIF.d"))
end

# test benchmark helpers, skip constrained problems (hs7 has constraints)
run_ampl_problem(:trunk, :dixmaanj, 0, verbose=true, monotone=false)
probs = [:dixmaane, :dixmaanf, :dixmaang, :dixmaanh, :dixmaani, :dixmaanj, :hs7]
bmark_and_profile(solvers, probs, 99, bmark_args=Dict{Symbol, Any}(:skipif => m -> m.meta.ncon > 0))
