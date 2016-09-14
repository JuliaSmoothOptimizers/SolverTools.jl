using Optimize
using NLPModels
using AmplNLReader
using OptimizationProblems

models = [AmplModel("dixmaanj.nl"), JuMPNLPModel(dixmaanj())]
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

# clean up the test directory
here = dirname(@__FILE__)
so_files = filter(x -> (ismatch(r".so$", x) || ismatch(r".dylib$", x)), readdir(here))

for so_file in so_files
	rm(joinpath(here, so_file))
end

rm(joinpath(here, "AUTOMAT.d"))
rm(joinpath(here, "OUTSDIF.d"))

