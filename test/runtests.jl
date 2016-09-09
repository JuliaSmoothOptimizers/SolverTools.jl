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

path = dirname(@__FILE__)
list_so_files = filter(x->contains(x,".so"), readdir(path))

for so_file=list_so_files
    run(`rm $path/$so_file `)
end

run(`rm $path/AUTOMAT.d`)
run(`rm $path/OUTSDIF.d`)
