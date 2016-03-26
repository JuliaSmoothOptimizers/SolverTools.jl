using Optimize

ampl_prob_dir = "../problems"
probs = filter(name -> name != :OptimizationProblems, names(OptimizationProblems))
ampl_probs = [split(p, ".")[1] for p in filter(x -> contains(x, ".nl"), readdir(ampl_prob_dir))]

for prob in probs ∩ [symbol(p) for p in ampl_probs]
  jump_vs_ampl(prob, ampl_prob_dir)
end
