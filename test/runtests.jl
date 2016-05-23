using Optimize
using NLPModels
using OptimizationProblems

nlp_jump = JuMPNLPModel(dixmaanj())

x0 = [1.0; 2.0]
f(x) = dot(x,x)/2
g(x) = x
H(x) = eye(2)

nlp_simple = SimpleNLPModel(x0, f, grad=g, hess=H)

for nlp in [nlp_jump; nlp_simple]
  for solver in [:trunk, :lbfgs]
    stats = run_solver(solver, nlp, verbose=true)
  end
end
