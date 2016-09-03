export display_header, run_problems, run_jump_problem, run_ampl_problem, run_solver


function display_header()
  @printf("%-15s  %8s  %9s  %7s  %5s  %5s  %6s  %s\n",
          "Name", "nvar", "f", "‖∇f‖", "#obj", "#grad", "#hprod", "status")
end

"Apply a solver to a set of problems."
function run_problems(solver :: Symbol, problems :: Vector{Symbol}, dim :: Int; format :: Symbol=:jump, args...)
  format in (:jump, :ampl) || error("format not recognized---use :jump or :ampl")
  run_problem = eval(symbol("run_" * string(format) * "_problem"))
  display_header()
  nprobs = length(problems)
  verbose = nprobs ≤ 1
  stats = Array(Int, nprobs, 3)
  k = 1
  for problem in problems
    try
      (f, g, h) = run_problem(solver, problem, dim, verbose=verbose; args...)
      stats[k, :] = [f, g, h]
    catch e
      stats[k, :] = [-1, -1, -1]
    end
    k = k + 1
  end
  return stats
end

"Apply a solver to a problem as a `JuMPNLPModel`."
function run_jump_problem(solver :: Symbol, problem :: Symbol, dim :: Int; verbose :: Bool=true, args...)
  problem_s = string(problem)
  problem_f = eval(problem)
  nlp = JuMPNLPModel(problem_f(dim))
  nlp.meta.ncon == 0 || error("problem has constraints")
  # scale_obj!(nlp)  # not implemented
  stats = run_solver(solver, nlp, problem_s, verbose=verbose; args...)
  # unscale_obj!(nlp)  # not implemented
  return stats
end

"Apply a solver to a problem as an `AmplModel`."
function run_ampl_problem(solver :: Symbol, problem :: Symbol, dim :: Int; verbose :: Bool=true, args...)
  problem_s = string(problem)
  nlp = AmplModel("$problem_s.nl")
  nlp.meta.ncon == 0 || error("problem has constraints")
  # Objective scaling not yet available.
  run_solver(solver, nlp, problem_s, verbose=verbose; args...)
  amplmodel_finalize(nlp)
end

"Apply a solver to a generic `AbstractNLPModel`."
function run_solver(solver :: Symbol, nlp :: AbstractNLPModel; verbose :: Bool=true, args...)
  solver_f = eval(solver)
  (x, f, gNorm, iter, optimal, tired, status) = solver_f(nlp, verbose=verbose; args...)
  # if nlp.scale_obj
  #   f /= nlp.scale_obj_factor
  #   gNorm /= nlp.scale_obj_factor
  # end
  @printf("%-15s  %8d  %9.2e  %7.1e  %5d  %5d  %6d  %s\n",
          nlp.meta.name, nlp.meta.nvar, f, gNorm,
          nlp.counters.neval_obj, nlp.counters.neval_grad,
          nlp.counters.neval_hprod, status)
  return optimal ? (nlp.counters.neval_obj, nlp.counters.neval_grad, nlp.counters.neval_hprod) : (-nlp.counters.neval_obj, -nlp.counters.neval_grad, -nlp.counters.neval_hprod)
end
