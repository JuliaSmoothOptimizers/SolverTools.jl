export display_header, run_problems, run_jump_problem, run_ampl_problem, run_solver

type SkipException <: Exception
end

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
      k = k + 1
    catch e
      isa(e, SkipException) || rethrow(e)
    end
  end
  return stats
end

"Apply a solver to a problem as a `JuMPNLPModel`."
function run_jump_problem(solver :: Symbol, problem :: Symbol, dim :: Int; verbose :: Bool=true, args...)
  problem_f = eval(problem)
  nlp = JuMPNLPModel(problem_f(dim), name=string(problem))
  # scale_obj!(nlp)  # not implemented
  stats = run_solver(solver, nlp, verbose=verbose; args...)
  # unscale_obj!(nlp)  # not implemented
  return stats
end

"Apply a solver to a problem as an `AmplModel`."
function run_ampl_problem(solver :: Symbol, problem :: Symbol, dim :: Int; verbose :: Bool=true, args...)
  problem_s = string(problem)
  nlp = AmplModel("$problem_s.nl")
  # Objective scaling not yet available.
  stats = run_solver(solver, nlp, verbose=verbose; args...)
  amplmodel_finalize(nlp)
  return stats
end

"Apply a solver to a generic `AbstractNLPModel`."
function run_solver(solver :: Symbol, nlp :: AbstractNLPModel; verbose :: Bool=true, args...)
  solver_f = eval(solver)
  args = Dict(args)
  skip = haskey(args, :skipif) ? pop!(args, :skipif) : x -> false
  skip(nlp) && throw(SkipException())

  # Julia nonsense
  optimal = false
  f = 0.0
  gNorm = 0.0
  status = "fail"
 @printf("%-15s", nlp.meta.name)
  try
    (x, f, gNorm, iter, optimal, tired, status) = solver_f(nlp, verbose=verbose; args...)
  catch e
      try
          status = e.msg
      catch
          status = "unknown failure"
      end
  end
  # if nlp.scale_obj
  #   f /= nlp.scale_obj_factor
  #   gNorm /= nlp.scale_obj_factor
  # end
  @printf(" %8d  %9.2e  %7.1e  %5d  %5d  %6d  %s\n",
          nlp.meta.nvar, f, gNorm,
          nlp.counters.neval_obj, nlp.counters.neval_grad,
          nlp.counters.neval_hprod, status)
  return optimal ? (nlp.counters.neval_obj, nlp.counters.neval_grad, nlp.counters.neval_hprod) : (-nlp.counters.neval_obj, -nlp.counters.neval_grad, -nlp.counters.neval_hprod)
end
