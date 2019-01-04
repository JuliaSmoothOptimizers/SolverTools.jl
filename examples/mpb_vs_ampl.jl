using AmplNLReader, Optimize, NLPModels, NLPModelsJuMP, Printf, OptimizationProblems,
      LinearAlgebra

"""Compare the objective function, constraints and their first and second
derivatives of the `nlp_mpb` and `nlp_ampl` at a randomly-generated point.
The comparison is repeated `nloops` times. Values must agree in a relative
way to within `rtol`.
"""
function mpb_vs_ampl_helper(nlp_mpb :: AbstractNLPModel,
                             nlp_ampl :: AbstractNLPModel;
                             nloops :: Int=100, rtol :: Float64=1.0e-10)

  n = nlp_ampl.meta.nvar
  m = nlp_ampl.meta.ncon

  for k = 1 : nloops
    x = 10 * (rand(n) .- 0.5)

    f_mpb = obj(nlp_mpb, x)
    f_ampl = obj(nlp_ampl, x)
    err_f = abs(f_mpb - f_ampl) / max(abs(f_ampl), 1.0)
    err_f > rtol && @info @sprintf("\n|∆f/f| = %7.1e", err_f)

    g_mpb = grad(nlp_mpb, x)
    g_ampl = grad(nlp_ampl, x)
    err_g = norm(g_mpb - g_ampl) / max(norm(g_ampl), 1.0)
    err_g > rtol && @info @sprintf("\n|∆g/g| = %7.1e", err_g)

    H_mpb = hess(nlp_mpb, x)
    H_ampl = hess(nlp_ampl, x)
    err_H = norm(H_mpb - H_ampl) / max(norm(H_ampl), 1.0)
    err_H > rtol && @info @sprintf("\n|∆H/H| = %7.1e", err_H)

    v = 10 * (rand(n) .- 0.5)
    Hv_mpb = hprod(nlp_mpb, x, v)
    Hv_ampl = hprod(nlp_ampl, x, v)
    err_Hv = norm(Hv_mpb - Hv_ampl) / max(norm(Hv_ampl), 1.0)
    err_Hv > rtol && @info @sprintf("\n|∆Hv/Hv| = %7.1e", err_Hv)

    if m > 0
      c_mpb = cons(nlp_mpb, x)
      c_ampl = cons(nlp_ampl, x)
      # JuMP subtracts the lhs or rhs of one-sided
      # nonlinear inequality and nonlinear equality constraints
      nln_low = ∩(nlp_ampl.meta.nln, nlp_ampl.meta.jlow)
      c_ampl[nln_low] -= nlp_ampl.meta.lcon[nln_low]
      nln_upp = ∩(nlp_ampl.meta.nln, nlp_ampl.meta.jupp)
      c_ampl[nln_upp] -= nlp_ampl.meta.ucon[nln_upp]
      nln_fix = ∩(nlp_ampl.meta.nln, nlp_ampl.meta.jfix)
      c_ampl[nln_fix] -= nlp_ampl.meta.lcon[nln_fix]
      # JuMP orders linear constraints first
      lin = nlp_ampl.meta.lin
      lnet = nlp_ampl.meta.lnet
      nnet = nlp_ampl.meta.nnet
      nln = nlp_ampl.meta.nln
      p = [lin ; lnet ; nnet ; nln]
      c_ampl = c_ampl[p]
      err_c = norm(c_mpb - c_ampl) / max(norm(c_ampl), 1.0)
      err_c > rtol && @info @sprintf("\n|∆c/c| = %7.1e", err_c)

      J_mpb = jac(nlp_mpb, x)
      J_ampl = jac(nlp_ampl, x)
      J_ampl = J_ampl[p, :]
      err_J = norm(J_mpb - J_ampl) / max(norm(J_ampl), 1.0)
      err_J > rtol && @info @sprintf("\n|∆J/J| = %7.1e", err_J)

      y = 10 * (rand(m) .- 0.5)

      # MPB sets the Lagrangian to f + Σᵢ yᵢ cᵢ
      # AmplNLReader sets it to    f - Σᵢ yᵢ cᵢ
      H_mpb = hess(nlp_mpb, x, -y)
      H_ampl = hess(nlp_ampl, x, y=y[p])
      err_H = norm(H_mpb - H_ampl) / max(norm(H_ampl), 1.0)
      err_H > rtol && @info @sprintf("\n|∆H/H| = %7.1e", err_H)

      Hv_mpb = hprod(nlp_mpb, x, -y, v)
      Hv_ampl = hprod(nlp_ampl, x, v, y=y[p])
      err_Hv = norm(Hv_mpb - Hv_ampl) / max(norm(Hv_ampl), 1.0)
      err_Hv > rtol && @info @sprintf("\n|∆Hv/Hv| = %7.1e", err_Hv)
    end
  end
end

"Compare the MathProgNLPModel and AmplModel of the given `problem`."
function mpb_vs_ampl(problem :: Symbol, ampl_dir :: String; nloops=100, rtol=1.0e-10)
  problem_s = string(problem)
  @info @sprintf("Checking problem %-15s\t", problem_s)
  problem_f = eval(problem)
  nlp_mpb = MathProgNLPModel(problem_f())
  nlp_ampl = AmplModel("$ampl_dir/$problem_s.nl")
  mpb_vs_ampl_helper(nlp_mpb, nlp_ampl, nloops=nloops, rtol=rtol)
  @info @sprintf("✓\n")
  amplmodel_finalize(nlp_ampl)
end

ampl_prob_dir = "ampl"
probs = filter(name -> name != :OptimizationProblems, names(OptimizationProblems))
ampl_probs = [split(p, ".")[1] for p in filter(x -> occursin(".nl", x), readdir(ampl_prob_dir))]

for prob in probs ∩ [Symbol(p) for p in ampl_probs]
  mpb_vs_ampl(prob, ampl_prob_dir)
end
