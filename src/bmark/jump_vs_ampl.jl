export jump_vs_ampl

"""Compare the objective function, constraints and their first and second
derivatives of the `nlp_jump` and `nlp_ampl` at a randomly-generated point.
The comparison is repeated `nloops` times. Values must agree in a relative
way to within `rtol`.
"""
function jump_vs_ampl_helper(nlp_jump :: AbstractNLPModel,
                             nlp_ampl :: AbstractNLPModel;
                             nloops :: Int=100, rtol :: Float64=1.0e-10)

  n = nlp_ampl.meta.nvar
  m = nlp_ampl.meta.ncon

  for k = 1 : nloops
    x = 10 * (rand(n) - 0.5)

    f_jump = obj(nlp_jump, x)
    f_ampl = obj(nlp_ampl, x)
    err_f = abs(f_jump - f_ampl) / max(abs(f_ampl), 1.0)
    err_f > rtol && println(err_f)  # @printf("\n|∆f/f| = %7.1e", err_f)

    g_jump = grad(nlp_jump, x)
    g_ampl = grad(nlp_ampl, x)
    err_g = norm(g_jump - g_ampl) / max(norm(g_ampl), 1.0)
    err_g > rtol && println(err_g)  # @printf("\n|∆g/g| = %7.1e")

    H_jump = hess(nlp_jump, x)
    H_ampl = hess(nlp_ampl, x)
    err_H = vecnorm(H_jump - H_ampl) / max(vecnorm(H_ampl), 1.0)
    err_H > rtol && println(err_H)  # @printf("\n|∆H/H| = %7.1e", err_H)

    v = 10 * (rand(n) - 0.5)
    Hv_jump = hprod(nlp_jump, x, v)
    Hv_ampl = hprod(nlp_ampl, x, v)
    err_Hv = norm(Hv_jump - Hv_ampl) / max(norm(Hv_ampl), 1.0)
    err_Hv > rtol && println(err_Hv)  # @printf("\n|∆Hv/Hv| = %7.1e", err_Hv)

    if m > 0
      c_jump = cons(nlp_jump, x)
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
      err_c = norm(c_jump - c_ampl) / max(norm(c_ampl), 1.0)
      err_c > rtol && println(err_c)  # @printf("\n|∆c/c| = %7.1e", err_c)

      J_jump = jac(nlp_jump, x)
      J_ampl = jac(nlp_ampl, x)
      J_ampl = J_ampl[p, :]
      err_J = vecnorm(J_jump - J_ampl) / max(vecnorm(J_ampl), 1.0)
      err_J > rtol && println(err_J)  # @printf("\n|∆J/J| = %7.1e", err_J)

      y = 10 * (rand(m) - 0.5)

      # MPB sets the Lagrangian to f + Σᵢ yᵢ cᵢ
      # AmplNLReader sets it to    f - Σᵢ yᵢ cᵢ
      H_jump = hess(nlp_jump, x, -y)
      H_ampl = hess(nlp_ampl, x, y=y[p])
      err_H = vecnorm(H_jump - H_ampl) / max(vecnorm(H_ampl), 1.0)
      err_H > rtol && println(err_H)  # @printf("\n|∆H/H| = %7.1e", err_H)

      Hv_jump = hprod(nlp_jump, x, -y, v)
      Hv_ampl = hprod(nlp_ampl, x, v, y=y[p])
      err_Hv = norm(Hv_jump - Hv_ampl) / max(norm(Hv_ampl), 1.0)
      err_Hv > rtol && println(err_Hv)  # @printf("\n|∆Hv/Hv| = %7.1e", err_Hv)
    end
  end
end

"Compare the JuMPNLPModel and AmplModel of the given `problem`."
function jump_vs_ampl(problem :: Symbol, ampl_dir :: AbstractString; nloops=100, rtol=1.0e-10)
  problem_s = string(problem)
  @printf("Checking problem %-15s\t", problem_s)
  problem_f = eval(problem)
  nlp_jump = JuMPNLPModel(problem_f())
  nlp_ampl = AmplModel("$ampl_dir/$problem_s.nl")
  jump_vs_ampl_helper(nlp_jump, nlp_ampl, nloops=nloops, rtol=rtol)
  @printf("✓\n")
  amplmodel_finalize(nlp_ampl)
end
