function test_logging()
  nlps = [ADNLPModel(x -> sum(x.^k), ones(2k), name="Sum of power $k") for k = 2:4]
  push!(nlps, ADNLPModel(x -> dot(x, x), ones(2), c=x->[sum(x)-1], lcon=[0.0], ucon=[0.0], name="linquad"))

  @info "Testing logger"
  log_header([:col_float, :col_int, :col_symbol, :col_string], [Float64, Int, Symbol, String])
  log_row([1.0, 1, :one, "one"])

  with_logger(ConsoleLogger()) do
    @info "Testing dummy solver with logger"
    dummy_solver(nlps[1], max_eval=20)
    reset!.(nlps)

    @info "Testing simple logger on `solve_problems`"
    solve_problems(dummy_solver, nlps)
    reset!.(nlps)

    @info "Testing logger with specific columns on `solve_problems`"
    solve_problems(dummy_solver, nlps, colstats=[:name, :nvar, :elapsed_time, :objective, :dual_feas])
    reset!.(nlps)

    @info "Testing logger with hdr_override on `solve_problems`"
    hdr_override = Dict(:dual_feas => "‖∇L(x)‖", :primal_feas => "‖c(x)‖")
    solve_problems(dummy_solver, nlps, info_hdr_override=hdr_override)
    reset!.(nlps)
  end
end

test_logging()
