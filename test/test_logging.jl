function test_logging()
  nlps = [ADNLPModel(x -> sum(x.^k), ones(2k), name="Sum of power $k") for k = 2:4]
  push!(nlps, ADNLPModel(x -> dot(x, x), ones(2), x->[sum(x)-1], [0.0], [0.0], name="linquad"))

  @info "Testing logger"
  log_header([:col_float, :col_int, :col_symbol, :col_string], [Float64, Int, Symbol, String])
  log_row([1.0, 1, :one, "one"])

  with_logger(ConsoleLogger()) do
    @info "Testing dummy solver with logger"
    dummy_solver(nlps[1], max_eval=20)
    reset!.(nlps)
  end
end

test_logging()
