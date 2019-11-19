using DataFrames

mutable struct CallableSolver
end

function (solver :: CallableSolver)(nlp :: AbstractNLPModel; kwargs...)
  return GenericExecutionStats(:unknown, nlp)
end

function test_bmark()
  @testset "Testing bmark" begin
    problems = [ADNLPModel(x -> sum(x.^k), ones(2k), name="Sum of power $k") for k = 2:4]
    callable = CallableSolver()
    stats = solve_problems(dummy_solver, problems)
    @test stats isa DataFrame
    stats = solve_problems(dummy_solver, problems, reset_problem=false)
    stats = solve_problems(dummy_solver, problems, reset_problem=true)

    solve_problems(callable, problems)

    solvers = Dict(:dummy => dummy_solver, :callable => callable)
    stats = bmark_solvers(solvers, problems)
    @test stats isa Dict{Symbol, DataFrame}
    for k in keys(solvers)
      @test haskey(stats, k)
    end
  end
end

test_bmark()
