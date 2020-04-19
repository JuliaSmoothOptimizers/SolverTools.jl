using DataFrames

mutable struct CallableSolver
end

function (solver :: CallableSolver)(nlp :: AbstractNLPModel; kwargs...)
  return GenericExecutionStats(:unknown, nlp)
end

function test_bmark()
  @testset "Testing bmark" begin
    problems = [ADNLPModel(x -> sum(x.^2), ones(2), name="Quadratic"),
                ADNLPModel(x -> sum(x.^2), ones(2), c=x->[sum(x)-1], lcon=[0.0], ucon=[0.0], name="Cons quadratic"),
                ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2, ones(2), c=x->[x[1]^2 + x[2]^2 - 1], lcon=[0.0], ucon=[0.0], name="Cons Rosen")]
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

    # write stats to file
    filename = tempname()
    save_stats(stats, filename)

    # read stats from file
    stats2 = load_stats(filename)

    # check that they are the same
    for k ∈ keys(stats)
      @test k ∈ keys(stats2)
      @test stats[k] == stats2[k]
    end

    statuses, avgs = quick_summary(stats)
  end
end

test_bmark()
