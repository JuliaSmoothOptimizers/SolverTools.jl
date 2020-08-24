@testset "Linesearch" begin
  ls_methods = [:armijo, :armijo_wolfe]
  @testset "Unconstrained problems" begin
    specific_tests = Dict(
      :armijo => Dict(:nbk => [0, 0, 1, 0]),
      :armijo_wolfe => Dict(:nbk => [0, 0, 1, 5], :nbW => [0, 0, 0, 1]),
    )
    for ls_method in ls_methods
      @testset "Method $ls_method" begin
        nlp = ADNLPModel(x -> x[1]^2 + 4 * x[2]^2, ones(2))
        x = nlp.meta.x0
        d = -ones(2)
        lso = linesearch(nlp, x, d, method=ls_method)
        @test lso.t == 1
        @test lso.ϕt == 0.0
        for (k,v) in specific_tests[ls_method]
          @test lso.specific[k] == v[1]
        end

        d = -ones(2) / 2
        lso = linesearch(nlp, x, d, method=ls_method)
        @test lso.t == 1
        @test lso.ϕt == 1.25
        for (k,v) in specific_tests[ls_method]
          @test lso.specific[k] == v[2]
        end

        d = -2 * ones(2)
        lso = linesearch(nlp, x, d, method=ls_method)
        @test lso.t < 1
        for (k,v) in specific_tests[ls_method]
          @test lso.specific[k] == v[3]
        end

        nlp = ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2, zeros(2))
        x = nlp.meta.x0
        d = [1.7; 3.2]
        lso = linesearch(nlp, x, d, method=ls_method)
        for (k,v) in specific_tests[ls_method]
          @test lso.specific[k] == v[4]
        end
      end # @testset
    end # for
  end # @testset

  @testset "Constrainted problems" begin
    specific_tests = Dict(
      :armijo => Dict(:nbk => [0, 0, 1]),
      :armijo_wolfe => Dict(:nbk => [0, 0, 1]),
    )
    for ls_method in ls_methods
      @testset "Method $ls_method" begin
        nlp = ADNLPModel(x -> x[1]^2 + 4 * x[2]^2, ones(2), x -> [x[1] + 2x[2]], [0.0], [0.0])
        x = nlp.meta.x0
        ϕ = L1Merit(nlp, 1.0)
        d = -ones(2)
        lso = linesearch(ϕ, x, d, method=ls_method, update_obj_at_x=true, update_derivative_at_x=true)
        @test lso.t == 1
        @test lso.ϕt == 0.0
        for (k,v) in specific_tests[ls_method]
          @test lso.specific[k] == v[1]
        end

        d = -ones(2) / 2
        lso = linesearch(ϕ, x, d, method=ls_method, update_obj_at_x=true, update_derivative_at_x=true)
        @test lso.t == 1
        @test lso.ϕt == 1.25 + 1.5
        for (k,v) in specific_tests[ls_method]
          @test lso.specific[k] == v[2]
        end

        d = -2 * ones(2)
        lso = linesearch(ϕ, x, d, method=ls_method, update_obj_at_x=true, update_derivative_at_x=true)
        @test lso.t < 1
        for (k,v) in specific_tests[ls_method]
          @test lso.specific[k] == v[3]
        end
      end # @testset
    end # for
  end # @testset
end