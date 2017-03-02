using OptimizationProblems

@testset "Simple tests" begin
  @testset "Quadratic with linear constraints" begin
    for D in [ ones(2), linspace(1e-2, 100, 2), linspace(1e-2, 1e2, 10), linspace(1e-4, 1e4, 10) ]
      n = length(D)
      for x0 in Any[ zeros(n), ones(n), -collect(linspace(1, n, n)) ]
        nlp = ADNLPModel(x->dot(x,D.*x), x0,
                         c=x->[sum(x)-1], lcon=[0], ucon=[0])

        λ = -1/sum(1./D)
        stats = auglag(nlp, verbose=false, rtol=0.0)
        x, fx, gpx, cx = stats.solution, stats.obj, stats.opt_norm, stats.feas_norm
        @test isapprox(x, -λ./D, atol=1e-4)
        @test isapprox(fx, -λ, atol=1e-5)
        @test isapprox(gpx, 0.0, atol=1e-4)
        @test isapprox(cx, 0.0, atol=1e-4)
      end
    end
  end

  @testset "HS6" begin
    nlp = MathProgNLPModel(hs6())

    stats = auglag(nlp, verbose=false, rtol=0.0)
    x, fx, gpx, cx = stats.solution, stats.obj, stats.opt_norm, stats.feas_norm
    @test isapprox(x, ones(2), atol=1e-4)
    @test isapprox(fx, 0.0, atol=1e-5)
    @test isapprox(gpx, 0.0, atol=1e-4)
    @test isapprox(cx, 0.0, atol=1e-4)
  end

  @testset "HS7" begin
    nlp = MathProgNLPModel(hs7())

    stats = auglag(nlp, verbose=false, rtol=0.0)
    x, fx, gpx, cx = stats.solution, stats.obj, stats.opt_norm, stats.feas_norm
    @test isapprox(x, [0.0; sqrt(3)], atol=1e-4)
    @test isapprox(fx, -sqrt(3), atol=1e-5)
    @test isapprox(gpx, 0.0, atol=1e-4)
    @test isapprox(cx, 0.0, atol=1e-4)
  end

  @testset "HS8" begin
    nlp = MathProgNLPModel(hs8())

    stats = auglag(nlp, verbose=false, rtol=0.0)
    x, fx, gpx, cx = stats.solution, stats.obj, stats.opt_norm, stats.feas_norm
    sol = sqrt( (25 + sqrt(301)*[1;-1])/2 )
    @test isapprox(abs(x), sol, atol=1e-4)
    @test isapprox(fx, -1.0, atol=1e-5)
    @test isapprox(gpx, 0.0, atol=1e-4)
    @test isapprox(cx, 0.0, atol=1e-4)
  end

  @testset "HS9" begin
    nlp = MathProgNLPModel(hs9())

    stats = auglag(nlp, verbose=false, rtol=0.0)
    x, fx, gpx, cx = stats.solution, stats.obj, stats.opt_norm, stats.feas_norm
    @test isapprox((x + [3;4]) .% [12; 16], zeros(2), atol=1e-4)
    @test isapprox(fx, -0.5, atol=1e-5)
    @test isapprox(gpx, 0.0, atol=1e-4)
    @test isapprox(cx, 0.0, atol=1e-4)
  end

  @testset "HS26" begin
    nlp = MathProgNLPModel(hs26())

    stats = auglag(nlp, verbose=false, rtol=0.0)
    x, fx, gpx, cx = stats.solution, stats.obj, stats.opt_norm, stats.feas_norm
    @test isapprox(fx, 0.0, atol=1e-5)
    @test isapprox(gpx, 0.0, atol=1e-4)
    @test isapprox(cx, 0.0, atol=1e-4)
  end

  @testset "HS27" begin
    nlp = MathProgNLPModel(hs27())

    stats = auglag(nlp, verbose=false, rtol=0.0)
    x, fx, gpx, cx = stats.solution, stats.obj, stats.opt_norm, stats.feas_norm
    @test isapprox(x, [-1.0; 1.0; 0.0], atol=1e-4)
    @test isapprox(fx, 0.04, atol=1e-5)
    @test isapprox(gpx, 0.0, atol=1e-4)
    @test isapprox(cx, 0.0, atol=1e-4)
  end

end

@testset "Troublesome problems" begin
  @testset "No Lagrangian multiplier" begin
    nlp = ADNLPModel(x->x[1], [1.0], c=x->[x[1]^2], lcon=[0.0], ucon=[0.0])

    stats = auglag(nlp, verbose=false, rtol=0.0)
    x, fx, gpx, cx = stats.solution, stats.obj, stats.opt_norm, stats.feas_norm
    @test isapprox(x, [0.0], atol=1e-4)
    @test isapprox(fx, 0.0, atol=1e-4)
    @test isapprox(gpx, 0.0, atol=1e-4)
    @test isapprox(cx, 0.0, atol=1e-4)
  end
end

@testset "Larger problems" begin
  @testset "Quadratic with linear constraints" begin
    for n = 10.^(2:4)
      for D in [ ones(n), linspace(1e-2, 100, n)]
        for x0 in Any[ zeros(n), ones(n), -collect(linspace(1, n, n)) ]
          nlp = SimpleNLPModel(x->dot(x,D.*x)/2, x0,
                               g = x->D.*x,
                               Hp! = (x,v,Hv;y=[],obj_weight=1.0)->Hv .= obj_weight*D.*v,
                               c=x->[sum(x)-1], lcon=[0], ucon=[0],
                               Jp = (x,v)->[sum(v)],
                               Jtp = (x,v)->ones(n)*v[1])

          λ = -1/sum(1./D)
          stats = auglag(nlp, verbose=false, rtol=0.0)
          x, fx, gpx, cx = stats.solution, stats.obj, stats.opt_norm, stats.feas_norm
          @test isapprox(x, -λ./D, atol=1e-4)
          @test isapprox(fx, -λ/2, atol=1e-5)
          @test isapprox(gpx, 0.0, atol=1e-4)
          @test isapprox(cx, 0.0, atol=1e-4)
        end
      end
    end
  end
end
