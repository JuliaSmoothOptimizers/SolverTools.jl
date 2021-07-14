@testset "Linesearch" begin
  @testset "LineModel" begin
    n = 200
    nlp = SimpleModel(n)
    x = nlp.meta.x0
    d = -ones(n)
    lm = LineModel(nlp, x, d)
    g = zeros(n)

    @test obj(lm, 0.0) == obj(nlp, x)
    @test grad(lm, 0.0) == dot(grad(nlp, x), d)
    @test grad!(lm, 0.0, g) == dot(grad(nlp, x), d)
    @test g == grad(nlp, x)
    @test derivative(lm, 0.0) == dot(grad(nlp, x), d)
    @test derivative!(lm, 0.0, g) == dot(grad(nlp, x), d)
    @test g == grad(nlp, x)
    @test objgrad!(lm, 1.0, g) == (obj(nlp, x + d), dot(grad(nlp, x + d), d))
    @test g == grad(nlp, x + d)
    @test objgrad(lm, 0.0) == (obj(nlp, x), dot(grad(nlp, x), d))
    @test hess(lm, 0.0) == dot(d, Symmetric(hess(nlp, x), :L) * d)
    @test hess!(lm, 0.0, g) == dot(d, hprod!(nlp, x, d, g))

    @test obj(lm, 1.0) == 0.0
    @test grad(lm, 1.0) == 0.0
    @test hess(lm, 1.0) == 0.0

    @test obj(lm, 0.0) ≈ n / 12
    @test grad(lm, 0.0) ≈ -n / 3
    @test hess(lm, 0.0) == n

    @test neval_obj(lm) == 5
    @test neval_grad(lm) == 8
    @test neval_hess(lm) == 4
  end

  @testset "Armijo-Wolfe" begin
    nlp = ADNLPModel(x -> x[1]^2 + 4 * x[2]^2, ones(2))
    lm = LineModel(nlp, nlp.meta.x0, -ones(2))
    g = zeros(2)

    t, good_grad, ft, nbk, nbW = armijo_wolfe(lm, obj(lm, 0.0), grad(lm, 0.0), g)
    @test t == 1
    @test ft == 0.0
    @test nbk == 0
    @test nbW == 0

    redirect!(lm, nlp.meta.x0, -ones(2) / 2)
    t, good_grad, ft, nbk, nbW = armijo_wolfe(lm, obj(lm, 0.0), grad(lm, 0.0), g)
    @test t == 1
    @test ft == 1.25
    @test nbk == 0
    @test nbW == 0

    redirect!(lm, nlp.meta.x0, -2 * ones(2))
    t, good_grad, ft, nbk, nbW = armijo_wolfe(lm, obj(lm, 0.0), grad(lm, 0.0), g)
    @test t < 1
    @test nbk > 0
    @test nbW == 0

    nlp = ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2, zeros(2))
    lm = LineModel(nlp, nlp.meta.x0, [1.7; 3.2])
    t, good_grad, ft, nbk, nbW = armijo_wolfe(lm, obj(lm, 0.0), grad(lm, 0.0), g)
    @test t < 1
    @test nbk > 0
    @test nbW > 0
  end

  if VERSION ≥ v"1.6"
    @testset "Don't allocate" begin
      n = 200
      nlp = SimpleModel(n)
      x = nlp.meta.x0
      g = zeros(n)
      d = -40 * ones(n)
      lm = LineModel(nlp, x, d)

      al = @wrappedallocs obj(lm, 1.0)
      @test al == 0

      al = @wrappedallocs grad!(lm, 1.0, g)
      @test al == 0

      al = @wrappedallocs objgrad!(lm, 1.0, g)
      @test al == 0

      al = @wrappedallocs hess!(lm, 1.0, g)
      @test al == 0

      h₀ = obj(lm, 0.0)
      slope = grad(lm, 0.0)
      (t, gg, ht, nbk, nbW) = armijo_wolfe(lm, h₀, slope, g)
      al = @wrappedallocs armijo_wolfe(lm, h₀, slope, g)
      @test al == 0

      function armijo_wolfe_alloc(lm, h₀, slope, g, bk_max)
        @allocated armijo_wolfe(lm, h₀, slope, g, bk_max = bk_max)
      end

      for bk_max = 0:8
        (t, gg, ht, nbk, nbW) = armijo_wolfe(lm, h₀, slope, g, bk_max = bk_max)
        al = armijo_wolfe_alloc(lm, h₀, slope, g, bk_max)
        @test al == 0
      end
    end
  end
end
