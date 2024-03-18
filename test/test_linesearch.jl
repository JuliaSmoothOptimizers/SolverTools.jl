@testset "Linesearch" begin
  @testset "LineModel" begin
    nlp = BROWNDEN()
    n = nlp.meta.nvar
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
    @test hess(lm, 0.0) ≈ dot(d, hess(nlp, x) * d)
    @test hess!(lm, 0.0, g) == dot(d, hprod!(nlp, x, d, g))

    @test obj(lm, 1.0) == obj(nlp, x + d)
    @test grad(lm, 1.0) == dot(grad(nlp, x + d), d)
    @test hess(lm, 1.0) ≈ dot(d, hess(nlp, x + d) * d)

    @test neval_obj(lm) == 4
    @test neval_grad(lm) == 7
    @test neval_hess(lm) == 3
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

  @testset "Armijo-Goldstein" begin
    nlp = ADNLPModel(x -> x[1]^2 + 4 * x[2]^2, ones(2))
    lm = LineModel(nlp, nlp.meta.x0, -ones(2))

    t, ft, nbk, nbG = armijo_goldstein(lm, obj(lm, 0.0), grad(lm, 0.0))
    @test t == 1
    @test ft == 0.0
    @test nbk == 0
    @test nbG == 0

    redirect!(lm, nlp.meta.x0, -ones(2) / 2)
    t, ft, nbk, nbG = armijo_goldstein(lm, obj(lm, 0.0), grad(lm, 0.0))
    @test t == 1
    @test ft == 1.25
    @test nbk == 0
    @test nbG == 0

    redirect!(lm, nlp.meta.x0, -2 * ones(2))
    t, ft, nbk, nbG = armijo_goldstein(lm, obj(lm, 0.0), grad(lm, 0.0))
    @test t < 1
    @test nbk > 0
    @test nbG == 0

    T=Float32

    nlp = ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2, zeros(T,2))
    lm = LineModel(nlp, nlp.meta.x0, T.([1.7; 3.2]))
    t, ft, nbk, nbG = armijo_goldstein(lm, obj(lm, T(0.0)), grad(lm, T(0.0)); t = T(1.), τ₀ = T(0.1), τ₁ = T(0.2))
    @test t < 1.
    @test nbk == 4
    @test nbG == 10

    t, ft, nbk, nbG = armijo_goldstein(lm, obj(lm, T(0.0)), grad(lm, T(0.0)); t = T(.001), τ₀ = T(0.1), τ₁ = T(0.2))
    @test t < 1.
    @test nbk == 2
    @test nbG == 10

  end

  if VERSION ≥ v"1.6"
    @testset "Don't allocate" begin
      nlp = BROWNDEN()
      n = nlp.meta.nvar
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

      # armijo-wolfe
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

      # armijo-goldstein
      (t, ht, nbk, nbG) = armijo_goldstein(lm, h₀, slope)
      al = @wrappedallocs armijo_goldstein(lm, h₀, slope)
      @test al == 0

      function armijo_goldstein_alloc(lm, h₀, slope, bk_max)
        @allocated armijo_goldstein(lm, h₀, slope, bk_max = bk_max)
      end

      for bk_max = 0:8
        (t, ht, nbk, nbG) = armijo_goldstein(lm, h₀, slope, bk_max = bk_max)
        al = armijo_goldstein_alloc(lm, h₀, slope, bk_max)
        @test al == 0
      end
    end
  end
end
