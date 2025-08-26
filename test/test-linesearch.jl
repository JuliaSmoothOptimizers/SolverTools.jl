function test_line_model(::Type{S}) where {S}
  nlp = BROWNDEN(S)
  n = nlp.meta.nvar
  x = nlp.meta.x0
  d = fill!(S(undef, n), -1)
  lm = LineModel(nlp, x, d)
  g = fill!(S(undef, n), 0)

  T = eltype(S)
  @test obj(lm, zero(T)) == obj(nlp, x)
  @test grad(lm, zero(T)) == dot(grad(nlp, x), d)
  @test grad!(lm, zero(T), g) == dot(grad(nlp, x), d)
  @test g == grad(nlp, x)
  @test derivative(lm, zero(T)) == dot(grad(nlp, x), d)
  @test derivative!(lm, zero(T), g) == dot(grad(nlp, x), d)
  @test g == grad(nlp, x)
  @test objgrad!(lm, one(T), g) == (obj(nlp, x + d), dot(grad(nlp, x + d), d))
  @test g == grad(nlp, x + d)
  @test objgrad(lm, zero(T)) == (obj(nlp, x), dot(grad(nlp, x), d))
  @test hess(lm, zero(T)) ≈ dot(d, hess(nlp, x) * d)
  @test hess!(lm, zero(T), g) == dot(d, hprod!(nlp, x, d, g))

  @test obj(lm, one(T)) == obj(nlp, x + d)
  @test grad(lm, one(T)) == dot(grad(nlp, x + d), d)
  @test hess(lm, one(T)) ≈ dot(d, hess(nlp, x + d) * d)

  @test neval_obj(lm) == 4
  @test neval_grad(lm) == 7
  @test neval_hess(lm) == 3
end

function test_armijo_wolfe(::Type{S}) where {S}
  T = eltype(S)
  x0 = fill!(S(undef, 2), 1)
  nlp = ADNLPModel(x -> x[1]^2 + 4 * x[2]^2, x0, matrix_free = true)
  d = fill!(S(undef, 2), -1)
  lm = LineModel(nlp, nlp.meta.x0, d)
  g = fill!(S(undef, 2), 0)

  t0 = zero(T)
  t, good_grad, ft, nbk, nbW = armijo_wolfe(lm, obj(lm, t0), grad(lm, t0), g)
  @test t == 1
  @test ft == 0
  @test nbk == 0
  @test nbW == 0

  redirect!(lm, nlp.meta.x0, fill!(S(undef, 2), -1 // 2))
  t, good_grad, ft, nbk, nbW = armijo_wolfe(lm, obj(lm, t0), grad(lm, t0), g)
  @test t == 1
  @test ft == 1.25
  @test nbk == 0
  @test nbW == 0

  redirect!(lm, nlp.meta.x0, fill!(S(undef, 2), -2))
  t, good_grad, ft, nbk, nbW = armijo_wolfe(lm, obj(lm, t0), grad(lm, t0), g)
  @test t < 1
  @test nbk > 0
  @test nbW == 0

  nlp =
    ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2, fill!(S(undef, 2), 0), matrix_free = true)
  d = S([1.7; 3.2])
  lm = LineModel(nlp, nlp.meta.x0, d)
  t, good_grad, ft, nbk, nbW = armijo_wolfe(lm, obj(lm, t0), grad(lm, t0), g)
  @test t < 1
  @test nbk > 0
  @test nbW > 0
end

function test_armijo_goldstein(::Type{S}) where {S}
  T = eltype(S)
  nlp = ADNLPModel(x -> x[1]^2 + 4 * x[2]^2, fill!(S(undef, 2), 1))
  lm = LineModel(nlp, nlp.meta.x0, fill!(S(undef, 2), -1))

  t0 = zero(T)
  t, ft, nbk, nbG = armijo_goldstein(lm, obj(lm, t0), grad(lm, t0))
  @test t == 1
  @test ft == zero(T)
  @test nbk == 0
  @test nbG == 0

  redirect!(lm, nlp.meta.x0, fill!(S(undef, 2), -1 // 2))
  t, ft, nbk, nbG = armijo_goldstein(lm, obj(lm, t0), grad(lm, t0))
  @test t == 1
  @test ft == 1.25
  @test nbk == 0
  @test nbG == 0

  redirect!(lm, nlp.meta.x0, fill!(S(undef, 2), -2))
  t, ft, nbk, nbG = armijo_goldstein(lm, obj(lm, t0), grad(lm, t0))
  @test t < 1
  @test nbk > 0
  @test nbG == 0
end

function test_armijo_goldstein2(::Type{S}) where {S}
  T = eltype(S)
  nlp = ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2, fill!(S(undef, 2), 0))
  lm = LineModel(nlp, nlp.meta.x0, S([1.7; 3.2]))

  t0 = zero(T)
  t, ft, nbk, nbG =
    armijo_goldstein(lm, obj(lm, t0), grad(lm, t0); t = T(1), τ₀ = T(0.1), τ₁ = T(0.2))
  @test t < one(T)
  @test nbk == 4
  @test nbG == 10

  t, ft, nbk, nbG =
    armijo_goldstein(lm, obj(lm, t0), grad(lm, t0); t = T(0.001), τ₀ = T(0.1), τ₁ = T(0.2))
  @test t < 1.0
  @test nbk == 2
  @test nbG == 10
end

@testset "Linesearch" begin
  @testset "LineModel" begin
    test_line_model(Vector{Float64})
  end

  if CUDA.functional()
    @testset "LineModel with CuArray" begin
      CUDA.allowscalar() do
        test_line_model(CuVector{Float64})
      end
    end
  end

  @testset "Armijo-Wolfe" begin
    test_armijo_wolfe(Vector{Float64})
  end

  if CUDA.functional()
    @testset "Armijo-Wolfe with CuArray" begin
      CUDA.allowscalar() do
        test_armijo_wolfe(CuVector{Float64})
      end
    end
  end

  @testset "Armijo-Goldstein" begin
    @testset "Armijo-Goldstein Float64" begin
      test_armijo_goldstein(Vector{Float64})
    end

    @testset "Armijo-Goldstein Float32" begin
      test_armijo_goldstein2(Vector{Float32})
    end

    if CUDA.functional()
      @testset "Armijo-Goldstein with CuArray" begin
        CUDA.allowscalar() do
          test_armijo_goldstein(CuVector{Float64})
        end
      end
    end
  end

  if VERSION ≥ v"1.6"
    @testset "Don't allocate" begin
      S = Vector{Float64}
      T = eltype(S)
      nlp = BROWNDEN(S)
      n = nlp.meta.nvar
      x = nlp.meta.x0
      g = fill!(S(undef, n), 0)
      d = fill!(S(undef, n), -40)
      lm = LineModel(nlp, x, d)

      al = @wrappedallocs obj(lm, one(T))
      @test al == 0

      al = @wrappedallocs grad!(lm, one(T), g)
      @test al == 0

      al = @wrappedallocs objgrad!(lm, one(T), g)
      @test al == 0

      al = @wrappedallocs hess!(lm, one(T), g)
      @test al == 0

      h₀ = obj(lm, zero(T))
      slope = grad(lm, zero(T))

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
