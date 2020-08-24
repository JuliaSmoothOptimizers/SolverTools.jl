function test_merit(merit_constructor, ϕ)
  @testset "Consistency" begin
    nlp = ADNLPModel(x -> dot(x, x), zeros(5), x -> [sum(x.^2) - 5; sum(x) - 1; prod(x)], zeros(3), zeros(3))

    for x in (zeros(5), ones(5), -ones(5), BigFloat.([π, -2π, 3π, -4π, 0.0])),
        d in (-ones(5), ones(5))
      η = one(eltype(x))
      merit = merit_constructor(nlp, η)
      ϕx = obj(merit, x)
      @test ϕx ≈ ϕ(nlp, x, η)
      @test obj(nlp, x) == merit.fx
      @test cons(nlp, x) == merit.cx

      Dϕx = derivative(merit, x, d)
      ϵ = √eps(eltype(η))
      Dϕxa = (ϕ(nlp, x + ϵ * d, η) - ϕ(nlp, x, η)) / ϵ
      @test isapprox(Dϕx, Dϕxa, rtol=1e-6)
      @test grad(nlp, x) == merit.gx
      @test jac(nlp, x) * d == merit.Ad

      ϕt = obj(merit, x + d)
      @test ϕt ≈ ϕ(nlp, x + d, η)
      @test obj(nlp, x + d) == merit.fx
      @test cons(nlp, x + d) == merit.cx
    end
  end

  @testset "Simple line search" begin
    nlp = ADNLPModel(x -> dot(x, x), [-1.0; 1.0], x -> [sum(x) - 1], [0.0], [0.0])

    sol = [0.5; 0.5]
    x = [-1.0; 1.0]
    d = 30 * (sol - x)
    η = 1.0

    merit = merit_constructor(nlp, η)

    @assert obj(nlp, x + d) > obj(nlp, x)
    @assert norm(cons(nlp, x + d)) > norm(cons(nlp, x))

    reset!(nlp)
    bk = 0
    ϕx = obj(merit, x)
    obj(merit, x, update=false)
    Dϕx = derivative(merit, x, d)
    derivative(merit, x, d, update=false)
    @test Dϕx < 0 # If d is not a descent direction for your merit, change your parameters
    t = 1.0
    ϕt = obj(merit, x + d)
    while ϕt > ϕx + 0.5 * t * Dϕx
      t /= 2
      ϕt = obj(merit, x + t * d)
      if t < 1e-16
        error("Failure")
      end
      bk += 1
    end
    @test t < 1
    @test neval_obj(nlp) == 2 + bk
    @test neval_cons(nlp) == 2 + bk
    @test neval_grad(nlp) == 1
    @test neval_jprod(nlp) == 1
  end

  @testset "Separate parts" begin
    nlp = ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2, [-1.2; 1.0], x -> [x[1]^2 + x[2]^2 - 1], [0.0], [0.0])
    ϕ = merit_constructor(nlp, 0.0)
    for i = 1:2
      x = randn(nlp.meta.nvar)
      d = randn(nlp.meta.nvar)
      ϕ.η = 0.0
      manual_dualobj = obj(ϕ, x)
      ϕ.η = 1.0
      manual_primalobj = obj(ϕ, x) - manual_dualobj
      @test dualobj(ϕ) ≈ manual_dualobj
      @test primalobj(ϕ) ≈ manual_primalobj
    end
  end

  @testset "Safe for unconstrained" begin
    nlp = ADNLPModel(x -> dot(x, x), ones(2))
    x = nlp.meta.x0
    d = ones(2)

    merit = merit_constructor(nlp, 1.0)
    @test obj(merit, x) == obj(nlp, x)
    @test derivative(merit, x, d) == dot(grad(nlp, x), d)
  end

  @testset "Use on solver" begin
    nlp = ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
    output = dummy_linesearch_solver(nlp, merit_constructor=merit_constructor, rtol=0)
    @test isapprox(output.solution, ones(2), rtol=1e-6)
    @test output.objective < 1e-6
    @test output.primal_feas == 0
    @test output.dual_feas < 1e-6

    nlp = ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2, [-1.2; 1.0], x -> [exp(x[1] - 1) - 1], [0.0], [0.0])
    output = dummy_linesearch_solver(nlp, merit_constructor=merit_constructor, rtol=0)
    @test isapprox(output.solution, ones(2), rtol=1e-6)
    @test output.objective < 1e-6
    @test output.primal_feas < 1e-6
    @test output.dual_feas < 1e-6

    nlp = ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
    output = dummy_trust_region_solver(nlp, merit_constructor=merit_constructor, rtol=0)
    @test isapprox(output.solution, ones(2), rtol=1e-6)
    @test output.objective < 1e-6
    @test output.primal_feas == 0
    @test output.dual_feas < 1e-6

    nlp = ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2, [-1.2; 1.0], x -> [exp(x[1] - 1) - 1], [0.0], [0.0])
    output = dummy_trust_region_solver(nlp, merit_constructor=merit_constructor, rtol=0)
    @test isapprox(output.solution, ones(2), rtol=1e-6)
    @test output.objective < 1e-6
    @test output.primal_feas < 1e-6
    @test output.dual_feas < 1e-6
  end
end

@testset "Merit functions" begin
  merits = [ # Name, Constructor, ϕ(nlp, x, η)
    (
      "L1Merit",
      L1Merit,
      (nlp, x, η) -> obj(nlp, x) + η * norm(cons(nlp, x), 1)
    ),
    (
      "AugLagMerit",
      (nlp, η; kwargs...) -> AugLagMerit(nlp, η, y=-ones(typeof(η), nlp.meta.ncon); kwargs...),
      (nlp, x, η) -> obj(nlp, x) - sum(cons(nlp, x)) + η * norm(cons(nlp, x))^2 / 2 # y = (-1,…,-1)ᵀ
    )
  ]
  for (name, merit, ϕ) in merits
    @testset "Merit $name" begin
      test_merit(merit, ϕ)
    end
  end
end