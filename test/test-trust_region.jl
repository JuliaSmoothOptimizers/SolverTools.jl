function trust_region_allocs_test(::Type{Solver}, ::Type{S}) where {Solver, S}
  T = eltype(S)
  @testset "Allocation - NLP" begin
    nlp = BROWNDEN(S)
    n = nlp.meta.nvar
    Δ₀ = T(10)
    tr = Solver(S, n, Δ₀)
    x = fill!(S(undef, n), 0)
    d = fill!(S(undef, n), 1)
    f, ft, Δm, slope = T(2), T(1), -T(1), -T(1)
    al = @wrappedallocs aredpred!(tr, nlp, f, ft, Δm, x, d, slope)
    @test al == 0
    al = @wrappedallocs aredpred!(tr, nlp, ft, ft, Δm, x, d, slope)
    @test al == 0
  end

  @testset "Allocation - NLS" begin
    nlp = LLS(S)
    n = nlp.meta.nvar
    Δ₀ = T(10)
    tr = Solver(S, n, Δ₀)
    x = fill!(S(undef, n), 0)
    d = fill!(S(undef, n), 1)
    f, ft, Δm, slope = T(2), T(1), -T(1), -T(1)
    Fx = fill!(S(undef, nlp.nls_meta.nequ), 1)
    al = @wrappedallocs aredpred!(tr, nlp, Fx, f, ft, Δm, x, d, slope)
    @test al == 0
    al = @wrappedallocs aredpred!(tr, nlp, Fx, ft, ft, Δm, x, d, slope)
    @test al == 0
  end
end

function test_aredpred(nlp, ::Type{Solver}, ::Type{S}) where {Solver, S}
  T = eltype(S)
  x = nlp.meta.x0
  d = fill!(S(undef, 2), -1)
  xt = x + d
  f = obj(nlp, x)
  ft = obj(nlp, xt)
  Δm = ft - f

  tr = Solver(S, 2, T(100))
  ared, pred = aredpred!(tr, nlp, f, ft, Δm, xt, d, dot(grad(nlp, x), d))
  @test abs(ared - pred) < 1e-12

  d = fill!(S(undef, 2), -1e-12)
  xt = x + d
  ft = obj(nlp, xt)
  Δm = ft - f
  ared, pred = aredpred!(tr, nlp, f, ft, Δm, xt, d, dot(grad(nlp, x), d))
  @test abs(ared - pred) < 1e-12
end

@testset "Trust Region" begin
  S = Vector{Float64}
  @testset "aredpred-$(nlp.meta.name) - $S - $TRSolver" for nlp in (
      ADNLPModel(x -> x[1]^2 + 4 * x[2]^2, fill!(S(undef, 2), 1), name = "NLP", matrix_free = true),
      ADNLSModel(x -> [x[1]; 2 * x[2]], fill!(S(undef, 2), 1), 2, name = "NLS", matrix_free = true),
    ),
    TRSolver in (TrustRegion, TRONTrustRegion)

    test_aredpred(nlp, TRSolver, S)
  end

  if CUDA.functional()
    CUDA.allowscalar() do
      S = CuVector{Float64}
      @testset "aredpred-$(nlp.meta.name) - $S - $TRSolver" for nlp in (
          ADNLPModel(
            x -> x[1]^2 + 4 * x[2]^2,
            fill!(S(undef, 2), 1),
            name = "NLP",
            matrix_free = true,
          ),
          ADNLSModel(
            x -> [x[1]; 2 * x[2]],
            fill!(S(undef, 2), 1),
            2,
            name = "NLS",
            matrix_free = true,
          ),
        ),
        TRSolver in (TrustRegion, TRONTrustRegion)

        test_aredpred(nlp, TRSolver, S)
      end
    end
  end

  @testset "BasicTrustRegion" begin
    S = Vector{Float64}
    T = eltype(S)
    Δ₀ = T(10)
    tr = TrustRegion(S, 2, Δ₀)
    tr.ratio = T(1)
    @test acceptable(tr) == true

    tr.ratio = T(10)
    update!(tr, Δ₀)
    @test tr.radius > Δ₀
    reset!(tr)
    tr.ratio = -T(1)
    update!(tr, Δ₀)
    @test tr.radius < Δ₀
    reset!(tr)

    if VERSION ≥ v"1.7"
      trust_region_allocs_test(TrustRegion, S)
    end
  end

  @testset "TRONTrustRegion" begin
    S = Vector{Float64}
    T = eltype(S)
    Δ₀ = T(10)
    tr = TRONTrustRegion(S, 2, Δ₀)
    tr.ratio = T(1)
    @test acceptable(tr) == true

    tr.ratio = tr.acceptance_threshold - 1
    update!(tr, Δ₀)
    @test tr.radius < Δ₀
    reset!(tr)
    tr.ratio = (tr.acceptance_threshold + tr.decrease_threshold) / 2
    update!(tr, Δ₀)
    @test tr.radius < Δ₀
    reset!(tr)
    tr.ratio = (tr.decrease_threshold + tr.increase_threshold) / 2
    update!(tr, Δ₀)
    @test tr.radius < Δ₀
    reset!(tr)
    tr.ratio = tr.increase_threshold + 1
    tr.quad_min = T(2)
    update!(tr, Δ₀)
    @test tr.radius > Δ₀

    if VERSION ≥ v"1.7"
      trust_region_allocs_test(TRONTrustRegion, S)
    end
  end

  struct Workspace{S}
    ∇fnext::S
  end

  @testset "ARTrustRegion" begin
    S = Vector{Float64}
    T = eltype(S)
    Δ₀ = T(0.8)
    ar = ARTrustRegion(T(0.8))
    @test ar.α₀ == 0.8
    @test ar.α == 0.8
    @test ar.max_α ≈ 1 / sqrt(eps(T))
    @test ar.acceptance_threshold == 0.1
    @test ar.increase_threshold == 0.75
    @test ar.reduce_threshold == 0.1
    @test ar.increase_factor == 5.0
    @test ar.decrease_factor == 0.1
    @test ar.large_decrease_factor == 0.01
    @test ar.max_unsuccinarow == 30

    function NLPModels.grad!(nlp::AbstractNLPModel, x, workspace::Workspace)
      return grad!(nlp, x, workspace.∇fnext)
    end

    nlp = BROWNDEN(S)
    f = T(1)
    Δf = T(0.2)
    Δq = T(0.4)
    slope = T(0.5)
    d = S([1.0, 2.0, 3.0, 4.0])
    xnext = S([0.5, 1.5, 2.5, 3.5])
    workspace = Workspace(S([0.8, 1.8, 2.8, 3.8]))
    robust = true

    r, good_grad, gnext = SolverTools.compute_r(nlp, f, Δf, Δq, slope, d, xnext, workspace, robust)

    @test r == 0.5
    @test good_grad == false
    @test gnext == [0.8, 1.8, 2.8, 3.8]

    Δq = T(1e-13)
    r, good_grad, gnext = SolverTools.compute_r(nlp, f, Δf, Δq, slope, d, xnext, workspace, robust)

    @test r == 3.833532881060353e19
    @test good_grad == true
    @test gnext ==
          [-909318.2269441625, -3.4465870681692427e6, 47999.033762642575, -2142.750031496445]

    workspace = Workspace(S([0.8, 1.8, 2.8, 3.8]))
    robust = false
    r, good_grad, gnext = SolverTools.compute_r(nlp, f, Δf, Δq, slope, d, xnext, workspace, robust)

    @test r == 2.0e12
    @test good_grad == false
    @test gnext == [0.8, 1.8, 2.8, 3.8]
  end
end
