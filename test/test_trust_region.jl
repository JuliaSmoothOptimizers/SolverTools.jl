@testset "Trust Region" begin
  @testset "aredpred" begin
    nlp = ADNLPModel(x -> x[1]^2 + 4 * x[2]^2, ones(2))
    x = nlp.meta.x0
    d = -ones(2)
    xt = x + d
    f = obj(nlp, x)
    ft = obj(nlp, xt)
    Δm = ft - f
    ared, pred = aredpred(nlp, f, ft, Δm, xt, d, dot(grad(nlp, x), d))
    @test abs(ared - pred) < 1e-12
    tr = TRONTrustRegion(100.0)
    ared, pred, qmin = aredpred(tr, nlp, f, ft, Δm, xt, d, dot(grad(nlp, x), d))
    @test abs(ared - pred) < 1e-12
    d = -1e-12 * ones(2)
    xt = x + d
    ft = obj(nlp, xt)
    Δm = ft - f
    ared, pred = aredpred(nlp, f, ft, Δm, xt, d, dot(grad(nlp, x), d))
    @test abs(ared - pred) < 1e-12
  end

  @testset "BasicTrustRegion" begin
    Δ₀ = 10.0
    tr = TrustRegion(Δ₀)
    tr.ratio = 1.0
    @test acceptable(tr) == true

    tr.ratio = 10.0
    update!(tr, Δ₀)
    @test tr.radius > Δ₀
    reset!(tr)
    tr.ratio = -1.0
    update!(tr, Δ₀)
    @test tr.radius < Δ₀
    reset!(tr)
  end

  @testset "TRONTrustRegion" begin
    Δ₀ = 10.0
    tr = TRONTrustRegion(Δ₀)
    tr.ratio = 1.0
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
    tr.quad_min = 2.0
    update!(tr, Δ₀)
    @test tr.radius > Δ₀
  end
end
