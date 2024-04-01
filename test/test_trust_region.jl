@testset "Trust Region" begin
  @testset "aredpred-$(nlp.meta.name)" for nlp in (
    ADNLPModel(x -> x[1]^2 + 4 * x[2]^2, ones(2), name = "NLP"),
    ADNLSModel(x -> [x[1]; 2 * x[2]], ones(2), 2, name = "NLS"),
  )
    x = nlp.meta.x0
    d = -ones(2)
    xt = x + d
    f = obj(nlp, x)
    ft = obj(nlp, xt)
    Δm = ft - f
    tr = TrustRegion(2, 100.0)
    ared, pred = aredpred!(tr, nlp, f, ft, Δm, xt, d, dot(grad(nlp, x), d))
    @test abs(ared - pred) < 1e-12
    tr = TRONTrustRegion(2, 100.0)
    ared, pred = aredpred!(tr, nlp, f, ft, Δm, xt, d, dot(grad(nlp, x), d))
    @test abs(ared - pred) < 1e-12
    d = -1e-12 * ones(2)
    xt = x + d
    ft = obj(nlp, xt)
    Δm = ft - f
    ared, pred = aredpred!(tr, nlp, f, ft, Δm, xt, d, dot(grad(nlp, x), d))
    @test abs(ared - pred) < 1e-12
  end

  @testset "BasicTrustRegion" begin
    Δ₀ = 10.0
    tr = TrustRegion(2, Δ₀)
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

    if VERSION ≥ v"1.7"
      @testset "Allocation" begin
        nlp = BROWNDEN()
        n = nlp.meta.nvar
        Δ₀ = 10.0
        tr = TrustRegion(n, Δ₀)
        x = zeros(n)
        d = ones(n)
        f, ft, Δm, slope = 2.0, 1.0, -1.0, -1.0
        al = @wrappedallocs aredpred!(tr, nlp, f, ft, Δm, x, d, slope)
        @test al == 0
        al = @wrappedallocs aredpred!(tr, nlp, ft, ft, Δm, x, d, slope)
        @test al == 0
      end

      @testset "Allocation - nonlinear least squares" begin
        nlp = LLS()
        n = nlp.meta.nvar
        Δ₀ = 10.0
        tr = TrustRegion(n, Δ₀)
        x = zeros(n)
        d = ones(n)
        f, ft, Δm, slope = 2.0, 1.0, -1.0, -1.0
        Fx = zeros(nlp.nls_meta.nequ)
        al = @wrappedallocs aredpred!(tr, nlp, Fx, f, ft, Δm, x, d, slope)
        @test al == 0
        al = @wrappedallocs aredpred!(tr, nlp, Fx, ft, ft, Δm, x, d, slope)
        @test al == 0
      end
    end
  end

  @testset "TRONTrustRegion" begin
    Δ₀ = 10.0
    tr = TRONTrustRegion(2, Δ₀)
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

    if VERSION ≥ v"1.7"
      @testset "Allocation" begin
        nlp = BROWNDEN()
        n = nlp.meta.nvar
        Δ₀ = 10.0
        tr = TRONTrustRegion(n, Δ₀)
        x = zeros(n)
        d = ones(n)
        f, ft, Δm, slope = 2.0, 1.0, -1.0, -1.0
        al = @wrappedallocs aredpred!(tr, nlp, f, ft, Δm, x, d, slope)
        @test al == 0
        al = @wrappedallocs aredpred!(tr, nlp, ft, ft, Δm, x, d, slope)
        @test al == 0
      end

      @testset "Allocation - nonlinear least squares" begin
        nlp = LLS()
        n = nlp.meta.nvar
        Δ₀ = 10.0
        tr = TRONTrustRegion(n, Δ₀)
        x = zeros(n)
        d = ones(n)
        f, ft, Δm, slope = 2.0, 1.0, -1.0, -1.0
        Fx = zeros(nlp.nls_meta.nequ)
        al = @wrappedallocs aredpred!(tr, nlp, Fx, f, ft, Δm, x, d, slope)
        @test al == 0
        al = @wrappedallocs aredpred!(tr, nlp, Fx, ft, ft, Δm, x, d, slope)
        @test al == 0
      end
    end
  end
  
  struct Workspace
    ∇fnext::Array{Float64}
  end
  @testset "ARTrustRegion" begin
    Δ₀ = 0.8
    ar = ARTrustRegion(0.8)
    @test ar.α₀ == 0.8 
    @test ar.α == 0.8
    @test ar.max_α ≈ 1 / sqrt(eps(typeof(0.8)))
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

    nlp = BROWNDEN()
    f = 1.0 
    Δf = 0.2 
    Δq = 0.4
    slope = 0.5 
    d = [1.0, 2.0, 3.0, 4.0] 
    xnext = [0.5, 1.5, 2.5, 3.5] 
    workspace = Workspace([0.8, 1.8, 2.8, 3.8])
    robust = true

    r, good_grad, gnext = SolverTools.compute_r(nlp, f, Δf, Δq, slope, d, xnext, workspace, robust)

    @test r == 0.5
    @test good_grad == false
    @test gnext == [0.8, 1.8, 2.8, 3.8]

    Δq = 1e-13
    r, good_grad, gnext = SolverTools.compute_r(nlp, f, Δf, Δq, slope, d, xnext, workspace, robust)

    @test r == 3.833532881060353e19
    @test good_grad == true
    @test gnext == [-909318.2269441625, -3.4465870681692427e6, 47999.033762642575, -2142.750031496445]

    workspace = Workspace([0.8, 1.8, 2.8, 3.8])
    robust = false
    r, good_grad, gnext = SolverTools.compute_r(nlp, f, Δf, Δq, slope, d, xnext, workspace, robust)

    @test r == 2.0e12
    @test good_grad == false
    @test gnext == [0.8, 1.8, 2.8, 3.8]
  end
end
