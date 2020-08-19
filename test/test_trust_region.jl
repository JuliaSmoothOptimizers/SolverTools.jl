@testset "Trust Region" begin
  tr_methods = [:basic, :tron]

  @testset "Unconstrained problems" begin
    for tr_method in tr_methods
      @testset "Method $tr_method" begin
        nlp = ADNLPModel(x -> x[1]^2 + 4 * x[2]^2, ones(2))
        x = nlp.meta.x0
        d = -ones(2)
        Δ = norm(d)
        Δq = dot(grad(nlp, x), d) + dot(d, hprod(nlp, x, d)) / 2
        tro = trust_region(nlp, x, d, Δq, Δ, method=tr_method, update_obj_at_x=true)
        @test tro.status == :great
        @test tro.Δ ≥ Δ
        @test tro.xt == x + d
        @test tro.success

        Δq = -tro.ared / 0.25
        tro = trust_region(nlp, x, d, Δq, Δ, method=tr_method, update_obj_at_x=true)
        @test tro.ρ == 0.25
        @test tro.status == :good
        @test tro.xt == x + d
        @test tro.success

        Δq = -tro.ared / 1e-6
        tro = trust_region(nlp, x, d, Δq, Δ, method=tr_method, update_obj_at_x=true)
        @test tro.ρ == 1e-6
        @test tro.status == :bad
        @test tro.xt == x
        @test !tro.success
      end
    end
  end

  @testset "Constrained problems" begin
    for tr_method in tr_methods
      @testset "Method $tr_method" begin
        nlp = ADNLPModel(x -> x[1]^2 + 4 * x[2]^2, ones(2), x -> [x[1] + 2x[2]], [0.0], [0.0])
        x = nlp.meta.x0
        ϕ = L1Merit(nlp, 1.0)
        d = -ones(2)
        Δ = norm(d)
        Δq = dot(grad(nlp, x), d) + dot(d, hprod(nlp, x, d)) / 2
        Ad = jprod(nlp, x, d)
        tro = trust_region(ϕ, x, d, Δq, Ad, Δ, method=tr_method, update_obj_at_x=true)
        @test tro.status == :great
        @test tro.Δ ≥ Δ
        @test tro.xt == x + d
        @test tro.success

        Δq = -tro.ared / 0.25 - ϕ.η * (norm(ϕ.cx, 1) - norm(ϕ.cx + Ad, 1))
        tro = trust_region(ϕ, x, d, Δq, Ad, Δ, method=tr_method, update_obj_at_x=true)
        @test tro.ρ == 0.25
        @test tro.status == :good
        @test tro.xt == x + d
        @test tro.success

        Δq = -tro.ared / 1e-6 - ϕ.η * (norm(ϕ.cx, 1) - norm(ϕ.cx + Ad, 1))
        tro = trust_region(ϕ, x, d, Δq, Ad, Δ, method=tr_method, update_obj_at_x=true)
        @test tro.ρ == 1e-6
        @test tro.status == :bad
        @test tro.xt == x
        @test !tro.success
      end
    end
  end
end
