@testset "Linesearch" begin
  @testset "LineModel" begin
    nlp = ADNLPModel(x -> x[1]^2 + 4 * x[2]^2, ones(2))
    x = nlp.meta.x0
    d = -ones(2)
    lm = LineModel(nlp, x, d)
    g = zeros(2)

    @test obj(lm, 0.0) == obj(nlp, x)
    @test grad(lm, 0.0) == dot(grad(nlp, x), d)
    @test grad!(lm, 0.0, g) == dot(grad(nlp, x), d)
    @test g == grad(nlp, nlp.meta.x0)
    @test derivative(lm, 0.0) == dot(grad(nlp, x), d)
    @test derivative!(lm, 0.0, g) == dot(grad(nlp, x), d)
    @test g == grad(nlp, nlp.meta.x0)
    @test hess(lm, 0.0) == dot(d, Symmetric(hess(nlp, x), :L) * d)

    @test obj(lm,  1.0) == 0.0
    @test grad(lm, 1.0) == 0.0
    @test hess(lm, 1.0) == 2d[1]^2 + 8d[2]^2

    redirect!(lm, zeros(2), ones(2))
    @test obj(lm,  0.0) == 0.0
    @test grad(lm, 0.0) == 0.0
    @test hess(lm, 0.0) == 10.0

    @test neval_obj(lm) == 3
    @test neval_grad(lm) == 6
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
end
