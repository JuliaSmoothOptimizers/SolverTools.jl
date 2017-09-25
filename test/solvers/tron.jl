using LinearOperators

@testset "Test auxiliary functions" begin
  for t = 1:100
    x = rand(5)
    l = rand(5)
    u = rand(5)
    y = zeros(5)
    y_loop = zeros(5)
    project!(y, x, l, u)
    for i = 1:5
      y_loop[i] = max(l[i], min(x[i], u[i]))
    end
    @test y == y_loop

    d = ones(5)
    y_loop = zeros(5)
    project_step!(y, x, d, l, u)
    for i = 1:5
      y_loop[i] = max(l[i], min(x[i] + d[i], u[i])) - x[i]
    end
    @test y == y_loop
  end

  x = zeros(6)
  l = [-Inf; 0.0; -Inf; 0.0; -Inf; -1e-12]
  u = [ 0.0; Inf;  Inf; 0.0; 1e-12;   Inf]
  @test active(x, l, u) == [1; 2; 4; 5; 6]
  @test active(x, l, u, rtol=0.0, atol=0.0) == [1; 2; 4]

  x = ones(2)
  l = zeros(2)
  u = 2*ones(2)
  @test breakpoints(x, ones(2), l, u) == (2, 1.0, 1.0)
  @test breakpoints(x, [0.5; 1.0], l, u) == (2, 1.0, 2.0)
  @test breakpoints(x, [0.0; 1.0], l, u) == (1, 1.0, 1.0)
end

@testset "Simple test" begin
  @testset "No bounds" begin
    x0 = [1.0; 2.0]
    f(x) = dot(x,x)/2

    nlp = ADNLPModel(f, x0, lvar=[-Inf; -Inf], uvar=[Inf; Inf])

    x, fx, π, iter, optimal, tired, status = tron(nlp)
    @test isapprox(x, zeros(2), rtol=1e-6)
    @test isapprox(fx, 0.0, rtol=1e-12)
    @test π < 1e-6
    @test optimal == true
  end

  @testset "Bounds" begin
    x0 = [1.0; 2.0]
    f(x) = dot(x,x)/2
    l = [0.5; 0.25]
    u = [1.2; 1.5]

    nlp = ADNLPModel(f, x0, lvar=l, uvar=u)

    x, fx, π, iter, optimal, tired, status = tron(nlp)
    @test x ≈ l
    @test fx == f(l)
    @test π < 1e-6
    @test optimal == true
  end

  @testset "Rosenbrock" begin
    x0 = [-1.2; 1.0]
    f(x) = 1.0 + (x[1]-1)^2 + 100*(x[2]-x[1]^2)^2

    nlp = ADNLPModel(f, x0)

    x, fx, π, iter, optimal, tired, status = tron(nlp)
    @test isapprox(x, [1.0;1.0], rtol=1e-3)
    @test isapprox(fx, 1.0, rtol=1e-5)
    @test π < 1e-6
    @test optimal == true
  end

  @testset "Rosenbrock inactive bounds" begin
    l = [0.5; 0.25]
    u = [1.2; 1.5]
    x0 = (l+u)/2
    f(x) = 1.0 + (x[1]-1)^2 + 100*(x[2]-x[1]^2)^2

    nlp = ADNLPModel(f, x0, lvar=l, uvar=u)

    x, fx, π, iter, optimal, tired, status = tron(nlp)
    @test isapprox(x, [1.0;1.0], rtol=1e-3)
    @test isapprox(fx, 1.0, rtol=1e-5)
    @test π < 1e-6
    @test optimal == true
  end

  @testset "Rosenbrock active bounds" begin
    l = [0.5; 0.25]
    u = [0.9; 1.5]
    x0 = (l+u)/2
    f(x) = 1.0 + (x[1]-1)^2 + 100*(x[2]-x[1]^2)^2

    nlp = ADNLPModel(f, x0, lvar=l, uvar=u)

    sol = [0.9; 0.81]

    x, fx, π, iter, optimal, tired, status = tron(nlp)
    @test isapprox(x, sol, rtol=1e-3)
    @test isapprox(fx, f(sol), rtol=1e-5)
    @test π < 1e-6
    @test optimal == true
  end
end

@testset "Fixed variables" begin
  @testset "One fixed" begin
    x0 = ones(3)
    l = [1.0; 0.0; 0.0]
    u = [1.0; 2.0; 2.0]
    f(x) = 0.5*dot(x - 3, x - 3)
    nlp = ADNLPModel(f, x0, lvar=l, uvar=u)
    x, fx, π, iter, optimal, tired, status = tron(nlp)
    @test optimal == true
    @test π < 1e-6
    @test isapprox(x, [1.0; 2.0; 2.0], rtol=1e-3)
    @test x[1] == 1.0
  end

  @testset "All fixed" begin
    n = 100
    x0 = zeros(n)
    l = 0.9*ones(n)
    u = copy(l)
    f(x) = sum(x.^4)
    nlp = ADNLPModel(f, x0, lvar=l, uvar=u)
    x, fx, π, iter, optimal, tired, status = tron(nlp)
    @test optimal == true
    @test π == 0.0
    @test x == l
  end
end

@testset "Larger test" begin
  n = 10
  # TODO: Check the bounds on the errors
  (Q,R) = qr(rand(n,n))
  Qop = LinearOperator(Q)
  Λ = linspace(1e-2, 1.0, n)
  B = Qop'*opDiagonal(Λ)*Qop
  r = ones(n)
  x0 = r + 1./Λ
  @testset "Quadratic unconstrained" begin
    for t = 1:10
      f(x) = 1.0 + 0.5 * dot(x-r, B * (x-r))

      nlp = ADNLPModel(f, x0)

      x, fx, π, iter, optimal, tired, status = tron(nlp)
      normg0 = norm(B*(x0-r))
      @test norm(x - r) < 1e-4 * normg0
      @test isapprox(fx, 1.0, rtol=1e-6)
      @test π < 1e-6 * normg0
      @test optimal == true
    end
  end

  @testset "Positive quadratic with bounds" begin
    for t = 1:10
      # Create a problem so that the solution is known
      r = zeros(n)
      l, u, λl, λu = -1-rand(n), 1+rand(n), zeros(n), zeros(n)
      for i = 1:n
        rnd = rand()
        if rnd < 1//3
          r[i] = l[i]
          λl[i] = 1 + rand()
        elseif rnd < 2//3
          r[i] = u[i]
          λu[i] = 1 + rand()
        else
          r[i] = l[i] + (0.1 + 0.8*rand())*(u[i] - l[i])
        end
      end
      v = -B*r - λu + λl
      x0 = [l[i] + rand() * (u[i] - l[i]) for i = 1:n]
      f(x) = 0.5 * dot(x, B*x) + dot(x, v)

      nlp = ADNLPModel(f, x0, lvar=l, uvar=u)

      x, fx, π, iter, optimal, tired, status = tron(nlp)
      normg0 = norm(B*(x0-r))

      @test norm(x - r) < 1e-3
      @test isapprox(fx, f(r), rtol=1e-4)
      @test π < 1e-4
      @test optimal == true
      @test iter <= 10000
      @test status == "first-order stationary"
    end
  end
end

@testset "Non-first-order exits" begin
  n = 10
  x0 = 10*ones(n)
  f(x) = 1e7*sum(exp.(x))

  nlp = ADNLPModel(f, x0)

  @testset "Iteration limit" begin
    x, fx, π, iter, optimal, tired, status = tron(nlp, itmax=1)
    @test iter == 1
    @test tired == true
  end

  @testset "Time limit" begin
    x, fx, π, iter, optimal, tired, status, el_time = tron(nlp, timemax=0)
    @test el_time > 0
    @test tired == true
  end

  @testset "x0 outside box" begin
    l = rand(n)
    u = l + rand(n)
    x0 = u + rand(n)
    nlp = ADNLPModel(x->dot(x,x), x0, lvar=l, uvar=u)
    x, fx, π, iter, optimal, tired, status = tron(nlp)
    @test norm(x - l) < 1e-3
    @test isapprox(fx, dot(l,l), rtol=1e-5)
    @test π < 1e-4
    @test optimal == true
  end
end

@testset "Extended Rosenbrock" begin
  n = 30
  function f(x)
    fx = 1.0
    for i = 1:n-1
      fx += 100*(x[i+1] - x[i]^2)^2 + (x[i] - 1)^2
    end
    return fx
  end
  x0 = [i/(n+1) for i = 1:n]

  nlp = ADNLPModel(f, x0)

  x, fx, π, iter, optimal, tired, status = tron(nlp, timemax=600)
  @test isapprox(x, ones(n), rtol=1e-5*n)
  @test isapprox(fx, 1.0, rtol=1e-3)
  @test π < 1e-3*n
  @test optimal == true

  nlp = ADNLPModel(f, x0, lvar=zeros(n), uvar=0.3*ones(n))

  x, fx, π, iter, optimal, tired, status = tron(nlp, timemax=600)
  @test π < 1e-3*n
  @test optimal == true
end

@static if is_unix()
  @testset "CUTEst" begin
    problems = CUTEst.select(max_var=10, max_con=0, only_bnd_var=true)
    @printf("%8s  %5s  %4s  %9s  %9s  %9s  %6s  %6s  %6s  %6s  %s\n",
            "Problem", "n", "type", "f(x)", "π", "time", "it", "#f", "#g", "#Hp", "status")
    for p in problems
      nlp = CUTEstModel(p)
      x, fx, π, iter, optimal, tired, status, el_time = tron(nlp, timemax=3.0)
      finalize(nlp)

      ctype = length(nlp.meta.ifree) == nlp.meta.nvar ? "unc" : "bnd"
      @printf("%8s  %5d  %4s  %9.2e  %9.2e  %9.2e  %6d  %6d  %6d  %6d  %s\n", p,
              nlp.meta.nvar, ctype, fx, π, el_time, iter, neval_obj(nlp), neval_grad(nlp),
              neval_hprod(nlp), status)
    end
  end
end
