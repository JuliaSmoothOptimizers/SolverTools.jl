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

    stats = tron(nlp)
    @test isapprox(stats.solution, zeros(2), rtol=1e-6)
    @test isapprox(stats.obj, 0.0, rtol=1e-12)
    @test stats.dual_feas < 1e-6
    @test stats.status == :first_order
  end

  @testset "Bounds" begin
    x0 = [1.0; 2.0]
    f(x) = dot(x,x)/2
    l = [0.5; 0.25]
    u = [1.2; 1.5]

    nlp = ADNLPModel(f, x0, lvar=l, uvar=u)

    stats = tron(nlp)
    @test stats.solution ≈ l
    @test stats.obj == f(l)
    @test stats.dual_feas < 1e-6
    @test stats.status == :first_order
  end

  @testset "Rosenbrock" begin
    x0 = [-1.2; 1.0]
    f(x) = 1.0 + (x[1]-1)^2 + 100*(x[2]-x[1]^2)^2

    nlp = ADNLPModel(f, x0)

    stats = tron(nlp)
    @test isapprox(stats.solution, [1.0;1.0], rtol=1e-3)
    @test isapprox(stats.obj, 1.0, rtol=1e-5)
    @test stats.dual_feas < 1e-6
    @test stats.status == :first_order
  end

  @testset "Rosenbrock inactive bounds" begin
    l = [0.5; 0.25]
    u = [1.2; 1.5]
    x0 = (l+u)/2
    f(x) = 1.0 + (x[1]-1)^2 + 100*(x[2]-x[1]^2)^2

    nlp = ADNLPModel(f, x0, lvar=l, uvar=u)

    stats = tron(nlp)
    @test isapprox(stats.solution, [1.0;1.0], rtol=1e-3)
    @test isapprox(stats.obj, 1.0, rtol=1e-5)
    @test stats.dual_feas < 1e-6
    @test stats.status == :first_order
  end

  @testset "Rosenbrock active bounds" begin
    l = [0.5; 0.25]
    u = [0.9; 1.5]
    x0 = (l+u)/2
    f(x) = 1.0 + (x[1]-1)^2 + 100*(x[2]-x[1]^2)^2

    nlp = ADNLPModel(f, x0, lvar=l, uvar=u)

    sol = [0.9; 0.81]

    stats = tron(nlp)
    @test isapprox(stats.solution, sol, rtol=1e-3)
    @test isapprox(stats.obj, f(sol), rtol=1e-5)
    @test stats.dual_feas < 1e-6
    @test stats.status == :first_order
  end
end

@testset "Fixed variables" begin
  @testset "One fixed" begin
    x0 = ones(3)
    l = [1.0; 0.0; 0.0]
    u = [1.0; 2.0; 2.0]
    f(x) = 0.5*dot(x - 3, x - 3)
    nlp = ADNLPModel(f, x0, lvar=l, uvar=u)
    stats = tron(nlp)
    @test stats.status == :first_order
    @test stats.dual_feas < 1e-6
    @test isapprox(stats.solution, [1.0; 2.0; 2.0], rtol=1e-3)
    @test stats.solution[1] == 1.0
  end

  @testset "All fixed" begin
    n = 100
    x0 = zeros(n)
    l = 0.9*ones(n)
    u = copy(l)
    f(x) = sum(x.^4)
    nlp = ADNLPModel(f, x0, lvar=l, uvar=u)
    stats = tron(nlp)
    @test stats.status == :first_order
    @test stats.dual_feas == 0.0
    @test stats.solution == l
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

      stats = tron(nlp)
      normg0 = norm(B*(x0-r))
      @test norm(stats.solution - r) < 1e-4 * normg0
      @test isapprox(stats.obj, 1.0, rtol=1e-6)
      @test stats.dual_feas < 1e-6 * normg0
      @test stats.status == :first_order
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

      stats = tron(nlp)
      normg0 = norm(B*(x0-r))

      @test norm(stats.solution - r) < 1e-3
      @test isapprox(stats.obj, f(r), rtol=1e-4)
      @test stats.dual_feas < 1e-4
      @test stats.status == :first_order
      @test stats.iter <= 10000
      @test stats.status == :first_order
    end
  end
end

@testset "Non-first-order exits" begin
  n = 10
  x0 = 10*ones(n)
  f(x) = begin
    sleep(0.1)
    1e7*sum(exp.(x))
  end

  nlp = ADNLPModel(f, x0)

  @testset "Iteration limit" begin
    stats = tron(nlp, itmax=1)
    @test stats.iter == 1
    @test stats.status == :max_iter
  end

  @testset "Time limit" begin
    stats = tron(nlp, timemax=0)
    @test stats.elapsed_time > 0
    @test stats.status == :max_time
  end

  @testset "x0 outside box" begin
    l = rand(n)
    u = l + rand(n)
    x0 = u + rand(n)
    nlp = ADNLPModel(x->dot(x,x), x0, lvar=l, uvar=u)
    stats = tron(nlp)
    @test norm(stats.solution - l) < 1e-3
    @test isapprox(stats.obj, dot(l,l), rtol=1e-5)
    @test stats.dual_feas < 1e-4
    @test stats.status == :first_order
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

  stats = tron(nlp, timemax=600)
  @test isapprox(stats.solution, ones(n), rtol=1e-5*n)
  @test isapprox(stats.obj, 1.0, rtol=1e-3)
  @test stats.dual_feas < 1e-3*n
  @test stats.status == :first_order

  nlp = ADNLPModel(f, x0, lvar=zeros(n), uvar=0.3*ones(n))

  stats = tron(nlp, timemax=600)
  @test stats.dual_feas < 1e-3*n
  @test stats.status == :first_order
end

@static if is_unix()
  @testset "CUTEst" begin
    problems = CUTEst.select(max_var=10, max_con=0, only_bnd_var=true)
    stline = statshead([:obj, :dual_feas, :elapsed_time, :iter, :neval_obj,
                        :neval_grad, :neval_hprod, :status])
    @printf("%8s  %5s  %4s  %s\n", "Problem", "n", "type", stline)
    for p in problems
      nlp = CUTEstModel(p)
      stats = tron(nlp, timemax=3.0)
      finalize(nlp)

      ctype = length(nlp.meta.ifree) == nlp.meta.nvar ? "unc" : "bnd"
      stline = statsline(stats, [:obj, :dual_feas, :elapsed_time, :iter, :neval_obj,
                                 :neval_grad, :neval_hprod, :status])
      @printf("%8s  %5d  %4s  %s\n", p, nlp.meta.nvar, ctype, stline)
    end
  end
end
