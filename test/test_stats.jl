function dummy_solver(nlp :: AbstractNLPModel;
                      x :: AbstractVector = nlp.meta.x0,
                      atol :: Real = sqrt(eps(eltype(x))),
                      rtol :: Real = sqrt(eps(eltype(x))),
                      max_eval :: Int = 1000,
                      max_time :: Float64 = 30.0,
                      max_iter :: Int = 1000
                     )

  return GenericExecutionStats(:first_order, nlp,
                               objective=obj(nlp, x),
                               dual_feas=norm(grad(nlp, x)),
                               primal_feas=nlp.meta.ncon > 0 ? norm(cons(nlp, x)) : zero(eltype(x)),
                               solution=x,
                              )
end

function test_stats()
  nlp = ADNLPModel(x->dot(x,x), zeros(2))
  stats = GenericExecutionStats(:first_order, nlp, objective=1.0, dual_feas=1e-12,
                         solution=ones(100), iter=10,
                         solver_specific=Dict(:matvec=>10, :dot=>25,
                                              :empty_vec=>[],
                                              :small_vec=>[2.0;3.0],
                                              :axpy=>20, :ray=>-1 ./ (1:100)))

  show(stats)
  print(stats)
  println(stats)
  open("teststats.out", "w") do f
    println(f, stats)
  end

  println(stats, showvec=(io,x)->print(io,x))
  open("teststats.out", "a") do f
    println(f, stats, showvec=(io,x)->print(io,x))
  end

  line = [:status, :neval_obj, :objective, :iter]
  for field in line
    value = statsgetfield(stats, field)
    println("$field -> $value")
  end
  println(statshead(line))
  println(statsline(stats, line))

  @testset "Testing inference" begin
    for T in (Float16, Float32, Float64, BigFloat)
      nlp = ADNLPModel(x->dot(x, x), ones(T, 2))

      stats = GenericExecutionStats(:first_order, nlp)
      @test stats.status == :first_order
      @test typeof(stats.objective) == T
      @test typeof(stats.dual_feas) == T
      @test typeof(stats.primal_feas) == T

      nlp = ADNLPModel(x->dot(x, x), ones(T, 2), c=x->[sum(x)-1], lcon=[0.0], ucon=[0.0])

      stats = GenericExecutionStats(:first_order, nlp)
      @test stats.status == :first_order
      @test typeof(stats.objective) == T
      @test typeof(stats.dual_feas) == T
      @test typeof(stats.primal_feas) == T
    end
  end

  @testset "Test throws" begin
    @test_throws Exception GenericExecutionStats(:bad, nlp)
    @test_throws Exception GenericExecutionStats(:unkwown, nlp, bad=true)
  end

  @testset "Testing Dummy Solver with multi-precision" begin
    for T in (Float16, Float32, Float64, BigFloat)
      nlp = ADNLPModel(x->dot(x, x), ones(T, 2))

      stats = dummy_solver(nlp)
      @test stats.status == :first_order
      @test typeof(stats.objective) == T
      @test typeof(stats.dual_feas) == T
      @test typeof(stats.primal_feas) == T

      nlp = ADNLPModel(x->dot(x, x), ones(T, 2), c=x->[sum(x)-1], lcon=[0.0], ucon=[0.0])

      stats = dummy_solver(nlp)
      @test stats.status == :first_order
      @test typeof(stats.objective) == T
      @test typeof(stats.dual_feas) == T
      @test typeof(stats.primal_feas) == T
    end
  end
end

test_stats()
