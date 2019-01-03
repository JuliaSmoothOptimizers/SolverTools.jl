using Test

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

  @test_throws Exception GenericExecutionStats(:unkwown, nlp, bad=true)
end

test_stats()
