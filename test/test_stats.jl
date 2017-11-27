function test_stats()
  stats = ExecutionStats(:first_order, objective=1.0, dual_feas=1e-12,
                         solution=ones(100),
                         solver_specific=Dict(:matvec=>10, :dot=>25,
                                              :axpy=>20, :ray=>-1 ./ (1:100)))

  println(stats)
  open("teststats.out", "w") do f
    println(f, stats)
  end

  println(stats, showvec=(io,x)->print(io,x))
  open("teststats.out", "a") do f
    println(f, stats, showvec=(io,x)->print(io,x))
  end
end

test_stats()
