using Optimize, NLPModels, Test

function multiprecision_test()
  for T in (Float16, Float32, Float64, BigFloat)
    ϵ = eps(T)^T(1/4)
    nlp = ADNLPModel(x -> (x[1] - 1)^2 + 4 * (x[2] - x[1]^2)^2, T[-1.2; 1.0])
    for mtd in [trunk, lbfgs, tron]
      stats = with_logger(NullLogger()) do
        mtd(nlp, atol=ϵ, rtol=ϵ)
      end
      @test stats.objective isa T
      @test eltype(stats.solution) == T

      stats = with_logger(NullLogger()) do
        mtd(nlp, x=zeros(T, 2), atol=ϵ, rtol=ϵ)
      end
      @test stats.objective isa T
      @test eltype(stats.solution) == T
    end
  end
end

multiprecision_test()
