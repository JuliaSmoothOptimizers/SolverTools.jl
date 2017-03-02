using OptimizationProblems

context("Simple tests") do
  facts("Quadratic with linear constraints") do
    for D in [ ones(2), linspace(1e-2, 100, 2), linspace(1e-2, 1e2, 10), linspace(1e-4, 1e4, 10), linspace(1e-4, 1e4, 30) ]
      n = length(D)
      for x0 in Any[ zeros(n), ones(n), -collect(linspace(1, n, n)) ]
        nlp = ADNLPModel(x->dot(x,D.*x), x0,
                         c=x->[sum(x)-1], lcon=[0], ucon=[0])

        λ = -1/sum(1./D)
        stats = lagaug(nlp, verbose=false, rtol=0.0)
        x, fx, gpx, cx = stats.solution, stats.obj, stats.opt_norm, stats.feas_norm
        println(stats)
        @fact x --> roughly(-λ./D, 1e-4)
        @fact fx --> roughly(-λ, 1e-5)
        @fact gpx --> roughly(0.0, 1e-4)
        @fact cx --> roughly(0.0, 1e-4)
      end
    end
  end

  facts("HS6") do
    nlp = MathProgNLPModel(hs6())

    stats = lagaug(nlp, verbose=false, rtol=0.0)
    x, fx, gpx, cx = stats.solution, stats.obj, stats.opt_norm, stats.feas_norm
    println(stats)
    @fact x --> roughly(ones(2), 1e-4)
    @fact fx --> roughly(0.0, 1e-5)
    @fact gpx --> roughly(0.0, 1e-4)
    @fact cx --> roughly(0.0, 1e-4)
  end

  facts("HS7") do
    nlp = MathProgNLPModel(hs7())

    stats = lagaug(nlp, verbose=false, rtol=0.0)
    x, fx, gpx, cx = stats.solution, stats.obj, stats.opt_norm, stats.feas_norm
    println(stats)
    @fact x --> roughly([0.0; sqrt(3)], 1e-4)
    @fact fx --> roughly(-sqrt(3), 1e-5)
    @fact gpx --> roughly(0.0, 1e-4)
    @fact cx --> roughly(0.0, 1e-4)
  end

  facts("HS8") do
    nlp = MathProgNLPModel(hs8())

    stats = lagaug(nlp, verbose=false, rtol=0.0)
    x, fx, gpx, cx = stats.solution, stats.obj, stats.opt_norm, stats.feas_norm
    println(stats)
    sol = sqrt( (25 + sqrt(301)*[1;-1])/2 )
    @fact abs(x) --> roughly(sol, 1e-4)
    @fact fx --> roughly(-1.0, 1e-5)
    @fact gpx --> roughly(0.0, 1e-4)
    @fact cx --> roughly(0.0, 1e-4)
  end

  facts("HS9") do
    nlp = MathProgNLPModel(hs9())

    stats = lagaug(nlp, verbose=false, rtol=0.0)
    x, fx, gpx, cx = stats.solution, stats.obj, stats.opt_norm, stats.feas_norm
    println(stats)
    @fact (x + [3;4]) .% [12; 16] --> roughly(zeros(2), 1e-4)
    @fact fx --> roughly(-0.5, 1e-5)
    @fact gpx --> roughly(0.0, 1e-4)
    @fact cx --> roughly(0.0, 1e-4)
  end

  facts("HS26") do
    nlp = MathProgNLPModel(hs26())

    stats = lagaug(nlp, verbose=false, rtol=0.0)
    x, fx, gpx, cx = stats.solution, stats.obj, stats.opt_norm, stats.feas_norm
    println(stats)
    @fact fx --> roughly(0.0, 1e-5)
    @fact gpx --> roughly(0.0, 1e-4)
    @fact cx --> roughly(0.0, 1e-4)
  end

  facts("HS27") do
    nlp = MathProgNLPModel(hs27())

    stats = lagaug(nlp, verbose=false, rtol=0.0)
    x, fx, gpx, cx = stats.solution, stats.obj, stats.opt_norm, stats.feas_norm
    println(stats)
    @fact x --> roughly([-1.0; 1.0; 0.0], 1e-4)
    @fact fx --> roughly(0.04, 1e-5)
    @fact gpx --> roughly(0.0, 1e-4)
    @fact cx --> roughly(0.0, 1e-4)
  end

end

context("Larger problems") do
  facts("Quadratic with linear constraints") do
    for n = 10.^(2:5)
      for D in [ ones(n), linspace(1e-2, 100, n)]
        for x0 in Any[ zeros(n), ones(n), -collect(linspace(1, n, n)) ]
          nlp = SimpleNLPModel(x->dot(x,D.*x)/2, x0,
                               g = x->D.*x,
                               Hp! = (x,v,Hv;y=[],obj_weight=1.0)->Hv .= obj_weight*D.*v,
                               c=x->[sum(x)-1], lcon=[0], ucon=[0],
                               Jp = (x,v)->[sum(v)],
                               Jtp = (x,v)->ones(n)*v[1])

          λ = -1/sum(1./D)
          stats = lagaug(nlp, verbose=false, rtol=0.0)
          x, fx, gpx, cx = stats.solution, stats.obj, stats.opt_norm, stats.feas_norm
          @fact x --> roughly(-λ./D, 1e-4)
          @fact fx --> roughly(-λ/2, 1e-5)
          @fact gpx --> roughly(0.0, 1e-4)
          @fact cx --> roughly(0.0, 1e-4)
        end
      end
    end
  end
end
