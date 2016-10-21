context("Simple tests") do
  facts("Quadratic with linear constraints") do
    f(x) = x[1]^2 + x[2]^2
    c(x) = [x[1] + x[2] - 1]
    x0 = zeros(2)

    nlp = ADNLPModel(f, x0, c=c, lcon=[0], ucon=[0])

    x, fx, gpx, cx, iter, optimal, tired, status = lagaug(nlp, verbose=true)
    println("x = $x")
    println("status = $status")
    @fact x --> roughly([0.5;0.5], 1e-6)
    @fact fx --> roughly(f([0.5;0.5]), 1e-12)
    @fact gpx --> roughly(0.0, 1e-6)
    @fact cx --> roughly(0.0, 1e-6)
    @fact optimal --> true
  end
end
