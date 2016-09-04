facts("Steihaug") do
  context("Solution is inside Δ") do
    n = 100
    Λ = linspace(1e-4, 1.0, n)
    for t = 1:100
      (Q,R) = qr(rand(n,n))
      B = Q'*diagm(Λ)*Q
      g = -1 + 2*rand(n)
      sol = -B\g
      Δ = 10*norm(sol)
      z, status = steihaug(B, g, Δ)
      @fact z --> roughly(sol, 1e-5 * (1 + norm(z)))
    end
  end
  context("Solution is outside Δ") do
    n = 100
    Λ = linspace(1e-4, 1.0, n)
    for t = 1:100
      (Q,R) = qr(rand(n,n))
      B = Q'*diagm(Λ)*Q
      g = -1 + 2*rand(n)
      sol = -B\g
      Δ = (0.1 + 0.8*rand())*norm(sol)
      sc = -min(dot(g,g)/dot(g,B*g), Δ/norm(g))*g
      z, status = steihaug(B, g, Δ)
      q(d) = 0.5*dot(d,B*d) + dot(d,g)
      @fact q(z) --> less_than(q(sc) + 1e-3)
    end
  end
  context("Indefinite matrix") do
    n = 100
    for t = 1:100
      m = rand(2:n-2)
      Λ = vcat(linspace(-1.0, -1e-4, m), linspace(1e-4, 1.0, n-m))
      (Q,R) = qr(rand(n,n))
      B = Q'*diagm(Λ)*Q
      g = -1 + 2*rand(n)
      Δ = 100*rand()
      sc = -min(dot(g,g)/dot(g,B*g), Δ/norm(g))*g
      z, status = steihaug(B, g, Δ)
      q(d) = 0.5*dot(d,B*d) + dot(d,g)
      @fact q(z) --> less_than(q(sc) + 1e-3)
    end
  end
end
