function simple_dixmaanj(n :: Int=99)
  α = 1.0
  β = γ = δ = 0.0625
  n % 3 == 0 || warn("dixmaanj: number of variables adjusted to be a multiple of 3")
  m = max(1, div(n, 3))
  n = 3m
  h(i) = (i / n)^2

  f(x) = 1 + sum((i / n)^2 * α * x[i]^2                 for i = 1:n) +
             sum(β * x[i]^2 * (x[i + 1] + x[i + 1]^2)^2 for i = 1:n-1) +
             sum(γ * x[i]^2 * x[i + m]^4                for i = 1:2m) +
             sum((i / n)^2 * δ * x[i] * x[i + 2m]       for i = 1:m)
  g!(x, gx) = begin
    fill!(gx, 0.0)
    for i = 1:n
      gx[i] = 2 * h(i) * α * x[i] +
              2 * β * (
                       (i < n ? x[i] * (x[i + 1] + x[i + 1]^2)^2 : 0.0) +
                       (i > 1 ? x[i - 1]^2 * (x[i] + x[i]^2) * (1 + 2 * x[i]) : 0.0)) +
              γ * (
                   (i <= 2m ? 2 * x[i] * x[i + m]^4 : 0.0) +
                   (i > m ?   4 * x[i - m]^2 * x[i]^3 : 0.0)) +
              δ * (
                   (i <= m ? h(i) * x[i + 2m] : 0.0) +
                   (i > 2m ? h(i - 2m) * x[i - 2m] : 0.0))
    end
    return gx
  end
  g(x) = begin
    gx = zeros(n)
    return g!(x, gx)
  end
  nnz = 3n - 1
  H(x; y=Float64[], obj_weight=1.0) = begin
    I = zeros(Int, nnz)
    J = zeros(Int, nnz)
    V = zeros(nnz)
    for i = 1:n
      I[i] = i
      J[i] = i
      V[i] = 2 * h(i) * α +
               2 * β * (
                        (i < n ? (x[i + 1] + x[i + 1]^2)^2 : 0.0) +
                        (i > 1 ? x[i - 1]^2 * (1 + 6 * x[i] + 6 * x[i]^2) : 0.0)) +
               γ * (
                    (i <= 2m ?  2 * x[i + m]^4 : 0.0) +
                    (i > m ?   12 * x[i - m]^2 * x[i]^2 : 0.0))
    end
    for i = 1:n-1
      I[n + i] = i + 1
      J[n + i] = i
      V[n + i] = 4 * β * x[i] * (x[i + 1] + x[i + 1]^2) * (1 + 2 * x[i + 1])
    end
    nz = 2n - 1
    for i = 1:2m
      I[nz + i] = i + m
      J[nz + i] = i
      V[nz + i] = 8 * γ * x[i] * x[i + m]^3
    end
    nz += 2m
    for i = 1:m
      I[nz + i] = i + 2m
      J[nz + i] = i
      V[nz + i] = (i / n)^2 * δ
    end
    return sparse(I, J, V)
  end
  Hp!(x, v, Hv; y=Float64[], obj_weight=1.0) = begin
    Hx = H(x)
    Hv .= Hx * v .+ triu(Hx',1) * v
    return Hv
  end
  Hp(x, v; y=Float64[], obj_weight=1.0) = begin
    Hv = zeros(n)
    return Hp!(x, v, Hv)
  end
  nlp = SimpleNLPModel(f, fill(2.0, n), g=g, g! =g!, H=H, Hp=Hp, Hp! =Hp!)
  return nlp
end
