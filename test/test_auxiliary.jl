function test_auxiliary()
  @testset "BLAS" begin
    n = 100
    for T in (Float16, Float32, Float64, BigFloat)
      x = rand(T, 100)
      y = rand(T, 100)
      t = one(T) / 2
      s = -one(T) / 4
      @test nrm2(n, x) == norm(x)
      @test dot(n, x, y) == dot(x, y)

      xtd = x + t * y
      SolverTools.axpy!(n, t, y, x)
      @test xtd == x

      xtd = t * y + s * x
      SolverTools.axpby!(n, t, y, s, x)
      @test xtd == x

      y .= x / 2
      scal!(n, t, x)
      @test y == x

      copyaxpy!(n, t, y, x, xtd)
      @test xtd == x + t * y
    end
  end

  @testset "Bounds" begin
    x = [0.0; 1.0; 2.0; 3.0; 4.0]
    ℓ = zeros(5)
    u = 4 * ones(5)
    @test active(x, ℓ, u) == [1, 5]

    d = [1.0; 1.0; 0.0; -1.0; -1.0]
    @test breakpoints(x, d, ℓ, u) == (4, 3.0, 4.0)
    a = @allocated breakpoints(x, d, ℓ, u)
    @test a == 0

    H = diagm(0 => 2 * ones(5), -1 => -ones(4), 1 => -ones(4))
    Hx = zeros(5)
    g = rand(5)
    slope, qx = compute_Hs_slope_qs!(Hx, H, x, g)
    @test slope == dot(g, x)
    @test qx == dot(x, Hx) / 2 + slope
    a = @allocated compute_Hs_slope_qs!(Hx, H, x, g)
    @test a == 0

    x1 = [10.0; 11.0; 12.0; 13.0; 14.0; 15.0; 16.0]
    A = diagm(0 => 2 * ones(7), -1 => -ones(6), 1 => -ones(6))
    As = zeros(7)
    Fx = rand(7)
    slope, qx = compute_As_slope_qs!(As, A, x1, Fx)
    @test slope == dot(As, Fx)
    @test qx == dot(As, As) / 2 + slope
    b = @allocated compute_As_slope_qs!(As, A, x1, Fx)
    @test b == 0

    z = [ℓ[1:2] - rand(2); x[3]; u[4:5] + rand(2)]
    y = rand(5)
    project!(y, z, ℓ, u)
    @test y == [ℓ[1:2]; x[3]; u[4:5]]

    project_step!(y, x, -d, ℓ, u)
    @test y == [0.0; -1.0; 0.0; 1.0; 0.0]

    ind = BitVector(undef, 5)
    active!(ind, x, ℓ, u)
    a = @allocated active!(ind, x, ℓ, u)
    @test a == 0
  end

  if CUDA.functional()
    CUDA.allowscalar() do
      @testset "Bounds with CuVector" begin
        T = Float64
        S = CuVector{T}
        x = S([0; 1; 2; 3; 4])
        ℓ = fill!(S(undef, 5), 0)
        u = fill!(S(undef, 5), 4)
        @test active(x, ℓ, u) == [1, 5]
    
        d = S([1; 1; 0; -1; -1])
        @test breakpoints(x, d, ℓ, u) == (4, 3.0, 4.0)
    
        z = S([-rand(2); x[3]; 4 .+ rand(2)])
        y = S(rand(5))
        project!(y, z, ℓ, u)
        @test y == [ℓ[1:2]; x[3]; u[4:5]]
    
        project_step!(y, x, -d, ℓ, u)
        @test y == [0.0; -1.0; 0.0; 1.0; 0.0]
    
        ind = BitVector(undef, 5)
        active!(ind, x, ℓ, u)
      end
    end
  end
end

test_auxiliary()
