function LbfgsB(nlp :: AbstractNLPModel;
                #x₀ :: Array{Float64,1};
                atol :: Float64=1.0e-8, rtol :: Float64=1.0e-6,
                max_f :: Int=0,
                verbose :: Bool=true,
                mem :: Int=5)

    x₀ = copy(nlp.meta.x0)
    n = nlp.meta.nvar
    
    g = Array(Float64,n)
    g₀ = Array(Float64,n)
    g = grad(nlp,x₀)
    grad!(nlp,x₀,g₀)

    function _ogFunc!(x, g::Array{Float64})
        grad!(nlp, x, g);
        return obj(nlp,x);
    end

    max_f == 0 && (max_f = max(min(100, 2 * n), 500))
    max_fg = 2 * max_f
    
    #tol2 = max(atol, rtol * norm(g₀)) 
    #tolI = tol2 / (sqrt(n))
    tolI = max(atol, rtol * norm(g₀)) 
    verbose && println("LbfgsB: atol = ",atol," rtol = ",rtol," tolI = ",tolI, " norm(g₀) = ",norm(g₀))

    verblevel = verbose ? 1 : -1
    
    f, x, iterB, callB, status  = lbfgsb(_ogFunc!,
                                         #grad!,
                                         x₀, 
                                         m=mem,
                                         maxiter = max_f,
                                         iprint = verblevel,
                                         factr = 0.0,
                                         pgtol = tolI)

    tired = ! (callB < max_fg)

    grad!(nlp,x,g)
    gNorm = norm(g,Inf)

    optimal = gNorm <= tolI

    #calls = [nlp.counters.neval_obj, nlp.counters.neval_grad, nlp.counters.neval_hess, nlp.counters.neval_hprod]  
    if tired status = :UserLimit 
    elseif optimal  status =  :Optimal
    else status =  :SubOptimal
    end

    return (x, f, gNorm, iterB, optimal, tired, status)
end
