export LbfgsB

function LbfgsB(nlp :: AbstractNLPModel;
                atol :: Float64=1.0e-5, rtol :: Float64=1.0e-6,
                max_f :: Int=0,
                max_fg :: Int=0,
                MaxIters :: Int = 500,
                verbose :: Bool=true,
                m :: Int=5)
#    println("LbfgsB: atol = ",atol," rtol = ",rtol," max_f = ",max_f," max_fg = ",max_fg)
    #obj(x) = obj(nlp,x)
    #grad!(x,g) = grad!(nlp,x,g)
    #grad(x) = grad(nlp,x)
    #x₀ = copy(nlp.meta.x0)
    n = nlp.meta.nvar
    x₀ = copy(nlp.meta.x0)

    g = Array(Float64,n)
    g₀ = Array(Float64,n)
    g = grad(nlp,x₀)
    grad!(nlp,x₀,g₀)

    function _ogFunc!(x, g::Array{Float64})
        grad!(nlp, x, g);
        return obj(nlp,x);
    end

    max_f == 0 && (max_f = max(min(100, 2 * n), 500))
    max_fg == 0 && (max_fg = 2 * max_f)
    
    #tol2 = max(atol, rtol * norm(g₀)) 
    #tolI = tol2 / (sqrt(n))
    tolI = max(atol, rtol * norm(g₀)) 
    verbose && println("LbfgsB: atol = ",atol," rtol = ",rtol," tolI = ",tolI, " norm(g₀) = ",norm(g₀))

    verblevel = verbose ? 1 : -1
    
    f, x, iterB, callB, status  = lbfgsb(_ogFunc!,
                                         #grad!,
                                         x₀, 
                                         m=m,
                                         maxiter = min(MaxIters,max_fg),
                                         iprint = verblevel,
                                         factr = 0.0,
                                         pgtol = tolI)

    tired = ! ((iterB < MaxIters) | (callB < max_fg))

    grad!(nlp,x,g)
    gNorm = norm(g,Inf)

    optimal = gNorm <= tolI

    #status =  "first-order stationary"
    #if ! optimal
    #    if tired
    #        status = "maximum number of evaluations"
    #    else status = "terminated sub-optimal"
    #    end
    #end

#    if optimal
#        status = string(status,"-- optimal")
#    else
#        status = string(status,"-- suboptimal")
#    end
    calls = [nlp.counters.neval_obj, nlp.counters.neval_grad, nlp.counters.neval_hess, nlp.counters.neval_hprod]  
    if tired status = :UserLimit 
    elseif optimal  status =  :Optimal
    else status =  :SubOptimal
    end

    return (x, f, gNorm, iterB, optimal, tired, status)
end
