export L1Merit

@doc raw"""
    L1Merit(nlp, η; kwargs...)

Creates a ℓ₁ merit function for the equality constrained problem
```math
\min f(x) \quad \text{s.to} \quad c(x) = 0
```
defined by
```math
\phi_1(x; η) = f(x) + η\|c(x)\|₁
```
"""
mutable struct L1Merit{M <: AbstractNLPModel, T <: Real, V <: AbstractVector} <: AbstractMeritModel
    nlp :: M
    η :: T
    fx :: T
    gx :: V
    cx :: V
    Ad :: V
end

function L1Merit(
    nlp :: M,
    η :: T;
    fx :: T = T(Inf),
    gx :: V = fill(T(Inf), nlp.meta.nvar),
    cx :: V = fill(T(Inf), nlp.meta.ncon),
    Ad :: V = fill(T(Inf), nlp.meta.ncon)
) where {M <: AbstractNLPModel, T <: Real, V <: AbstractVector{<: T}}
    L1Merit{M,T,V}(nlp, η, fx, gx, cx, Ad)
end

function NLPModels.obj(merit :: L1Merit, x :: AbstractVector; update :: Bool = true)
    if update
        merit.fx = obj(merit.nlp, x)
        merit.nlp.meta.ncon > 0 && cons!(merit.nlp, x, merit.cx)
    end
    return merit.fx + merit.η * norm(merit.cx, 1)
end

function derivative(merit :: L1Merit, x :: AbstractVector, d :: AbstractVector; update :: Bool = true)
    if update
        grad!(merit.nlp, x, merit.gx)
        merit.nlp.meta.ncon > 0 && jprod!(merit.nlp, x, d, merit.Ad)
    end
    if merit.nlp.meta.ncon == 0
        return dot(merit.gx, d)
    else
        return dot(merit.gx, d) + merit.η * sum(
            if ci > 0
                Adi
            elseif ci < 0
                -Adi
            else
                abs(Adi)
            end
            for (ci, Adi) in zip(merit.cx, merit.Ad)
        )
    end
end