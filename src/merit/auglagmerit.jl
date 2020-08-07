export AugLagMerit

@doc raw"""
    AugLagMerit(nlp, η; kwargs...)

Creates an augmented Lagrangian merit function for the equality constrained problem
```math
\min f(x) \quad \text{s.to} \quad c(x) = 0
```
defined by
```math
\phi(x, yₖ; η) = f(x) - yₖᵀc(x) + η ½\|c(x)\|²
```

In addition to the keyword arguments declared in [`AbstractMeritModel`](@ref), an `AugLagMerit` also
accepts the argument `y`.
"""
mutable struct AugLagMerit{M <: AbstractNLPModel, T <: Real, V <: AbstractVector} <: AbstractMeritModel
    nlp :: M
    η :: T
    fx :: T
    gx :: V
    cx :: V
    Ad :: V
    y :: V
end

function AugLagMerit(
    nlp :: M,
    η :: T;
    fx :: T = T(Inf),
    gx :: V = fill(T(Inf), nlp.meta.nvar),
    cx :: V = fill(T(Inf), nlp.meta.ncon),
    Ad :: V = fill(T(Inf), nlp.meta.ncon),
    y :: V = fill(T(Inf), nlp.meta.ncon)
) where {M <: AbstractNLPModel, T <: Real, V <: AbstractVector{<: T}}
    AugLagMerit{M,T,V}(nlp, η, fx, gx, cx, Ad, y)
end

function NLPModels.obj(merit :: AugLagMerit, x :: AbstractVector; update :: Bool = true)
    if update
        merit.fx = obj(merit.nlp, x)
        merit.nlp.meta.ncon > 0 && cons!(merit.nlp, x, merit.cx)
    end
    return merit.fx - dot(merit.y, merit.cx) + merit.η * dot(merit.cx, merit.cx) / 2
end

function derivative(merit :: AugLagMerit, x :: AbstractVector, d :: AbstractVector; update :: Bool = true)
    if update
        grad!(merit.nlp, x, merit.gx)
        merit.nlp.meta.ncon > 0 && jprod!(merit.nlp, x, d, merit.Ad)
    end
    if merit.nlp.meta.ncon == 0
        return dot(merit.gx, d)
    else
        return dot(merit.gx, d) - dot(merit.y, merit.Ad) + merit.η * dot(merit.cx, merit.Ad)
    end
end