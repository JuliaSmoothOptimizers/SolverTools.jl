# A trust-region type and basic utility functions.
export TrustRegionException
export trust_region, trust_region!
# import NLPModels: reset!
# export TrustRegionException, acceptable, aredpred, get_property, ratio, ratio!, reset!, set_property!, update!

include("trust-region-output.jl")
include("basic-trust-region.jl")
include("tron-trust-region.jl")

const tr_dict = Dict(
  :basic => basic_trust_region!,
  :tron => tron_trust_region!,
)

"Exception type raised in case of error."
mutable struct TrustRegionException <: Exception
  msg  :: String
end

"""
    tro = trust_region(ϕ, x, d, Δq, Ad, Δ; method=:basic, penalty_update=:basic, kwargs...)
    tro = trust_region!(ϕ, x, d, xt, Δq, Ad, Δ; method=:basic, penalty_update=:basic, kwargs...)
    tro = trust_region(nlp, x, d, Δq, Δ; method=:basic, fx=obj(nlp, x), gx=grad(nlp, x), kwargs...)
    tro = trust_region!(nlp, x, d, xt, Δq, Δ; method=:basic, fx=obj(nlp, x), gx=grad(nlp, x), kwargs...)

Update `xt` and `Δ` using a trust region strategy according to the `method` keyword.
The output is a `TrustRegionOutput`.

The actual reduction of the model is given by
```
ared = ϕx - ϕt,
```
where `ϕx = obj(nlp, x; update=update_obj_at_x)` and `ϕt = obj(nlp, xt)`. The predicted reduction is
given by
```
pred = ϕx - (ϕx + Δq) - ϕ.η P(ϕ.cx + Ad),
```
where `P` is the penalty function used by `ϕ` and `Ad` is the Jacobian times `d`.

The following keyword argument are available:
- `update_obj_at_x`: Whether to update `ϕ`'s objective at `x` (default: `false`);
- `update_obj_at_xt`: Whether to update `ϕ`s objective at `xt` or to use the `ft` and `ct` keywords (default: `true`);
- `ft`: If `update_obj_at_xt=false`, then used by the merit function as `obj(nlp, xt)` (default: `Inf`);
- `ct`: If `update_obj_at_xt=false`, then used by the merit function as `cons(nlp, xt)` (default: `[]`);
- `penalty_update`: Strategy to update the parameter `ϕ.η`. (default: `:basic`);
- `max_radius`: Maximum value for `Δ` (default 1/√ϵ(T));
- `ηacc`: threshold for successful step (default: 1.0e-4);
- `ηinc`: threshold for great step (default: 0.95),
- `σdec`: factor used in the radius decrease heuristics (default: 1/3);
- `σinc`: factor used in the radius increase heuristics (default: 3/2);

Some methods have additional keyword arguments.

For the `nlp` version, the problem must be unconstrained or bound-constrained (i.e., `nlp.meta.nequ
= 0`). In such case a `UncMerit` is created using the keyword argument `fx`.
"""
function trust_region end

function trust_region(nlp_or_ϕ, x, d, args...; kwargs...)
  xt = copy(x)
  trust_region!(nlp_or_ϕ, x, d, xt, args...; kwargs...)
end

function trust_region!(nlp :: AbstractNLPModel, x :: AbstractVector, d :: AbstractVector, xt :: AbstractVector, Δq :: Real, Δ :: Real; fx=obj(nlp, x), kwargs...)
  ϕ = UncMerit(nlp, fx=fx)
  trust_region!(ϕ, x, d, xt, Δq, eltype(x)[], Δ; kwargs...)
end

function trust_region!(ϕ :: UncMerit, x :: AbstractVector, d :: AbstractVector, xt :: AbstractVector, Δq :: Real, Δ :: Real; kwargs...)
  trust_region!(ϕ, x, d, xt, Δq, eltype(x)[], Δ; kwargs...)
end

function trust_region!(ϕ :: AbstractMeritModel, x :: AbstractVector, d :: AbstractVector, xt :: AbstractVector, Δq :: Real, Ad :: AbstractVector, Δ :: Real; method :: Symbol=:basic, kwargs...)
  SolverTools.tr_dict[method](ϕ, x, d, xt, Δq, Ad, Δ; kwargs...)
end