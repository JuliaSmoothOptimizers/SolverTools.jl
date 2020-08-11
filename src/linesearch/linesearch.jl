export linesearch, linesearch!

include("linesearchoutput.jl")
include("armijo.jl")
include("armijo_wolfe.jl")

const ls_dict = Dict(
  :armijo => armijo!,
  :armijo_wolfe => armijo_wolfe!
)

"""
    lso = linesearch(ϕ, x, d; method=:armijo, kwargs...)
    lso = linesearch!(ϕ, x, d, xt; method=:armijo, kwargs...)
    lso = linesearch(nlp, x, d; method=:armijo, fx=obj(nlp, x), gx=grad(nlp, x), kwargs...)
    lso = linesearch!(nlp, x, d, xt; method=:armijo, fx=obj(nlp, x), gx=grad(nlp, x), kwargs...)

Perform a line search for unconstrained or bound-constrained problem `nlp` or merit model `ϕ` starting at `x` on direction `d`, storing the result in `xt`, for the in place version.
Returns a `LineSearchOutput`.
The `method` keyword argument can be any of the following:

- `:armijo`: Calls the simple Armijo line search, performing backtracking.
- `:armijo_wolfe`: Calls the Armijo-Wolfe line search, that tries to satisfy the Wolfe conditions in addition to the Armijo conditions.

When the input is `ϕ`, a merit model, the following keyword arguments are available for any method:
- `update_obj_at_x`: Whether to call `obj(ϕ, x; update=true)` to update `ϕ.fx` and `ϕ.cx` (default: `false`);
- `update_derivative_at_x`: Whether to call `derivative(ϕ, x, d; update=true)` to update `ϕ.gx` and `ϕ.Ad`.

For the `nlp` version, the problem must be unconstrained or bound-constrained (i.e., `nlp.meta.nequ = 0`).
In such case a `UncMerit` is created using the keyword arguments `fx` and `gx`. **Be warned** that `gx` will be overwritten.
"""
function linesearch end

function linesearch(nlp_or_ϕ, x, d; kwargs...)
  xt = copy(x)
  linesearch!(nlp_or_ϕ, x, d, xt; kwargs...)
end

function linesearch!(nlp :: AbstractNLPModel, x :: AbstractVector, args...; fx=obj(nlp, x), gx=grad(nlp, x), kwargs...)
  ϕ = UncMerit(nlp, fx=fx, gx=gx)
  linesearch!(ϕ, x, args...; kwargs...)
end

function linesearch!(ϕ :: AbstractMeritModel, x :: AbstractVector, args...; method :: Symbol=:armijo, kwargs...)
  SolverTools.ls_dict[method](ϕ, x, args...; kwargs...)
end