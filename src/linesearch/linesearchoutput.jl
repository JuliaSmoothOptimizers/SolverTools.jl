export LineSearchOutput

# TODO: Generalize AbstractExecutionStats?

"""
    LineSearchOutput

Structure to store the output of a line search algorithm.
The following variables are stored:
- `t`: The final step length;
- `xt`: The final iterate ``xt = x + t d``;
- `ϕt`: The final merit function value, i.e., `ϕ(xt)`;
- `good_grad`: Whether the gradient was computed at `xt`;
- `gt`: If `good_grad` is `true`, the gradient at `xt`;
- `specific`: A NamedTuple object storing specific method information. Check the specifc method for this information.
"""
struct LineSearchOutput{T <: Real, V <: AbstractVector{<: T}}
  t :: T
  xt :: V
  ϕt :: T
  good_grad :: Bool
  gt :: V
  specific :: NamedTuple
end

function LineSearchOutput(
  t :: T,
  xt :: V,
  ϕt :: T;
  good_grad :: Bool=false,
  gt :: V=T[],
  specific :: NamedTuple = NamedTuple()
) where {T <: Real, V <: AbstractVector{<: T}}
  LineSearchOutput(t, xt, ϕt, good_grad, gt, specific)
end