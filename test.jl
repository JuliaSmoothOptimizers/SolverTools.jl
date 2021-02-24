using SolverTools

slug = SolverLogger(
  [:f, :∇f, :iter, :status],
  [Float64, Float64, Int, Symbol]
)

@info header(slug)
@info SolverTools.log(slug, 1, 1, 1, missing)
@info SolverTools.log(slug, 1, 1, 2.0, missing)
@info SolverTools.log(slug, 1, missing, 2, "Step failure")
@info slug(1, 2, 3, "Step failure")