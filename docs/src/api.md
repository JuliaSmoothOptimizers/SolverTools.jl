# API

```@contents
Pages = ["api.md"]
```
## Auxiliary

```@docs
active
breakpoints
compute_Hs_slope_qs!
log_header
log_row
project!
project_step!
```

## Benchmarking

```@docs
bmark_solvers
solve_problems
```

## Line-Search

```@docs
LineModel
obj
grad
grad!
hess
redirect!
```

## Stats

```@docs
GenericExecutionStats
```

## Trust-Region

```@docs
TrustRegionException
SolverTools.AbstractTrustRegion
aredpred
acceptable
reset!
get_property
set_property!
update!
TrustRegion
TRONTrustRegion
```
