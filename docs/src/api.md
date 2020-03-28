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

### Main Benchmarking Functions

```@docs
bmark_solvers
solve_problems
```

### Utilities

```@docs
save_stats
load_stats
count_unique
quick_summary
```

## Line-Search

```@docs
LineModel
obj
grad
grad!
hess
redirect!
armijo_wolfe
```

## Stats

```@docs
GenericExecutionStats
show_statuses
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
