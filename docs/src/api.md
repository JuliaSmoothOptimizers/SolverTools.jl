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

## Merit

See also [`obj`](@ref).

```@docs
AbstractMeritModel
derivative
L1Merit
AugLagMerit
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
