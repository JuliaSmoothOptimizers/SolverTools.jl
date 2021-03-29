# API

```@contents
Pages = ["api.md"]
```
## Auxiliary

```@docs
active
breakpoints
compute_Hs_slope_qs!
project!
project_step!
```

## Line-Search

```@docs
LineModel
obj
grad
grad!
objgrad
objgrad!
hess
redirect!
armijo_wolfe
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
