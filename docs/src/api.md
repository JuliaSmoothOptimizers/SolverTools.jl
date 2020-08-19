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
linesearch
LineSearchOutput
armijo!
armijo_wolfe!
```

## Merit

```@docs
AbstractMeritModel
obj
dualobj
primalobj
derivative
L1Merit
AugLagMerit
UncMerit
```

## Stats

```@docs
GenericExecutionStats
show_statuses
```

## Trust-Region

```@docs
TrustRegionException
trust_region
basic_trust_region!
tron_trust_region!
```
