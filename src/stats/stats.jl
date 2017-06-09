export ExecutionStats, statshead, statsline

const STATUS = Dict(:unknown => "unknown",
                    :first_order => "first-order stationary",
                    :max_eval => "maximum number of function evaluations",
                    :max_time => "maximum elapsed time",
                    :neg_pred => "negative predicted reduction",
                    :exception => "unhandled exception"
                   )

type ExecutionStats
  status :: Symbol
  solved :: Bool
  tired :: Bool
  stalled :: Bool
  solution :: Vector # x
  obj :: Float64 # f(x)
  opt_norm :: Float64 # ‖∇f(x)‖ for unc, ‖P[x - ∇f(x)] - x‖ for bnd, etc.
  feas_norm :: Float64 # ‖c(x)‖
  iter :: Int
  eval :: NLPModels.Counters
  elapsed_time :: Float64
  solver_specific :: Dict{Symbol,Any}
end

function ExecutionStats{T}(status :: Symbol; solved :: Bool=false, tired :: Bool=false, stalled :: Bool=false,
                        x :: Vector=Float64[], f :: Float64=Inf, normg :: Float64=Inf,
                        c :: Float64=0.0, iter :: Int=-1, t :: Float64=Inf,
                        eval :: NLPModels.Counters=Counters(),
                        solver_specific :: Dict{Symbol,T}=Dict{Symbol,Any}(),
                        kwargs...)
  if !(status in keys(STATUS))
    s = join(keys(STATUS, ", "))
    error("$status is not a valid status. Use one of the following: $s")
  end
  for (k,v) in kwargs
    if k == :solution || k == :sol
      x = v
    elseif k == :objective || k == :obj
      f = v
    elseif k == :opt_norm || k == :opt
      normg = v
    elseif k == :feas_norm || k == :feas
      c = v
    elseif k == :elapsed_time || k == :time
      t = v
    end
  end
  return ExecutionStats(status, solved, tired, stalled, x, f, normg, c, iter, deepcopy(eval), t, solver_specific)
end

import Base.show

function show(io :: IO, stats :: ExecutionStats)
  # TODO: Show evaluations
  println("Execution stats")
  println("  status: $(getStatus(stats))")
  println("  solved: $(stats.solved)")
  println("  objective value: $(stats.obj)")
  println("  optimality measure: $(stats.opt_norm)")
  println("  feasibility measure: $(stats.feas_norm)")
  println("  iterations: $(stats.iter)")
  println("  elapsed time: $(stats.elapsed_time)")
  if length(stats.solver_specific) > 0
    println("  solver specifics:")
    for (k,v) in stats.solver_specific
      println("    $k: $v")
    end
  end
end

const headsym = Dict(:solved  => "  Solved",
                     :tired   => "   Tired",
                     :stalled => " Stalled",
                     :status  => "  Status",
                     :iter         => "   Iter",
                     :neval_obj    => "   #obj",
                     :neval_grad   => "  #grad",
                     :neval_cons   => "  #cons",
                     :neval_jcon   => "  #jcon",
                     :neval_jgrad  => " #jgrad",
                     :neval_jac    => "   #jac",
                     :neval_jprod  => " #jprod",
                     :neval_jtprod => "#jtprod",
                     :neval_hess   => "  #hess",
                     :neval_hprod  => " #hprod",
                     :neval_jhprod => "#jhprod",
                     :obj          => "              f",
                     :opt_norm     => "           ‖∇f‖",
                     :feas_norm    => "            ‖c‖",
                     :elapsed_time => "  Elaspsed time")

function statsgetfield(stats :: ExecutionStats, name :: Symbol)
  t = Int
  if name == :status
    v = getStatus(stats)
    t = String
  elseif name in fieldnames(ExecutionStats)
    v = getfield(stats, name)
    t = fieldtype(ExecutionStats, name)
  elseif name in fieldnames(NLPModels.Counters)
    v = getfield(stats.eval, name)
  else
    throw("No such field '$name' in ExecutionStats")
  end
  if t == Int
    @sprintf("%7d", v)
  elseif t == Float64
    @sprintf("%15.8e", v)
  else
    @sprintf("%8s", v)
  end
end

function statshead(line :: Array{Symbol})
  return join([headsym[x] for x in line], "  ")
end

function statsline(stats :: ExecutionStats, line :: Array{Symbol})
  return join([statsgetfield(stats, x) for x in line], "  ")
end

function getStatus(stats :: ExecutionStats)
  return STATUS[stats.status]
end
