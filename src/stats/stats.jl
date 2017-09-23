export ExecutionStats, statshead, statsline

const STATUS = Dict(:unknown => "unknown",
                    :first_order => "first-order stationary",
                    :max_eval => "maximum number of function evaluations",
                    :max_time => "maximum elapsed time",
                    :max_iter => "maximum iteration",
                    :neg_pred => "negative predicted reduction",
                    :unbounded => "objective function may be unbounded from below",
                    :exception => "unhandled exception",
                    :stalled => "stalled"
                   )

type ExecutionStats
  status :: Symbol
  solution :: Vector # x
  dual_vector :: Vector # λ
  obj :: Float64 # f(x)
  dual_feas :: Float64 # ‖∇f(x)‖ for unc, ‖P[x - ∇f(x)] - x‖ for bnd, etc.
  primal_feas :: Float64 # ‖c(x)‖
  iter :: Int
  eval :: NLPModels.Counters
  elapsed_time :: Float64
  solver_specific :: Dict{Symbol,Any}
end

function ExecutionStats{T}(status :: Symbol;
                           x :: Vector=Float64[], λ :: Vector=Float64[], f :: Float64=Inf,
                           normg :: Float64=Inf, c :: Float64=0.0, iter :: Int=-1,
                           t :: Float64=Inf, eval :: NLPModels.Counters=Counters(),
                           solver_specific :: Dict{Symbol,T}=Dict{Symbol,Any}(),
                           kwargs...)
  if !(status in keys(STATUS))
    s = join(keys(STATUS, ", "))
    error("$status is not a valid status. Use one of the following: $s")
  end
  for (k,v) in kwargs
    if k == :solution || k == :sol
      x = v
    elseif k == :dual_vector
      λ = v
    elseif k == :objective || k == :obj
      f = v
    elseif k == :dual_feas || k == :opt
      normg = v
    elseif k == :primal_feas || k == :feas
      c = v
    elseif k == :elapsed_time || k == :time
      t = v
    else
      throw(UndefVarError(k))
    end
  end
  return ExecutionStats(status, x, λ, f, normg, c, iter, deepcopy(eval), t, solver_specific)
end

import Base.show, Base.print, Base.println

function show(io :: IO, stats :: ExecutionStats)
  show(io, "Execution stats: $(getStatus(stats))")
end

# TODO: Expose NLPModels dsp in nlp_types.jl function print
function disp_vector(io :: IO, x :: Vector)
  if length(x) == 0
    @printf(io, "∅")
  elseif length(x) <= 5
    Base.show_delim_array(io, x, "", " ", "", false)
  else
    Base.show_delim_array(io, x[1:4], "", " ", "", false)
    @printf(io, " ⋯ %s", x[end])
  end
end

function print(io :: IO, stats :: ExecutionStats; showvec :: Function=disp_vector)
  # TODO: Show evaluations
  @printf(io, "Execution stats\n")
  @printf(io, "  status: "); show(io, getStatus(stats)); @printf(io, "\n")
  @printf(io, "  objective value: "); show(io, stats.obj); @printf(io, "\n")
  @printf(io, "  dual feasibility: "); show(io, stats.dual_feas); @printf(io, "\n")
  @printf(io, "  primal feasibility: "); show(io, stats.primal_feas); @printf(io, "\n")
  @printf(io, "  solution: "); showvec(io, stats.solution); @printf(io, "\n")
  @printf(io, "  iterations: "); show(io, stats.iter); @printf(io, "\n")
  @printf(io, "  elapsed time: "); show(io, stats.elapsed_time); @printf(io, "\n")
  if length(stats.solver_specific) > 0
    @printf(io, "  solver specifics:\n")
    for (k,v) in stats.solver_specific
      @printf(io, "    %s: ", k)
      if isa(v, Vector)
        showvec(io, v)
      else
        show(io, v)
      end
      @printf(io, "\n")
    end
  end
end
print(stats :: ExecutionStats; showvec :: Function=disp_vector) = print(STDOUT, stats, showvec=showvec)
println(io :: IO, stats :: ExecutionStats; showvec :: Function=disp_vector) = print(io, stats, showvec=showvec)
println(stats :: ExecutionStats; showvec :: Function=disp_vector) = print(STDOUT, stats, showvec=showvec)

const headsym = Dict(:status  => "  Status",
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
                     :dual_feas    => "           ‖∇f‖",
                     :primal_feas  => "            ‖c‖",
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
