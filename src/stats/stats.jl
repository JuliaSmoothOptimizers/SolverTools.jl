export AbstractExecutionStats, GenericExecutionStats,
       statshead, statsline

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

abstract type AbstractExecutionStats end

type GenericExecutionStats <: AbstractExecutionStats
  status :: Symbol
  solution :: Vector # x
  objective :: Float64 # f(x)
  dual_feas :: Float64 # ‖∇f(x)‖₂ for unc, ‖P[x - ∇f(x)] - x‖₂ for bnd, etc.
  iter :: Int
  counters :: NLPModels.Counters
  elapsed_time :: Float64
  solver_specific :: Dict{Symbol,Any}
end

type NullNLPModel <: AbstractNLPModel end

function GenericExecutionStats{T}(status :: Symbol;
                           solution :: Vector=Float64[],
                           objective :: Float64=Inf,
                           dual_feas :: Float64=Inf,
                           iter :: Int=-1,
                           nlp :: AbstractNLPModel=NullNLPModel(),
                           elapsed_time :: Float64=Inf,
                           solver_specific :: Dict{Symbol,T}=Dict{Symbol,Any}())
  if !(status in keys(STATUS))
    s = join(keys(STATUS, ", "))
    error("$status is not a valid status. Use one of the following: $s")
  end
  c = Counters()
  if !isa(nlp, NullNLPModel)
    for counter in fieldnames(Counters)
      setfield!(c, counter, eval(parse("$counter"))(nlp))
    end
  end
  return GenericExecutionStats(status, solution, objective, dual_feas, iter,
                        c, elapsed_time, solver_specific)
end

import Base.show, Base.print, Base.println

function show(io :: IO, stats :: AbstractExecutionStats)
  show(io, "Execution stats: $(getStatus(stats))")
end

# TODO: Expose NLPModels dsp in nlp_types.jl function print
function disp_vector(io :: IO, x :: Vector)
  if length(x) == 0
    @printf(io, "∅")
  elseif length(x) <= 5
    Base.show_delim_array(io, x, "[", " ", "]", false)
  else
    Base.show_delim_array(io, x[1:4], "[", " ", "", false)
    @printf(io, " ⋯ %s]", x[end])
  end
end

function print(io :: IO, stats :: GenericExecutionStats; showvec :: Function=disp_vector)
  # TODO: Show evaluations
  @printf(io, "Generic Execution stats\n")
  @printf(io, "  status: "); show(io, getStatus(stats)); @printf(io, "\n")
  @printf(io, "  objective value: "); show(io, stats.objective); @printf(io, "\n")
  @printf(io, "  dual feasibility: "); show(io, stats.dual_feas); @printf(io, "\n")
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
print(stats :: GenericExecutionStats; showvec :: Function=disp_vector) =
    print(STDOUT, stats, showvec=showvec)
println(io :: IO, stats :: GenericExecutionStats; showvec ::
        Function=disp_vector) = print(io, stats, showvec=showvec)
println(stats :: GenericExecutionStats; showvec :: Function=disp_vector) =
    print(STDOUT, stats, showvec=showvec)

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
                     :objective    => "              f",
                     :dual_feas    => "           ‖∇f‖",
                     :elapsed_time => "  Elaspsed time")

function statsgetfield(stats :: AbstractExecutionStats, name :: Symbol)
  t = Int
  if name == :status
    v = getStatus(stats)
    t = String
  elseif name in fieldnames(NLPModels.Counters)
    v = getfield(stats.counters, name)
  elseif name in fieldnames(AbstractExecutionStats)
    v = getfield(stats, name)
    t = fieldtype(AbstractExecutionStats, name)
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

function statsline(stats :: AbstractExecutionStats, line :: Array{Symbol})
  return join([statsgetfield(stats, x) for x in line], "  ")
end

function getStatus(stats :: AbstractExecutionStats)
  return STATUS[stats.status]
end
