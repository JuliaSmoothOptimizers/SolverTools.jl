export save_stats, load_stats, count_unique, quick_summary


"""
    save_stats(stats, filename; kwargs...)

Write the benchmark statistics `stats` to a file named `filename`.

#### Arguments

* `stats::Dict{Symbol,DataFrame}`: benchmark statistics such as returned by `bmark_solvers()`
* `filename::AbstractString`: the output file name.

#### Keyword arguments

* `force::Bool=false`: whether to overwrite `filename` if it already exists
* `key::String="stats"`: the key under which the data can be read from `filename` later.

#### Return value

This method returns an error if `filename` exists and `force==false`.
On success, it returns the value of `jldopen(filename, "w")`.
"""
function save_stats(stats::Dict{Symbol,DataFrame}, filename::AbstractString; force::Bool=false, key::String="stats")
  isfile(filename) && !force && error("$filename already exists; use `force=true` to overwrite")
  jldopen(filename, "w") do file
    file[key] = stats
  end
end

"""
    stats = load_stats(filename; kwargs...)

#### Arguments

* `filename::AbstractString`: the input file name.

#### Keyword arguments

* `key::String="stats"`: the key under which the data can be read in `filename`.
  The key should be the same as the one used when `save_stats()` was called.

#### Return value

A `Dict{Symbol,DataFrame}` containing the statistics stored in file `filename`.
The user should `import DataFrames` before calling `load_stats()`.
"""
function load_stats(filename::AbstractString; key::String="stats")
  jldopen(filename) do file
    file[key]
  end
end

@doc raw"""
    vals = count_unique(X)

Count the number of occurrences of each value in `X`.

#### Arguments

* `X`: an iterable.

#### Return value

A `Dict{eltype(X),Int}` whose keys are the unique elements in `X` and
values are their number of occurrences.

Example: the snippet

    stats = load_stats("mystats.jld2")
    for solver ∈ keys(stats)
      @info "$solver statuses" count_unique(stats[solver].status)
    end

displays the number of occurrences of each final status for each solver in `stats`.
"""
function count_unique(X)
  vals = Dict{eltype(X),Int}()
  for x ∈ X
    vals[x] = x ∈ keys(vals) ? (vals[x] + 1) : 1
  end
  vals
end

"""
    statuses, avgs = quick_summary(stats; kwargs...)

Call `count_unique()` and compute a few average measures for each solver in `stats`.

#### Arguments

* `stats::Dict{Symbol,DataFrame}`: benchmark statistics such as returned by `bmark_solvers()`.

#### Keyword arguments

* `cols::Vector{Symbol}`: symbols indicating `DataFrame` columns in solver statistics for which we
  compute averages. Default: `[:iter, :neval_obj, :neval_grad, :neval_hess, :neval_hprod, :elapsed_time]`.

#### Return value

* `statuses::Dict{Symbol,Dict{Symbol,Int}}`: a dictionary of number of occurrences of each final
  status for each solver in `stats`. Each value in this dictionary is returned by `count_unique()`
* `avgs::Dict{Symbol,Dict{Symbol,Float64}}`: a dictionary that contains averages of performance measures
  across all problems for each solver. Each `avgs[solver]` is a `Dict{Symbol,Float64}` where the measures
  are those given in the keyword argument `cols` and values are averages of those measures across all problems.

Example: the snippet

    statuses, avgs = quick_summary(stats)
    for solver ∈ keys(stats)
      @info "statistics for" solver statuses[solver] avgs[solver]
    end

displays quick summary and averages for each solver.
"""
function quick_summary(stats::Dict{Symbol,DataFrame};
                       cols::Vector{Symbol}=[:iter, :neval_obj, :neval_grad, :neval_hess, :neval_hprod, :elapsed_time])
  nproblems = size(stats[first(keys(stats))], 1)
  statuses = Dict{Symbol,Dict{Symbol,Int}}()
  avgs = Dict{Symbol,Dict{Symbol,Float64}}()
  for solver ∈ keys(stats)
    statuses[solver] = count_unique(stats[solver].status)
    avgs[solver] = Dict{Symbol,Float64}()
    for col ∈ cols
      avgs[solver][col] = sum(stats[solver][!, col]) / nproblems
    end
  end
  statuses, avgs
end

