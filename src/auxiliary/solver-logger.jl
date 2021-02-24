using Formatting

export SolverLogger, header

mutable struct SolverLogger
  keys       :: Vector{Symbol}
  types      :: Vector{DataType}
  formatters :: Vector
end

function SolverLogger(col_keys::Vector{Symbol}, col_types::Vector{DataType})
  formatters = [
    begin
      fmt = get(formats, t, nothing)
      if isnothing(fmt)
        fmt = get(formats, supertype(t), "%10s")
      end
      len = match(r"\%([0-9]*)", fmt)[1]
      (generate_formatter(fmt), generate_formatter("%$(len)s"))
    end for t in col_types
  ]
  SolverLogger(col_keys, col_types, formatters)
end

function header(logger::SolverLogger)
  join([logger.formatters[i][2](string(x)) for (i,x) in enumerate(logger.keys)], " ")
end

function log(logger::SolverLogger, args...)
  if length(args) != length(logger.keys)
    throw(ArgumentError("SolverLogger needs $(length(logger.keys)) arguments"))
  end
  join([
    ismissing(x) ? logger.formatters[i][2]("-") : logger.formatters[i][1](x)
    for (i,x) in enumerate(args)
  ], " ")
end

(logger::SolverLogger)(args...) = log(logger, args...)