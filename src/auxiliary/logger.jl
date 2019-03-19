export log_header, log_row

const formats = Dict{DataType, String}(Signed => "%-5d",
                                       AbstractFloat => "%-8.1e",
                                       AbstractString => "%-15s",
                                       Symbol => "%-15s",
                                       Missing => "%-15s"
                                      )

const default_headers = Dict{Symbol, String}(:name => "Name",
                                             :elapsed_time => "Time",
                                             :objective => "f(x)",
                                             :dual_feas => "Dual",
                                             :primal_feas => "Primal")

for (typ, fmt) in formats
  hdr_fmt_foo = Symbol("header_formatter_$typ")
  len = match(r"\%-([0-9]*)", fmt)[1]
  fmt2 = "%-$(len)s"

  @eval begin
    row_formatter(x :: $typ) = @sprintf($fmt, x)

    $hdr_fmt_foo(x) = @sprintf($fmt2, x)
    header_formatter(x :: Union{Symbol,String}, :: Type{<:$typ}) = $hdr_fmt_foo(x)
  end
end

"""
    log_header(colnames, coltypes)

Creates a header using the names in `colnames` formatted according to the types in `coltypes`.
Uses internal formatting specification given by `SolverTools.formats` and default header
translation given by `SolverTools.default_headers`.

Input:
- `colnames::Vector{Symbol}`: Column names.
- `coltypes::Vector{DataType}`: Column types.

Keyword arguments:
- `hdr_override::Dict{Symbol,String}`: Overrides the default headers.
- `colsep::Int`: Number of spaces between columns (Default: 2)

See also [`log_row`](@ref).
"""
function log_header(colnames :: AbstractVector{Symbol}, coltypes :: AbstractVector{DataType};
                    hdr_override :: Dict{Symbol,String} = Dict{Symbol,String}(),
                    colsep :: Int = 2,
                   )
  out = ""
  for (name, typ) in zip(colnames, coltypes)
    x = if haskey(hdr_override, name)
      hdr_override[name]
    elseif haskey(default_headers, name)
      default_headers[name]
    else
      string(name)
    end
    out *= header_formatter(x, typ) * " "^colsep
  end
  return out
end

"""
    log_row(vals)

Creates a table row from the values on `vals` according to their types. Pass the names
and types of `vals` to [`log_header`](@ref) for a logging table. Uses internal formatting
specification given by `SolverTools.formats`.

Keyword arguments:
- `colsep::Int`: Number of spaces between columns (Default: 2)
"""
function log_row(vals; colsep :: Int = 2)
  string_cols = (row_formatter(val) for val in vals)
  return join(string_cols, " "^colsep)
end
