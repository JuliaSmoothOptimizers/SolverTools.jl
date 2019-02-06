export latex_tabular_results, safe_latex_Signed, safe_latex_AbstractString,
       safe_latex_AbstractFloat, safe_latex_Symbol

const formats = Dict{DataType, String}(Signed => "%5d",
                                       AbstractFloat => "%7.1e",
                                       AbstractString => "%s",
                                       Symbol => "%s")

"""`safe_latex_Signed(s)`

For signed integers. Encloses `s` in `\\(` and `\\)`.
"""
safe_latex_Signed(s :: AbstractString) = "\\(" * s * "\\)"

"""`safe_latex_AbstractString(s)`

For strings. Replaces `_` by `\\_`.
"""
safe_latex_AbstractString(s :: AbstractString) = replace(s, "_" => "\\_")

"""`safe_latex_AbstractFloat(s)`

For floats. Bypasses `Inf` and `NaN`. Enclose both the mantissa and the
exponent in `\\(` and `\\)`.
"""
function safe_latex_AbstractFloat(s :: AbstractString)
  strip(s) == "Inf" && return "\\(\\infty\\)"
  strip(s) == "NaN" && return s
  mantissa, exponent = split(s, 'e')
  "\\(" * mantissa * "\\)e\\(" * exponent * "\\)"
end

"""`safe_latex_Symbol(s)`

For symbols. Same as strings.
"""
safe_latex_Symbol = safe_latex_AbstractString

for (typ, fmt) in formats
  safe = Symbol("safe_latex_$typ")
  @eval begin
    LTXformat(x :: $typ) = @sprintf($fmt, x) |> $safe
  end
end
LTXformat(x :: Missing) = "NA"

@doc """
    LTXformat(x)

Formats `x` according to its type. For types `Signed`, `AbstractFloat`,
`AbstractString` and `Symbol`, it uses a predefined formatting string passed to
`@sprintf` and then the corresponding `safe_latex_<type>` function.

For type `Missing`, it returns "NA".
"""
LTXformat

const headers = Dict{Symbol,String}(:name => "Name",
                                    :objective => "\\(f(x)\\)",
                                    :dual_feas => "\\(\\|\\nabla L\\|\\)")

"""
    latex_tabular_results(io, df, kwargs...)

Creates a latex longtable using LaTeXTabulars of a dataframe of results, formatting
the output for a publication-ready table.

Inputs:
- `io::IO`: where to send the table, e.g.:

      open("file.tex", "w") do io
        latex_tabular_results(io, df)
      end

- `df::DataFrame`: Dataframe as output by `solve_problems` or the dicts returned by
  `bmark_solvers`.

Keyword arguments:
- `cols::Array{Symbol}`: Which columns of the `df`. Defaults to using all columns;
- `ignore_missing_cols::Bool`: If `true`, filters out the columns in `cols` that don't
  exist in the data frame. Useful when creating tables for solvers in a loop where one
  solver has a column the other doesn't. If `false`, throws `BoundsError` in that
  situation.
- `fmt_override::Dict{Symbol,Function}`: Overrides format for a specific columns, such as

      fmt_override=Dict(:name => x->@sprintf("\\textbf{%s}", x) |> safe_latex_AbstractString)`
- `hdr_override::Dict{Symbol,String}`: Overrides header names, such as
  `hdr_override=Dict(:dual_feas => "\\\\(\\\\nabla f(x)\\\\)")`, mind the LaTeX naming.

We recommend using the `safe_latex_foo` functions when overriding formats, unless
you're sure you don't need them.
"""
function latex_tabular_results(io :: IO, df :: DataFrame;
                               cols :: Array{Symbol,1} = names(df),
                               ignore_missing_cols :: Bool = false,
                               fmt_override :: Dict{Symbol,Function} = Dict{Symbol,Function}(),
                               hdr_override :: Dict{Symbol,String} = Dict{Symbol,String}(),
                              )
  if ignore_missing_cols
    cols = filter(c->haskey(df, c), cols)
  elseif !all(haskey(df, c) for c in cols)
    missing_cols = setdiff(cols, names(df))
    @error("There are no columns `" * join(missing_cols, ", ") * "` in dataframe")
    throw(BoundsError)
  end
  string_cols = [map(haskey(fmt_override, col) ? fmt_override[col] : LTXformat, df[col]) for col in cols]
  table = hcat(string_cols...)

  headers_copy = copy(headers)
  for (h,s) in hdr_override
    headers_copy[h] = s
  end
  header = [haskey(headers_copy, c) ? headers_copy[c] : string(c) for c in cols]

  latex_tabular(io, LongTable("l" * "r"^(length(cols)-1), header),
                [table, Rule()])
  nothing
end
