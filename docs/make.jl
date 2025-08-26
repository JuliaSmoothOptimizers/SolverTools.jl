using SolverTools
using Documenter

DocMeta.setdocmeta!(SolverTools, :DocTestSetup, :(using SolverTools); recursive = true)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers
const numbered_pages = [
  file for
  file in readdir(joinpath(@__DIR__, "src")) if file != "index.md" && splitext(file)[2] == ".md"
]

makedocs(;
  modules = [SolverTools],
  authors = "Dominique Orban <dominique.orban@gerad.ca>, Abel Soares Siqueira <abel.s.siqueira@gmail.com>, Tangi Migot <tangi.migot@gmail.com>",
  repo = "https://github.com/JuliaSmoothOptimizers/SolverTools.jl/blob/{commit}{path}#{line}",
  sitename = "SolverTools.jl",
  format = Documenter.HTML(; canonical = "https://JuliaSmoothOptimizers.github.io/SolverTools.jl"),
  pages = ["index.md"; numbered_pages],
)

deploydocs(; repo = "github.com/JuliaSmoothOptimizers/SolverTools.jl")
