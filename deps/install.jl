Pkg.add("Compat")
using Compat
import Compat.String

const home = "https://github.com/JuliaSmoothOptimizers"
const deps = Dict{String, String}(
              "NLPModels" => "develop",
              "OptimizationProblems" => "master",
              "Krylov" => "develop",
              "AmplNLReader" => "develop",
              "Profiles" => "master")

const unix_deps = Dict{String, String}(
              "CUTEst" => "develop")

function dep_installed(dep)
  try
    Pkg.installed(dep)  # throws an error instead of returning false
    return true
  catch
    return false
  end
end

function dep_install(dep, branch)
  dep_installed(dep) || Pkg.clone("$home/$dep.jl.git")
  Pkg.checkout(dep, branch)
  Pkg.build(dep)
end

function deps_install(deps)
  for dep in keys(deps)
    dep_install(dep, deps[dep])
  end
end

deps_install(deps)
@static if is_unix() deps_install(unix_deps); end
