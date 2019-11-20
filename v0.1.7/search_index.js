var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#Home-1",
    "page": "Home",
    "title": "SolverTools.jl documentation",
    "category": "section",
    "text": "This package provides tools for developing nonlinear optimization solvers."
},

{
    "location": "api/#",
    "page": "API",
    "title": "API",
    "category": "page",
    "text": ""
},

{
    "location": "api/#API-1",
    "page": "API",
    "title": "API",
    "category": "section",
    "text": "Pages = [\"api.md\"]"
},

{
    "location": "api/#SolverTools.active",
    "page": "API",
    "title": "SolverTools.active",
    "category": "function",
    "text": "active(x, ℓ, u; rtol = 1e-8, atol = 1e-8)\n\nComputes the active bounds at x, using tolerance min(rtol * (uᵢ-ℓᵢ), atol). If ℓᵢ or uᵢ is not finite, only atol is used.\n\n\n\n\n\n"
},

{
    "location": "api/#SolverTools.breakpoints",
    "page": "API",
    "title": "SolverTools.breakpoints",
    "category": "function",
    "text": "nbrk, brkmin, brkmax = breakpoints(x, d, ℓ, u)\n\nFind the smallest and largest values of α such that x + αd lies on the boundary. x is assumed to be feasible. nbrk is the number of breakpoints from x in the direction d.\n\n\n\n\n\n"
},

{
    "location": "api/#SolverTools.compute_Hs_slope_qs!",
    "page": "API",
    "title": "SolverTools.compute_Hs_slope_qs!",
    "category": "function",
    "text": "slope, qs = compute_Hs_slope_qs!(Hs, H, s, g)\n\nComputes\n\nHs = H * s\nslope = dot(g,s)\nqs = ¹/₂sᵀHs + slope\n\n\n\n\n\n"
},

{
    "location": "api/#SolverTools.log_header",
    "page": "API",
    "title": "SolverTools.log_header",
    "category": "function",
    "text": "log_header(colnames, coltypes)\n\nCreates a header using the names in colnames formatted according to the types in coltypes. Uses internal formatting specification given by SolverTools.formats and default header translation given by SolverTools.default_headers.\n\nInput:\n\ncolnames::Vector{Symbol}: Column names.\ncoltypes::Vector{DataType}: Column types.\n\nKeyword arguments:\n\nhdr_override::Dict{Symbol,String}: Overrides the default headers.\ncolsep::Int: Number of spaces between columns (Default: 2)\n\nSee also log_row.\n\n\n\n\n\n"
},

{
    "location": "api/#SolverTools.log_row",
    "page": "API",
    "title": "SolverTools.log_row",
    "category": "function",
    "text": "log_row(vals)\n\nCreates a table row from the values on vals according to their types. Pass the names and types of vals to log_header for a logging table. Uses internal formatting specification given by SolverTools.formats.\n\nKeyword arguments:\n\ncolsep::Int: Number of spaces between columns (Default: 2)\n\n\n\n\n\n"
},

{
    "location": "api/#SolverTools.project!",
    "page": "API",
    "title": "SolverTools.project!",
    "category": "function",
    "text": "project!(y, x, ℓ, u)\n\nProjects x into bounds ℓ and u, in the sense of yᵢ = max(ℓᵢ, min(xᵢ, uᵢ)).\n\n\n\n\n\n"
},

{
    "location": "api/#SolverTools.project_step!",
    "page": "API",
    "title": "SolverTools.project_step!",
    "category": "function",
    "text": "project_step!(y, x, d, ℓ, u, α = 1.0)\n\nComputes the projected direction y = P(x + α * d) - x.\n\n\n\n\n\n"
},

{
    "location": "api/#Auxiliary-1",
    "page": "API",
    "title": "Auxiliary",
    "category": "section",
    "text": "active\nbreakpoints\ncompute_Hs_slope_qs!\nlog_header\nlog_row\nproject!\nproject_step!"
},

{
    "location": "api/#SolverTools.bmark_solvers",
    "page": "API",
    "title": "SolverTools.bmark_solvers",
    "category": "function",
    "text": "bmark_solvers(solvers :: Dict{Symbol,Any}, args...; kwargs...)\n\nRun a set of solvers on a set of problems.\n\nArguments\n\nsolvers: a dictionary of solvers to which each problem should be passed\nother positional arguments accepted by solve_problems(), except for a solver name\n\nKeyword arguments\n\nAny keyword argument accepted by solve_problems()\n\nReturn value\n\nA Dict{Symbol, AbstractExecutionStats} of statistics.\n\n\n\n\n\n"
},

{
    "location": "api/#SolverTools.solve_problems",
    "page": "API",
    "title": "SolverTools.solve_problems",
    "category": "function",
    "text": "solve_problems(solver, problems :: Any; kwargs...)\n\nApply a solver to a set of problems.\n\nArguments\n\nsolver: the function name of a solver\nproblems: the set of problems to pass to the solver, as an iterable of AbstractNLPModel.  It is recommended to use a generator expression (necessary for CUTEst problems).\n\nKeyword arguments\n\nsolver_logger::AbstractLogger: logger wrapping the solver call. (default: NullLogger).\nskipif::Function: function to be applied to a problem and return whether to skip it (default: x->false)\nprune: do not include skipped problems in the final statistics (default: true)\nany other keyword argument to be passed to the solver\n\nReturn value\n\na DataFrame where each row is a problem, minus the skipped ones if prune is true.\n\n\n\n\n\n"
},

{
    "location": "api/#Benchmarking-1",
    "page": "API",
    "title": "Benchmarking",
    "category": "section",
    "text": "bmark_solvers\nsolve_problems"
},

{
    "location": "api/#SolverTools.LineModel",
    "page": "API",
    "title": "SolverTools.LineModel",
    "category": "type",
    "text": "A type to represent the restriction of a function to a direction. Given f : R → Rⁿ, x ∈ Rⁿ and a nonzero direction d ∈ Rⁿ,\n\nϕ = LineModel(nlp, x, d)\n\nrepresents the function ϕ : R → R defined by\n\nϕ(t) := f(x + td).\n\n\n\n\n\n"
},

{
    "location": "api/#NLPModels.obj",
    "page": "API",
    "title": "NLPModels.obj",
    "category": "function",
    "text": "obj(f, t) evaluates the objective of the LineModel\n\nϕ(t) := f(x + td).\n\n\n\n\n\n"
},

{
    "location": "api/#NLPModels.grad",
    "page": "API",
    "title": "NLPModels.grad",
    "category": "function",
    "text": "grad(f, t) evaluates the first derivative of the LineModel\n\nϕ(t) := f(x + td),\n\ni.e.,\n\nϕ\'(t) = ∇f(x + td)ᵀd.\n\n\n\n\n\n"
},

{
    "location": "api/#NLPModels.grad!",
    "page": "API",
    "title": "NLPModels.grad!",
    "category": "function",
    "text": "grad!(f, t, g) evaluates the first derivative of the LineModel\n\nϕ(t) := f(x + td),\n\ni.e.,\n\nϕ\'(t) = ∇f(x + td)ᵀd.\n\nThe gradient ∇f(x + td) is stored in g.\n\n\n\n\n\n"
},

{
    "location": "api/#NLPModels.hess",
    "page": "API",
    "title": "NLPModels.hess",
    "category": "function",
    "text": "Evaluate the second derivative of the LineModel\n\nϕ(t) := f(x + td),\n\ni.e.,\n\nϕ\"(t) = dᵀ∇²f(x + td)d.\n\n\n\n\n\n"
},

{
    "location": "api/#SolverTools.redirect!",
    "page": "API",
    "title": "SolverTools.redirect!",
    "category": "function",
    "text": "redirect!(ϕ, x, d)\n\nChange the values of x and d of the LineModel ϕ, but retains the counters.\n\n\n\n\n\n"
},

{
    "location": "api/#Line-Search-1",
    "page": "API",
    "title": "Line-Search",
    "category": "section",
    "text": "LineModel\nobj\ngrad\ngrad!\nhess\nredirect!"
},

{
    "location": "api/#SolverTools.GenericExecutionStats",
    "page": "API",
    "title": "SolverTools.GenericExecutionStats",
    "category": "type",
    "text": "GenericExecutionStats(status, nlp; ...)\n\nA GenericExecutionStats is a struct for storing output information of solvers. It contains the following fields:\n\nstatus: Indicates the output of the solver. Use show_statuses() for the full list;\nsolution: The final approximation returned by the solver (default: []);\nobjective: The objective value at solution (default: Inf);\ndual_feas: The dual feasibility norm at solution (default: Inf);\nprimal_feas: The primal feasibility norm at solution (default: 0.0 if uncontrained, Inf otherwise);\niter: The number of iterations computed by the solver (default: -1);\nelapsed_time: The elapsed time computed by the solver (default: Inf);\ncounters::NLPModels.NLSCounters: The Internal structure storing the number of functions evaluations;\nsolver_specific::Dict{Symbol,Any}: A solver specific dictionary.\n\nThe counters variable is a copy of nlp\'s counters, and status is mandatory on construction. All other variables can be input as keyword arguments.\n\nNotice that GenericExecutionStats does not compute anything, it simply stores.\n\n\n\n\n\n"
},

{
    "location": "api/#SolverTools.show_statuses",
    "page": "API",
    "title": "SolverTools.show_statuses",
    "category": "function",
    "text": "show_statuses()\n\nShow the list of available statuses to use with GenericExecutionStats.\n\n\n\n\n\n"
},

{
    "location": "api/#Stats-1",
    "page": "API",
    "title": "Stats",
    "category": "section",
    "text": "GenericExecutionStats\nshow_statuses"
},

{
    "location": "api/#SolverTools.TrustRegionException",
    "page": "API",
    "title": "SolverTools.TrustRegionException",
    "category": "type",
    "text": "Exception type raised in case of error.\n\n\n\n\n\n"
},

{
    "location": "api/#SolverTools.AbstractTrustRegion",
    "page": "API",
    "title": "SolverTools.AbstractTrustRegion",
    "category": "type",
    "text": "AbstractTrustRegion\n\nAn abstract trust region type so that specific trust regions define update rules differently. Child types must have at least the following fields:\n\nacceptance_threshold :: AbstractFloat\ninitial_radius :: AbstractFloat\nradius :: AbstractFloat\nratio :: AbstractFloat\n\nand the following function:\n\nupdate!(tr, step_norm)\n\n\n\n\n\n"
},

{
    "location": "api/#SolverTools.aredpred",
    "page": "API",
    "title": "SolverTools.aredpred",
    "category": "function",
    "text": "ared, pred = aredpred(nlp, f, f_trial, Δm, x_trial, step, slope)\n\nCompute the actual and predicted reductions ∆f and Δm, where Δf = f_trial - f is the actual reduction is an objective/merit/penalty function, Δm = m_trial - m is the reduction predicted by the model m of f. We assume that m is being minimized, and therefore that Δm < 0.\n\n\n\n\n\n"
},

{
    "location": "api/#SolverTools.acceptable",
    "page": "API",
    "title": "SolverTools.acceptable",
    "category": "function",
    "text": "acceptable(tr)\n\nReturn true if a step is acceptable\n\n\n\n\n\n"
},

{
    "location": "api/#LinearOperators.reset!",
    "page": "API",
    "title": "LinearOperators.reset!",
    "category": "function",
    "text": "reset!(tr)\n\nReset the trust-region radius to its initial value\n\n\n\n\n\n"
},

{
    "location": "api/#SolverTools.get_property",
    "page": "API",
    "title": "SolverTools.get_property",
    "category": "function",
    "text": "A basic getter for AbstractTrustRegion instances. Should be overhauled when it\'s possible to overload getfield() and setfield!(). See https://github.com/JuliaLang/julia/issues/1974\n\n\n\n\n\n"
},

{
    "location": "api/#SolverTools.set_property!",
    "page": "API",
    "title": "SolverTools.set_property!",
    "category": "function",
    "text": "A basic setter for AbstractTrustRegion instances.\n\n\n\n\n\n"
},

{
    "location": "api/#SolverTools.update!",
    "page": "API",
    "title": "SolverTools.update!",
    "category": "function",
    "text": "update!(tr, step_norm)\n\nUpdate the trust-region radius based on the ratio of actual vs. predicted reduction and the step norm.\n\n\n\n\n\n"
},

{
    "location": "api/#SolverTools.TrustRegion",
    "page": "API",
    "title": "SolverTools.TrustRegion",
    "category": "type",
    "text": "Basic trust region type.\n\n\n\n\n\n"
},

{
    "location": "api/#SolverTools.TRONTrustRegion",
    "page": "API",
    "title": "SolverTools.TRONTrustRegion",
    "category": "type",
    "text": "Trust region used by TRON\n\n\n\n\n\n"
},

{
    "location": "api/#Trust-Region-1",
    "page": "API",
    "title": "Trust-Region",
    "category": "section",
    "text": "TrustRegionException\nSolverTools.AbstractTrustRegion\naredpred\nacceptable\nreset!\nget_property\nset_property!\nupdate!\nTrustRegion\nTRONTrustRegion"
},

{
    "location": "reference/#",
    "page": "Reference",
    "title": "Reference",
    "category": "page",
    "text": ""
},

{
    "location": "reference/#Reference-1",
    "page": "Reference",
    "title": "Reference",
    "category": "section",
    "text": ""
},

]}
