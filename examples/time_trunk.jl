using Profile
using ProfileView
using NLPModels
using NLPModelsJuMP
using Optimize
using OptimizationProblems

nlp = MathProgNLPModel(chainwoo());
stuff = trunk(nlp)
@profile stuff = trunk(nlp)
ProfileView.view()

# @time stuff = trunk(nlp, verbose=false);
