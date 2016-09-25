using ProfileView
using NLPModels
using Optimize
using OptimizationProblems

nlp = MathProgNLPModel(chainwoo());
stuff = trunk(nlp, verbose=false);
@profile stuff = trunk(nlp, verbose=false);
ProfileView.view()

# @time stuff = trunk(nlp, verbose=false);
