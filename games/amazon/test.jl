cd("/home/david/Documents/gitclones/AlphaZero.jl")
using Pkg
Pkg.activate(".")
using AlphaZero
include("/home/david/Documents/gitclones/AlphaZero.jl/games/amazon/main.jl")

gs = Amazon.GameSpec()
g = GI.init(gs)
GI.render(g)
while !g.finished
  action = g|>GI.actions_mask|>findall|>rand
  println("action taken ",GI.action_string(gs,action))
  GI.play!(g,action)
  GI.render(g)
end
