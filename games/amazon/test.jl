#=
#cd("/home/david/Documents/gitclones/AlphaZero.jl")
cd("/home/davidyang/Documents/gitclones/AlphaZero.jl")
using Pkg
Pkg.activate(".")
using AlphaZero
#include("/home/david/Documents/gitclones/AlphaZero.jl/games/amazon/main.jl")
include("/home/davidyang/Documents/gitclones/AlphaZero.jl/games/amazon/main.jl")


gs = Amazon.GameSpec()
g = GI.init(gs)
GI.render(g)
while !g.finished
  action = g|>GI.actions_mask|>findall|>rand
  println("action taken ",GI.action_string(gs,action))
  GI.play!(g,action)
  GI.render(g)
end

gs = Amazon.GameSpec()
g = GI.init(gs)
GI.render(g)
Int.(GI.vectorize_state(gs,GI.current_state(g))[:,:,4])
GI.play!(g,CartesianIndex(1,2))
GI.render(g)
Int.(GI.vectorize_state(gs,GI.current_state(g))[:,:,5])
GI.play!(g,CartesianIndex(4,5))
GI.render(g)
Int.(GI.vectorize_state(gs,GI.current_state(g))[:,:,6])
=#


cd("/home/davidyang/Documents/gitclones/AlphaZero.jl")
using Pkg
Pkg.activate(".")
using AlphaZero
experiment = Examples.experiments["amazon"]
session = Session(experiment, dir="sessions/amazon")
resume!(session)



duel1 = session.benchmark[1]
progress = UI.Log.Progress(session.logger, duel1.sim.num_games)
env = session.env
net() = Network.copy(env.bestnn, on_gpu=duel1.sim.use_gpu, test_mode=true)
simulator = Simulator(net, record_trace) do net
  player = Benchmark.instantiate(duel1.player, env.gspec, net)
  baseline = Benchmark.instantiate(duel1.baseline, env.gspec, net)
  return TwoPlayers(player, baseline)
end
samples, elapsed = @timed simulate(
    simulator, env.gspec, duel1.sim,
    game_simulated=(() -> next!(progress)))
