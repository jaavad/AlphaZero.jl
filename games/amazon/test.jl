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


#cd("/home/davidyang/Documents/gitclones/AlphaZero.jl")
cd("/home/david/Documents/gitclones/AlphaZero.jl")
using Pkg
Pkg.activate(".")
using AlphaZero
experiment = Examples.experiments["amazon"]
session = Session(experiment, dir="sessions/amazon")
#resume!(session)



duel1 = session.benchmark[1]
#duel1 equivalent to eval
progress = UI.Log.Progress(session.logger, duel1.sim.num_games)
env = session.env
net() = Network.copy(env.bestnn, on_gpu=duel1.sim.use_gpu, test_mode=true)
simulator = Simulator(net, record_trace) do net
  player = Benchmark.instantiate(duel1.player, env.gspec, net)
  baseline = Benchmark.instantiate(duel1.baseline, env.gspec, net)
  return TwoPlayers(player, baseline)
end
#=
samples, elapsed = @timed simulate(
    simulator, env.gspec, duel1.sim,
    game_simulated=(() -> next!(progress)))
=#
oracles = simulator.make_oracles()
spawn_oracles, done =
    AlphaZero.batchify_oracles(oracles; duel1.sim.num_workers, duel1.sim.batch_size, duel1.sim.fill_batches)

println("debugging play game")
trace = play_game(env.gspec, simulator.make_player(oracles), flip_probability=1.0)


flip_probability = 0.0
player = simulator.make_player(oracles)
game = GI.init(env.gspec)
trace = Trace(GI.current_state(game))
#actions, π_target = think(player, game)
env = player.white.mcts
#MCTS.explore!(env,game,player.white.niters)
#η = MCTS.dirichlet_noise(game, env.noise_α)
#MCTS.run_simulation!(env,GI.clone(game),η=η)
#MCTS.state_info(env,GI.current_state(game))
#env.oracle(state)
#Network.evaluate(env.oracle,state)
nn = env.oracle
gspec = Network.game_spec(nn)
actions_mask = GI.actions_mask(GI.init(gspec, state))
x = GI.vectorize_state(gspec, state)
a = Float32.(actions_mask)
xnet, anet = Network.to_singletons.(Network.convert_input_tuple(nn, (x, a)))
#net_output = Network.forward_normalized(nn, xnet, anet)
p, v = Network.forward(nn, xnet)

while true
  if GI.game_terminated(game)
    return trace
  end
  if !iszero(flip_probability) && rand() < flip_probability
    GI.apply_random_symmetry!(game)
  end
  actions, π_target = think(player, game)
  τ = player_temperature(player, game, length(trace))
  π_sample = apply_temperature(π_target, τ)
  a = actions[Util.rand_categorical(π_sample)]
  GI.play!(game, a)
  push!(trace, π_target, GI.white_reward(game), GI.current_state(game))
end