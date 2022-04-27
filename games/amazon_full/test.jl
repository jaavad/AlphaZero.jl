
#cd("/home/david/Documents/gitclones/AlphaZero.jl")
cd("/home/davidyang/Documents/gitclones/AlphaZero.jl")
using Pkg
Pkg.activate(".")
using AlphaZero
#include("/home/david/Documents/gitclones/AlphaZero.jl/games/amazon_full/main.jl")
include("/home/davidyang/Documents/gitclones/AlphaZero.jl/games/amazon_full/main.jl")


gs = Amazon_Full.GameSpec()
g = GI.init(gs)
GI.vectorize_state(gs,GI.current_state(g))
GI.play!(g,g|>GI.actions_mask|>findall|>rand)
GI.vectorize_state(gs,GI.current_state(g))

g = GI.init(gs)
GI.render(g)
while !g.finished
  action = g|>GI.actions_mask|>findall|>rand
  println(GI.action_string(gs,action))
  GI.play!(g,action)
  GI.render(g)
end


cd("/home/davidyang/Documents/gitclones/AlphaZero.jl")
#cd("/home/david/Documents/gitclones/AlphaZero.jl")
#@everywhere using Pkg
#@everywhere Pkg.activate(".")
#@everywhere using AlphaZero
using Pkg
Pkg.activate(".")
using AlphaZero
experiment = Examples.experiments["amazon_full"]
session = Session(experiment, dir="sessions/amazon_full4")
resume!(session)

duel1 = session.benchmark[1]
#duel1 equivalent to eval
progress = UI.Log.Progress(session.logger, duel1.sim.num_games)
env = session.env
params = env.params.self_play
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
oracles = spawn_oracles()
player = simulator.make_player(oracles)

println("debugging play game")
trace = @time play_game(env.gspec, simulator.make_player(oracles), flip_probability=0.0)

colors_flipped = true
report = simulator.measure(trace, colors_flipped, player)

results, elapsed = @timed simulate_distributed(
    simulator, env.gspec, params.sim,
    game_simulated=()->Handlers.game_played(handler))

#seems stuck at the above function, so expand to see what's going on

simulate(simulator,
        gspec,
        SimParams(p; num_games=(w == workers[1] ? num_each + rem : num_each)),
        game_simulated=remote_game_simulated)

using Distributed
gspec = env.gspec
p = params.sim
chan = Distributed.RemoteChannel(()->Channel{Nothing}(1))
Util.@tspawn_main begin
  for i in 1:p.num_games
    take!(chan)
    game_simulated()
  end
end
remote_game_simulated() = put!(chan, nothing)
# Distributing the simulations across workers
num_each, rem = divrem(p.num_games, Distributed.nworkers())
@assert num_each >= 1
workers = Distributed.workers()
tasks = map(workers) do w
  Distributed.@spawnat w begin
    Util.@printing_errors begin
      simulate(
        simulator,
        gspec,
        SimParams(p; num_games=(w == workers[1] ? num_each + rem : num_each)),
        game_simulated=remote_game_simulated)
      end
  end
end
results = fetch.(tasks)
# If one of the worker raised an exception, we print it
for r in results
  if isa(r, Distributed.RemoteException)
    showerror(stderr, r, catch_backtrace())
  end
end


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