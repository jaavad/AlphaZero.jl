module Examples

  using ..AlphaZero

  include("../games/tictactoe/main.jl")
  export Tictactoe

  include("../games/connect-four/main.jl")
  export ConnectFour

  include("../games/grid-world/main.jl")
  export GridWorld

  include("../games/mancala/main.jl")
  export Mancala

  include("../games/amazon/main.jl")
  export Amazon

  const games = Dict(
    "grid-world" => GridWorld.GameSpec(),
    "tictactoe" => Tictactoe.GameSpec(),
    "connect-four" => ConnectFour.GameSpec(),
    "mancala" => Mancala.GameSpec(),
    "amazon" => Amazon.GameSpec())
    # "ospiel_ttt" => OSpielTictactoe.GameSpec()
  # ospiel_ttt is added from openspiel_example.jl when OpenSpiel.jl is imported


  const experiments = Dict(
    "grid-world" => GridWorld.Training.experiment,
    "tictactoe" => Tictactoe.Training.experiment,
    "connect-four" => ConnectFour.Training.experiment,
    "mancala" => Mancala.Training.experiment)
    #TODO:add Amazon.Training.experiment
    # "ospiel_ttt" => OSpielTictactoe.Training.experiment

end