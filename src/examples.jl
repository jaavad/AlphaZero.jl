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

  include("../games/amazon_full/main.jl")
  export Amazon_Full

  const games = Dict(
    "grid-world" => GridWorld.GameSpec(),
    "tictactoe" => Tictactoe.GameSpec(),
    "connect-four" => ConnectFour.GameSpec(),
    "mancala" => Mancala.GameSpec(),
    "amazon" => Amazon.GameSpec(),
    "amazon_full"=>Amazon_Full.GameSpec())
    # "ospiel_ttt" => OSpielTictactoe.GameSpec()
  # ospiel_ttt is added from openspiel_example.jl when OpenSpiel.jl is imported


  const experiments = Dict(
    "grid-world" => GridWorld.Training.experiment,
    "tictactoe" => Tictactoe.Training.experiment,
    "connect-four" => ConnectFour.Training.experiment,
    "mancala" => Mancala.Training.experiment,
    "amazon" => Amazon.Training.experiment,
    "amazon_full" => Amazon_Full.Training.experiment)
    # "ospiel_ttt" => OSpielTictactoe.Training.experiment

end