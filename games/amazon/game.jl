import AlphaZero.GI

using StaticArrays
using Crayons

const Cell = UInt8
const EMPTY = 0x00
const ARROW = 0x03

tmp = [0 1 0 0 1 0;
       1 0 0 0 0 1;
       0 0 0 0 0 0;
       0 0 0 0 0 0;
       2 0 0 0 0 2;
       0 2 0 0 2 0]

const NUM_COLS = size(tmp,1)
const NUM_ROWS = size(tmp,2)
const NUM_CELLS = length(tmp)
const Board = SMatrix{NUM_COLS, NUM_ROWS, Cell, NUM_CELLS}

const Player = UInt8
const WHITE = 0x01
const BLACK = 0x02

other(p::Player) = 0x03 - p
       

const INITIAL_BOARD = SMatrix{NUM_COLS,NUM_ROWS}(Player.(tmp))
const INITIAL_STATE = (board=INITIAL_BOARD, curplayer=WHITE, phase=0x00,selected=CartesianIndex(-1,-1))

struct GameSpec <: GI.AbstractGameSpec end


mutable struct GameEnv <: GI.AbstractGameEnv
  board :: Board
  curplayer :: Player
  finished :: Bool
  winner :: Player
  phase :: UInt8
  selected :: CartesianIndex{2}
end

GI.spec(::GameEnv) = GameSpec()

function GI.init(::GameSpec)
  board = INITIAL_STATE.board
  curplayer = INITIAL_STATE.curplayer
  finished = false
  winner = 0x00
  phase = INITIAL_STATE.phase
  selected = INITIAL_STATE.selected
  return GameEnv(board, curplayer, finished, winner, phase, selected)
end

GI.two_players(::GameSpec) = true

GI.actions(::GameSpec) = CartesianIndices((NUM_ROWS,NUM_COLS))



function GI.vectorize_state(::GameSpec, state)
  tmp = zeros(Float32,NUM_CELLS + 4)
  tmp[1:NUM_CELLS] = state.board[:]
  tmp[NUM_CELLS+1] = state.curplayer
  tmp[NUM_CELLS+2] = state.phase
  tmp[NUM_CELLS+3] = state.selected.I[0]
  tmp[NUM_CELLS+4] = state.selected.I[1]
  return tmp
end



function GI.set_state!(g::GameEnv, state)
  g.board = state.board
  g.curplayer = state.curplayer
  g.phase = state.phase
  g.selected = state.selected
  if any(GI.actions_mask(g))
    g.winner = 0x00
    g.finished = false
  else
    g.winner = other(g.curplayer)
    g.finished = true
  end
  return
end

GI.current_state(g::GameEnv) = 
  (board=g.board, curplayer=g.curplayer,phase=g.phase,selected=g.selected)


# TODO: maybe MCTS should make the copy itself. The performance cost should not be great
# and it would probably avoid people a lot of pain.


GI.game_terminated(g::GameEnv) = g.finished

GI.white_playing(g::GameEnv) = g.curplayer == WHITE

function get_queen_moves(b::Board,loc::CartesianIndex{2})
  valid_moves = falses(size(b))
  for dir∈CartesianIndex.([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)])
    for i=1:max(size(b)...)
      tmploc = loc + i * dir
      if checkbounds(Bool,b,tmploc) && b[tmploc]==0
        valid_moves[tmploc] = true
      else
        break
      end
    end
  end
  return valid_moves
end

function GI.actions_mask(g::GameEnv)
  if g.phase == 0
    return g.board .== g.curplayer
  else
    return get_queen_moves(g.board,g.selected)
  end
end

function GI.play!(g::GameEnv, a)
  if g.phase == 0
    g.selected = a
    if !any(get_queen_moves(g.board,a))
      g.finished = true
      g.winner = other(g.curplayer)
    end
  elseif g.phase == 1
    g.board = Base.setindex(g.board,0,g.selected)
    g.board = Base.setindex(g.board,g.curplayer,a)
    g.selected = a 
  else
    g.board = Base.setindex(g.board,3,a)
    g.curplayer = other(g.curplayer)
    g.selected = CartesianIndex(-1,-1)
  end
  g.phase = (g.phase+1)%3
end

function GI.white_reward(g::GameEnv)
  g.winner == WHITE && (return  1.)
  g.winner == BLACK && (return -1.)
  return 0.
end

pos_of_xy((x, y)) = (y - 1) * NUM_COLS + (x - 1) + 1

xy_of_pos(pos) = ((pos - 1) % NUM_COLS + 1, (pos - 1) ÷ NUM_COLS + 1)


function generate_dihedral_symmetries()
  N = NUM_COLS
  rot((x, y)) = (y, N - x + 1) # 90° rotation
  flip((x, y)) = (x, N - y + 1) # flip along vertical axis
  ap(f) = p -> pos_of_xy(f(xy_of_pos(p)))
  sym(f) = map(ap(f), collect(1:NUM_CELLS))
  rot2 = rot ∘ rot
  rot3 = rot2 ∘ rot
  return [
    sym(rot), sym(rot2), sym(rot3),
    sym(flip), sym(flip ∘ rot), sym(flip ∘ rot2), sym(flip ∘ rot3)]
end

const SYMMETRIES = generate_dihedral_symmetries()

function GI.symmetries(::GameSpec, s)
  return [
    ((board=Board(s.board[sym]), curplayer=s.curplayer,phase=s.phase,selected=s.selected), sym)
    for sym in SYMMETRIES]
end

#####
##### Interface for interactive exploratory tools
#####

player_color(p) = p == WHITE ? crayon"light_red" : crayon"light_blue"
player_name(p)  = p == WHITE ? "Red" : "Blue"
function cell_mark(c)
  if c == EMPTY
    return "."
  elseif c == ARROW
    return "x"
  else
    return "o"
  end
end

function cell_color(c;selected=false) 
  if selected
    return crayon"light_green"
  elseif c == ARROW || c == EMPTY
    return crayon""
  else
    return player_color(c)
  end
end

function GI.render(g::GameEnv; with_position_names=true, botmargin=true)
  pname = player_name(g.curplayer)
  pcol = player_color(g.curplayer)
  print(pcol, pname)
  if g.phase == 0
    print(" selects:")
  elseif g.phase == 1
    print(" moves:")
  else
    print(" shoots")
  end
  print(crayon"reset", "\n\n")
  # Print legend
  print(" ")
  for col in 1:NUM_COLS
    print(col, " ")
  end
  print("\n")
  # Print board
  for row in 1:NUM_ROWS
    print(row," ")
    for col in 1:NUM_COLS
      x = g.board[row,col]
      selected = CartesianIndex(row,col) == g.selected
      print(cell_color(x;selected=selected), cell_mark(x), crayon"reset", " ")
    end
    print("\n")
  end
  botmargin && print("\n")
end

function GI.action_string(::GameSpec, a)
  return string(a.I[1])*" "*string(a.I[2])
end

function GI.parse_action(::GameSpec, str)
  try
    x,y = parse(Int, str[1]),parse(Int,str[3])
    ((1 <= x,y <= NUM_COLS)|>all) ? CartesianIndex(x,y) : nothing
  catch
    nothing
  end
end

function GI.read_state(::GameSpec)
  board = Array(INITIAL_BOARD)
  try
    for col in 1:NUM_COLS
      input = readline()
      for (row, c) in enumerate(input)
        c = lowercase(c)
        if c ∈ ['o', 'w', '1']
          board[col, row] = WHITE
        elseif c ∈ ['x', 'b', '2']
          board[col, row] = BLACK
        end
      end
    end
    nw = count(==(WHITE), board)
    nb = count(==(BLACK), board)
    if nw == nb
      curplayer = WHITE
    elseif nw == nb + 1
      curplayer = BLACK
    else
      return nothing
    end
    return (board=board, curplayer=curplayer)
  catch e
    return nothing
  end
end


