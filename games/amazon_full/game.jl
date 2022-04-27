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
const INITIAL_STATE = (board=INITIAL_BOARD, curplayer=WHITE)

struct GameSpec <: GI.AbstractGameSpec end


mutable struct GameEnv <: GI.AbstractGameEnv
  board :: Board
  curplayer :: Player
  finished :: Bool
  winner :: Player
  #selected works as long as size of board is not too big
end

GI.spec(::GameEnv) = GameSpec()

function GI.init(::GameSpec)
  board = INITIAL_STATE.board
  curplayer = INITIAL_STATE.curplayer
  finished = false
  winner = 0x00
  return GameEnv(board, curplayer, finished, winner)
end

GI.two_players(::GameSpec) = true

GI.actions(::GameSpec) = collect(1:9412)
#=
15,15,15,15,15,15
15,17,17,17,17,15
15,17,19,19,17,15
15,17,19,19,17,15
15,17,17,17,17,15
15,15,15,15,15,15

Above is queen moves available at each square.
Because every move can be uniquely identified by a→b→c,
so if b is fixed, it can come from x places and go to x places,
contributing to a total of x^2 moves
So the total number of moves is 
20*15^2+12*17^2+4*19^2=9412
=#


flip_cell_color(c::Cell) = (0 < c < 3) ? other(c) : c

flip_colors(board::Board) = flip_cell_color.(board)

function GI.vectorize_state(::GameSpec, state)
  tmp = zeros(Float32,NUM_ROWS,NUM_COLS,3)
  board = state.curplayer == WHITE ? state.board : flip_colors(state.board)
  for i = 1:3
    tmp[:,:,i] = board .== i
  end
  return tmp
end



function GI.set_state!(g::GameEnv, state)
  g.board = state.board
  g.curplayer = state.curplayer
  if any(GI.actions_mask(g))
    g.winner = 0x00
    g.finished = false
  else
    g.winner = other(g.curplayer)
    g.finished = true
  end
  return
end

GI.current_state(g::GameEnv) = (board=g.board, curplayer=g.curplayer)

# TODO: maybe MCTS should make the copy itself. The performance cost should not be great
# and it would probably avoid people a lot of pain.

GI.game_terminated(g::GameEnv) = g.finished

GI.white_playing(g::GameEnv) = g.curplayer == WHITE

const DIRECTIONS = CartesianIndex.([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)])

pos_of_xy((x, y)) = (y - 1) * NUM_COLS + (x - 1) + 1
xy_of_pos(pos) = ((pos - 1) % NUM_COLS + 1, (pos - 1) ÷ NUM_COLS + 1)

function get_queen_moves!(b,loc_linear,container)
  loc = CartesianIndex(xy_of_pos(loc_linear))
  valid_moves = reshape(container,NUM_ROWS,NUM_COLS)
  for dir∈DIRECTIONS
    for i=1:max(size(b)...)
      tmploc = loc + i * dir
      if checkbounds(Bool,b,tmploc) && b[tmploc]==0
        valid_moves[tmploc] = true
      else
        break
      end
    end
  end
end

function get_queen_moves(b::Board,loc_linear)
  valid_moves = falses(length(b))
  get_queen_moves!(b,loc_linear,valid_moves)
  return valid_moves
end

function precompute_mask()
  mask = falses(36*36*36)
  tmp = reshape(mask,36,36,36)
  tmp_board = zeros(eltype(INITIAL_BOARD),NUM_ROWS,NUM_COLS)
  tmp_board = SMatrix{NUM_COLS,NUM_ROWS}(Player.(tmp_board))
  moves = falses(NUM_ROWS,NUM_COLS)
  shoots = falses(NUM_ROWS,NUM_COLS)
  for i=1:36
    fill!(moves,false)
    get_queen_moves!(tmp_board,i,moves)
    for j=1:36
      !moves[j] && continue 
      fill!(shoots,false)
      get_queen_moves!(tmp_board,j,shoots)
      for k=1:36
        !shoots[k] && continue
        tmp[i,j,k]=true
      end
    end
  end
  return mask
end

const PRECOMPUTED_MASK = precompute_mask()
const NON_ZERO_LOCS = findall(!iszero,PRECOMPUTED_MASK)

#=
Because of the representation, the matrix is 36^3=46656, but there are only 9412 valid moves
So the strategy is, given a state, find the valid moves, fill them into the 36^3 matrix, which has better representation
but lower efficiency. Most of the entries will always be zero, so filter them out using a precomputed mask.

Inversely, given a number x between 1:9412, find the location of the xth non-zero entry in the precomputed mask
=#

function GI.actions_mask(g::GameEnv)
  moves = falses(NUM_ROWS,NUM_COLS)
  shoots = falses(NUM_ROWS,NUM_COLS)
  board = zeros(Cell,NUM_ROWS,NUM_COLS)
  mask = falses(NUM_CELLS^3)
  tmp = reshape(mask,NUM_CELLS,NUM_CELLS,NUM_CELLS)
  for i=1:NUM_CELLS
    board[i] = g.board[i]
  end
  for i=1:NUM_CELLS
    board[i]!=g.curplayer && continue
    fill!(moves,false)
    get_queen_moves!(board,i,moves)
    for j=1:NUM_CELLS
      !moves[j] && continue 
      fill!(shoots,false)
      #don't forget in this case, the amazon could shoot
      #to where it came from. So temporarily delete the current
      #piece, and then add it back
      board[i]=0x00
      get_queen_moves!(board,j,shoots)
      board[i]=g.curplayer
      for k=1:NUM_CELLS
        !shoots[k] && continue
        tmp[i,j,k]=true
      end
    end
  end
  return mask[PRECOMPUTED_MASK]
end

function GI.play!(g::GameEnv, a)
  #first map from the compressed representation to the full representation,
  #then map from 1 based index to 0 based index
  a = NON_ZERO_LOCS[a]
  a = a - 1
  source = a % NUM_CELLS
  tmp = (a-source)÷NUM_CELLS
  move = tmp % NUM_CELLS
  shoot = (tmp - move) ÷ NUM_CELLS

  shoot += 1
  move += 1
  source += 1

  g.board = Base.setindex(g.board,0,source)
  g.board = Base.setindex(g.board,g.curplayer,move)
  g.board = Base.setindex(g.board,3,shoot)
  g.curplayer = other(g.curplayer)
  new_actions = GI.actions_mask(g)
  if !any(new_actions)
    g.finished=true
    g.winner = other(g.curplayer)
  end
end

function GI.white_reward(g::GameEnv)
  g.winner == WHITE && (return  1.)
  g.winner == BLACK && (return -1.)
  return 0.
end



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

function cell_color(c) 
  if c == ARROW || c == EMPTY
    return crayon""
  else
    return player_color(c)
  end
end

function GI.render(g::GameEnv; with_position_names=true, botmargin=true)
  pname = player_name(g.curplayer)
  pcol = player_color(g.curplayer)
  print(pcol, pname," next")
  print(crayon"reset", "\n\n")
  # Print legend
  print("  ")
  for col in 1:NUM_COLS
    print(col, " ")
  end
  print("\n")
  # Print board
  for row in 1:NUM_ROWS
    print(row," ")
    for col in 1:NUM_COLS
      x = g.board[row,col]
      print(cell_color(x), cell_mark(x), crayon"reset", " ")
    end
    print("\n")
  end
  botmargin && print("\n")
end

function GI.action_string(::GameSpec, a)
  a = NON_ZERO_LOCS[a]
  a = a - 1
  source = a % NUM_CELLS
  tmp = (a-source)÷NUM_CELLS
  move = tmp % NUM_CELLS
  shoot = (tmp - move) ÷ NUM_CELLS

  shoot += 1
  move += 1
  source += 1
  return "move from " * string(xy_of_pos(source)) *
         " to "       * string(xy_of_pos(move)) *
         " shoot to " * string(xy_of_pos(shoot))
end

function GI.parse_action(::GameSpec, str)
  try
    x,y = parse(Int, str[1]),parse(Int,str[3])
    ((1 <= x,y <= NUM_COLS)|>all) ? CartesianIndex(x,y) : nothing
  catch
    nothing
  end
end

#=
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

=#
