import os
import time as tm
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from preprocessing_id import preprocess_id

# blokus simulator


def next_state(row, col, p_id, p_var, padded_board, player_id, player_hands,
               position_square, count_pos_squares, position_attach, count_pos_attach, position_forbid, count_pos_forbid):
    
    # input parameters:
    #   row:                [5 - 24] row of the piece origin in the playing board 
    #   col:                [5 - 24] column of the piece origin in the playing board 
    #   p_id:               [0 - 20] id of the blokus piece, see 'Blokus Pieces.xlsx'
    #   p_var:              [0 - 7] id of the blokus piece variation, see 'Blokus Pieces.xlsx'
    #   board:              30 x 30 numpy array of the playing board, including padding
    #   player_id:          [1 - 4] id of the current player
    #   player_hands:       4 x 21 numpy array of the pieces in hand, boolean
    #   position_square:    see preprocessing_id.py
    #   count_pos_squares:  see preprocessing_id.py
    #   position_attach:    see preprocessing_id.py
    #   count_pos_attach:   see preprocessing_id.py
    #   position_forbid:    see preprocessing_id.py
    #   count_pos_forbid:   see preprocessing_id.py
    #
    # output parameters:
    #   state:              is < 0 if the move is invalid move, 1 for valid moves
    #   padded_board:       updated board
    #   player_id:          updated active player
    #   player_hands:       updated player hands
    #
    # returns same board, same player_id, same player_hands if move is not valid, 
    # otherwise returns updated board, updated player_id, updated player_hands
    
    # check if piece is available
    if player_hands[player_id - 1, p_id] == 0:
        # if piece already has been used -> invalid move
        return -1, padded_board, player_id, player_hands
    
    # check overlap with other pieces
    c_sq = count_pos_squares[p_id, p_var]
    row_squares = position_square[p_id, p_var, 0:c_sq, 0]
    col_squares = position_square[p_id, p_var, 0:c_sq, 1]
    if np.any(padded_board[row + row_squares, col + col_squares]):
        # if there is overlap (any non-zero) -> invalid move
        return -2, padded_board, player_id, player_hands
    
    # check attachment
    c_att = count_pos_attach[p_id, p_var]
    row_attach = position_attach[p_id, p_var, 0:c_att, 0]
    col_attach = position_attach[p_id, p_var, 0:c_att, 1]
    if not np.any(padded_board[row + row_attach, col + col_attach] == player_id):
        # if no attachment points match with player_id squares -> invalid move
        return -3, padded_board, player_id, player_hands
    
    # check adjacency with other current player pieces in forbidden zones
    c_forb = count_pos_forbid[p_id, p_var]
    row_forbid = position_forbid[p_id, p_var, 0:c_forb, 0]
    col_forbid = position_forbid[p_id, p_var, 0:c_forb, 1]
    if np.any(padded_board[row + row_forbid, col + col_forbid] == player_id):
        # if any forbidden zone corresponds to current player pieces -> invalid move
        return -4, padded_board, player_id, player_hands
    
    # if the move is valid...
    # place block and update board
    padded_board[row + row_squares, col + col_squares] = player_id
    # update hand
    player_hands[player_id - 1, p_id] = 0
    # update active player      
    next_player_list = [2, 3, 4, 1]
    player_id = next_player_list[player_id - 1] # 1 -> 2, ..., 4 -> 1
        
    return 1, padded_board, player_id, player_hands


if not os.path.isfile('pieces_data.pickle'):
    preprocess_id()

# load pieces data
with open('pieces_data.pickle', 'rb') as handle:
    pk_list = pk.load(handle)
    position_square = pk_list[0]
    count_pos_squares = pk_list[1]
    position_attach = pk_list[2]
    count_pos_attach = pk_list[3]
    position_forbid = pk_list[4]
    count_pos_forbid = pk_list[5]

pad_size = 5 # padding size = int(d_cell/2) where d_cell = 11 thus pad_size = 5
d_board = 20 # playing board edge dimension
n_pieces = 21 # number of blokus pieces for each player
# player hand, 4 x 21 boolean array (1 = in hand, 0 = already placed)
player_hands = np.ones((4, n_pieces), dtype='bool')
# game board, with padding = 5 to avoid checking for out of bounds exceptions
padded_board = np.ones((d_board + 2*pad_size, d_board + 2*pad_size))*5 # (unplayable area marked with 5)
padded_board[pad_size:-pad_size,pad_size:-pad_size] = 0 # only internal 20 x 20 is playable (marked with 0)
# player id: 1, 2, 3, 4 starting attachment point (corner outside of 20 x 20 playing board)
padded_board[4, 4] = 1      # start attachment point player 1
padded_board[4, 25] = 2     # start attachment point player 2
padded_board[25, 25] = 3    # start attachment point player 3
padded_board[25, 4] = 4     # start attachment point player 4
# starting player
player_id = 1 # player 1 always starts
# plot starting board
plt.matshow(padded_board)

# simulation
state_list = []
n_iter = int(1e5)
t_0 = tm.time()
for i in range(n_iter):
    row = np.random.randint(0, 19)
    col = np.random.randint(0, 19)
    p_id = np.random.randint(0, 21)
    p_var = np.random.randint(0, 7)
    state_param_list = []
    state_param_list += [row + pad_size, col + pad_size, p_id, p_var, padded_board, player_id, player_hands]
    state_param_list += [position_square, count_pos_squares, position_attach, count_pos_attach, position_forbid, count_pos_forbid]
    # compute next state
    state, padded_board, player_id, player_hands = next_state(*state_param_list)
    state_list.append(state)
# time estimation
elapsed = tm.time() - t_0
print('Elapsed time for %d iterations: %f s\n' % (n_iter, elapsed))
print('Average time for single iteration: %f ms' % (1000*elapsed/n_iter))

plt.figure()
plt.plot(state_list,'o')

plt.matshow(padded_board)

plt.show()
pass
