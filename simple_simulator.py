import os
import time as tm
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from preprocessing_id import preprocess_id
from blokus_environment import next_state

# blokus simulator
# https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/


# ----- main -----
# get piece data
(position_square, count_pos_squares, position_attach, count_pos_attach, position_forbid, count_pos_forbid) = preprocess_id()

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
player_id = 0 # player 1 always starts
# plot starting board
plt.matshow(padded_board)

# ----- simulation -----
state_list = []
n_iter = int(1e5)
t_0 = tm.time()
for i in range(n_iter):
    row = np.random.randint(0, 19)
    col = np.random.randint(0, 19)
    p_id = np.random.randint(0, 21)
    p_var = np.random.randint(0, 7)
    state_param_list = []
    state_param_list += [row + pad_size, col + pad_size, p_id, p_var, padded_board, player_id, player_hands, 4]
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
