import pygame

import numpy as np
from numpy import logical_and as np_and
from numpy import logical_or as np_or

from functools import wraps
import time

import torch
import torch.nn.functional as f
import gymnasium as gym
from gymnasium import spaces

from environments.preprocessing_id import preprocess_id
from environments.preprocessing_action import preprocess_action

from matplotlib import use as plt_use
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import random
from itertools import product


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # print(f'Function {func.__name__}{args} {kwargs} Took {1000*total_time:.4f} ms')
        print(f'Function {func.__name__} Took {1000*total_time:.4f} ms')
        return result
    return timeit_wrapper


class BlokusEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 8}

    def __init__(self, action_mode='discrete_masked', render_mode=None, d_board=20, win_width=640 * 2, win_height=480 * 2, win_reward=50,
                 invalid_reward=-10, dead_reward=-1):

        # two possible action spaces: 
        #   'discrete_masked' (uses exact action masking)
        #   'multi_discrete' (uses approximate action masking)
        assert action_mode in ['discrete_masked', 'multi_discrete']
        self.action_mode = action_mode

        # --- constants ---
        self.piece_data = preprocess_id()  # computes the blokus pieces data
        self.pad = 5  # pentominos need extra 5 spaces over the board for generic placement
        self.d = d_board  # playing board edge dimension
        self.n_pieces = 21  # number of blokus pieces for each player
        # number of vairant for each piece, 4 rotation times 2 flip states by default
        self.n_variant = 8
        self.n_pl = 4  # number of players during a game, by default is always 4
        self.rot90_variant = [1, 2, 3, 0, 5, 6, 7,
                              4]  # next variant when rotating 90 deg counter-clockwise, see preprocess_id.py
        self.win_reward = win_reward  # reward for placing all pieces
        # [< 0] penalty for performing an invalid action (with masking should not be relevant)
        self.invalid_reward = invalid_reward
        self.dead_reward = dead_reward  # [< 0] penalty for being dead early
        # pygame and rendering attributes
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_w = win_width  # pygame window width
        self.window_h = win_height  # pygame window height
        # white, red, green, blue, yellow, grey from: https://service.mattel.com/instruction_sheets/BJV44-Eng.pdf
        self.rgb_col = [[255, 255, 255, 0], [215, 29, 36, 255], [0, 161, 75, 255], [
            0, 90, 170, 255], [253, 184, 19, 255], [222, 217, 210, 255]]
        self.rgb_col = np.array(self.rgb_col) / 255
        self.cmap = [
            ListedColormap(np.vstack((self.rgb_col[0, :], np.roll(
                self.rgb_col[1:self.n_pl + 1], -i, axis=0), self.rgb_col[-1, :])))
            for i in range(self.n_pl)
        ]        

        # --- resettable ---
        # boolean array, in the i-th is True if the i-th player can no longer place pieces
        self.dead = np.zeros((self.n_pl, 1), dtype='bool')
        # True when the game ends (win condition)
        self.end_game = False
        # invalid last move [0 - 4], valid move = 0, invalid move = [1 - 3], see next_state for details
        self.invalid = 0
        # number of moves completed, counting all players
        self.move_count = 0
        # active player number, can be 0,1,2,3 with the default 4 players
        self.active_pl = int(self.move_count / self.n_pl)
        # player hand, 4 x 21 boolean array (1 = in hand, 0 = already placed), as seen by the 4 player
        self.player_hands = np.ones(
            (self.n_pl, self.n_pieces, self.n_pl), dtype='bool')
        # game board, with padding = 5 to avoid checking for out of bounds exceptions, one for each player
        self.pad_board = np.ones(
            (self.d + 2 * self.pad, self.d + 2 * self.pad, self.n_pl), dtype='int')
        # resets game board with padding (to do preprocess_action only in __init__() and not in reset())
        # (unplayable area marked with 5)
        self.pad_board = np.ones(
            (self.d + 2 * self.pad, self.d + 2 * self.pad, self.n_pl), dtype='int') * 5
        # only internal 20 x 20 is playable (marked with 0)
        self.pad_board[self.pad:-self.pad, self.pad:-self.pad, :] = 0
        # player id: 1, 2, 3, 4 starting attachment point (corner outside of 20 x 20 playing board)
        for i in range(self.n_pl):
            # places first attachment points in the corners just outside the playing board (player_color = player_id + 1)
            # start attachment point player 1, for each POV
            self.pad_board[
                self.pad - 1, self.pad - 1, i] = 1
            # start attachment point player 2, for each POV
            self.pad_board[
                self.pad - 1, self.d + self.pad, i] = 2
            # start attachment point player 3, for each POV
            self.pad_board[
                self.d + self.pad, self.d + self.pad, i] = 3
            # start attachment point player 4, for each POV
            self.pad_board[
                self.d + self.pad, self.pad - 1, i] = 4
        # id of the last placed piece, mainly used for rewarding
        self.last_p_id = -1

        # --- observation ---
        #   board is the playing board where 0 = empty, 1-4 = player square
        #   hands represent the available pieces in the hands of every player
        #   turn is the turn number of the current player, derives from move_count
        #   invalid is a flag for the validity of the played action (0 = valid, 1-3 = invalid)
        self.observation_space = spaces.Dict(
            {
                'board': spaces.MultiBinary((self.d, self.d, self.n_pl)),
                'hands': spaces.MultiBinary((self.n_pl, self.n_pieces)),
                'turn': spaces.Box(0, self.n_pieces - 1, dtype='int'),
                'invalid': spaces.Box(0, 4, dtype='int')
            }
        )

        if self.action_mode == 'discrete_masked':
            # --- constants ---
            # dimension of the action space, in default settings the game has 67.2k possible actions
            self.action_dim = self.d * self.d * self.n_pieces * self.n_variant
            # indexes used to avoid reshaping of the action masks when selecting a single piece
            temp_id = np.reshape(np.arange(
                0, self.action_dim, 1), (self.d, self.d, self.n_pieces, self.n_variant))
            self.p_flat_id = {i: temp_id[:, :, i, :]
                              for i in range(self.n_pieces)}

            # --- resettable ---
            # boolean mask of the action space for each player, indicating valid actions from each player's POV
            self.valid_act_mask = np.zeros(
                (self.n_pl, self.action_dim), dtype='bool')
            # boolean mask of the actve_player actions that are always invalid
            self.alw_inv = None
            # [i,j,:] of invalid_to_maybe_valid is a boolean mask for the action of the active_player that could potentially turn from
            # invalid to valid once a square of the color of the active_player is placed in [i, j] of the playing board (board without padding)
            self.inv_to_val = None
            # [i,j,:] of valid_to_invalid is a boolean mask for the action of each player that turns from valid to invalid
            # once a square of the color of the active_player is placed in [i, j] of the playing board (board without padding)
            self.val_to_inv = None
            # same as valid_to_invalid, but it applies only to the active_player (the one who does the action)
            self.val_to_inv_act_pl = None
            # boolean mask of the actve_player actions that are valid as their first move (from their POV)
            self.alw_inv = None
            # boolean mask that collects all the invalid moves due to placement and always invalid, is updated every step
            self.invalid_history = np.zeros(
                (self.n_pl, self.action_dim), dtype='bool')
            # preprocess action validity
            self.action_data = preprocess_action(
                self.pad_board, self.pad, self.n_pieces, self.n_variant, *self.piece_data)
            (self.alw_inv, self.inv_to_val, self.val_to_inv,
             self.val_to_inv_act_pl, self.val_at_start) = self.action_data
            # wether step method kills player after invalid move
            self.kill_if_invalid_move = False

            # --- action ---
            # array representation of a multidimensional space
            #   row position of the origin of the piece to play [0-d]
            #   column position of the origin of the piece to play [0-d]
            #   piece to play [0-n]
            #   variant of the piece to play [0-7]
            self.action_space = spaces.Discrete(self.action_dim)

        elif self.action_mode == 'multi_discrete':
            
            # wether step method kills player after invalid move
            self.kill_if_invalid_move = True
            
            # --- action ---
            # same action space as the discrete_masked space, but the action space is decomposed in the 4 basic actions
            #   row position of the origin of the piece to play [0-d]
            #   column position of the origin of the piece to play [0-d]
            #   piece to play [0-n]
            #   variant of the piece to play [0-7]
            self.action_space = spaces.MultiDiscrete(
                [self.d*self.d, self.n_pieces, self.n_variant])

    # ----- _get_reward helper functions -----

    def is_last_standing(self, player):
        # returns True if player [0 - 3] is the only one alive (all others are dead)
        win_mask = np.ones((self.n_pl,), dtype='bool')
        win_mask[player] = False
        return np.all(self.dead == win_mask)

    def all_pieces_placed(self, player):
        # returns True if player [0 - 3] has placed all his pieces, thus his hand is empty (all False)
        placed_pieces_id = np.where(~self.player_hands[player, :, 0])
        return len(placed_pieces_id) == self.n_pieces

    def all_dead(self):
        return np.all(self.dead)

    def is_active_player_dead(self):
        return self.dead[self.active_pl]

    def get_basic_score(self, player):
        # returns the total placed squares of player [0 -3]
        # number of squares in each piece
        count_pos_squares = self.piece_data[1]
        # ids of all placed pieces
        placed_pieces_id = np.where(~self.player_hands[player, :, 0])
        # variant is irrelevant, can be set to 0
        return np.sum(count_pos_squares[placed_pieces_id, 0])
    
    def get_last_p_score(self):
        # returns the squares of the last piece placed
        # number of squares in each piece
        count_pos_squares = self.piece_data[1]
        # variant is irrelevant, can be set to 0
        return np.sum(count_pos_squares[self.last_p_id, 0])

    def get_n_valid(self, player):
        # returns the number of valid actions remaining to player [0 - 3]
        return len(self.valid_act_mask[player, :])

    def get_valid_ratio(self, player):
        # returns the number of valid actions of player [0 - 3] divided by the total number of possible actions
        return self.get_n_valid(player)/len(self.valid_act_mask[player, :])

    def get_mean_valid_ratio(self):
        # returns the number of valid actions divided by the number of possible actions, averaged across all players
        return np.mean([self.get_valid_ratio(pl) for pl in range(self.n_pl)])

    def _get_reward(self):

        if self.action_mode == 'discrete_masked':
            # returns the reward summing the following contributions:
            # (0) win_reward (>> 0) in case of win condition (all pieces placed and all players are dead)
            # (1) win_reward (>> 0) in case of all dead but active_player has highest score
            # (2) dead_reward (<< 0) in case the active player made its last move or the active player is dead (simplyfies to active player is dead)
            # (3) ...
            # TODO: better tie breakers

            if self.all_pieces_placed(self.active_pl) and self.all_dead():
                self.end_game = True
                return self.win_reward  # (0)
            
            if self.all_dead():
                self.end_game = True
                active_pl_score = self.get_basic_score(self.active_pl)
                scores = np.array([self.get_basic_score(i) for i in range(self.n_pl)])
                if np.all(active_pl_score >= scores):
                    return self.win_reward  # (1)
                else:
                    return 0
            
            if self.dead[self.active_pl]:
                return self.dead_reward  # (2)

            step_rew = self.get_valid_ratio(self.active_pl)
            step_rew -= sum([self.get_valid_ratio((self.active_pl + i) % self.n_pl) for i in range(1, self.n_pl)])/(self.n_pl - 1)
            step_rew += self.get_last_p_score()/5
            return step_rew  # (3)

        elif self.action_mode == 'multi_discrete':
            # returns the reward summing the following contributions:
            # (0) invalid_reward (<< 0) in case of invalid move
            # (1) win_reward (>> 0) in case of win condition (all pieces placed)
            # (2) squares of last placed piece / 5 (> 0) in every other case
            # TODO: better tie breakers

            if self.invalid:
                return self.invalid_reward  # (0)

            if self.all_pieces_placed(self.active_pl):
                self.end_game = True
                return self.win_reward  # (1)

            return self.get_last_p_score()/5  # (2)

    def _get_terminated(self):
        # returns True in case of win condition (all pieces placed or all dead except for active player) or every player is dead
        return bool(self.end_game)

    def _get_truncated(self):
        # returns True when an invalid action is performed
        return bool(self.invalid)

    def _get_obs(self):
        # returns the game board, hands, turn and action validity as seen by the active_player's POV
        multibin_playing_board = self.pad_board[
            self.pad: self.d + self.pad,
            self.pad: self.d + self.pad,
            self.active_pl
        ]
        multibin_playing_board = np.vstack((np.zeros((1, self.n_pl), dtype=np.int8), np.eye(self.n_pl, dtype=np.int8)))[
            multibin_playing_board]
        return {
            'board': multibin_playing_board,
            'hands': self.player_hands[:, :, self.active_pl],
            'turn': np.array([int(self.move_count / self.n_pl)]),
            'invalid': np.array([self.invalid])
        }

    def _get_info(self):
        # returns some additional game state information
        if self.action_mode == 'discrete_masked':
            return {
                'valid_masks': self.valid_act_mask,
                'active_player': self.active_pl,
                'active_player_valid_mask': self.valid_act_mask[self.active_pl, :],
                'move_count': self.move_count
            }
        elif self.action_mode == 'multi_discrete':
            return {
                'valid_masks': None,
                'active_player': self.active_pl,
                'active_player_valid_mask': None,
                'move_count': self.move_count
            }

    # @timeit
    def action_masks(self):
        if self.action_mode == 'discrete_masked':
            # returns the mask of the valid action of the active player as an array of bool (True = valid action)
            return self.valid_act_mask[self.active_pl, :]
        elif self.action_mode == 'multi_discrete':
            # returns an approximate mask of the valid action (some actions will still be invalid unfortunately)
            # saves True in board where the active player has squares already placed (pad_board == 1, from his POV)
            board = self.pad_board[:, : , self.active_pl] == (0 + 1)
            board = torch.tensor(board[None, None, ...])  # adding 2 dimensions
            weight = torch.tensor([[[[1, -10, 1], [-10, -10, -10], [1, -10, 1]]]])  # convolution kernel
            board_conv = f.conv2d(board.float(), weight.float(), stride=1, padding=1).numpy()  # convolution with torch                     
            # valid positions (and):
            #   positive convolution result (board_conv > 0)
            #   pad_board not occupied ny other pieces (pad_board == 0)
            valid_row_col = np_and(
                board_conv[0, 0, self.pad:-self.pad, self.pad:-self.pad] > 0, 
                self.pad_board[self.pad:-self.pad, self.pad:-self.pad, self.active_pl] == 0
            )
            valid_row_col_flat = np.ndarray.flatten(valid_row_col)  # 400 in std game
            valid_pieces = self.player_hands[self.active_pl, :, 0]  # 21 in standard game
            valid_orient = np.ones((self.n_variant, ), dtype='bool')  # 8 in standard game
            return np.concatenate((valid_row_col_flat, valid_pieces, valid_orient))

    # @timeit
    def step(self, action):
        # computes a simulation step, given an action and returns observation and info, behaviour depends on action_mode

        if not self.dead[self.active_pl]:
            # decodes action
            if self.action_mode == 'discrete_masked':
                # action, must be a boolean vector with a single element equal to 1, returns (row, col, piece, variant)
                (row, col, p_id, var_id) = np.unravel_index(
                    action, (self.d, self.d, self.n_pieces, self.n_variant))
            elif self.action_mode == 'multi_discrete':
                # action, must be a numpy array containing the four action components (row, col, piece, variant)
                row_col, p_id, var_id = action[0], action[1], action[2]
                row, col = np.unravel_index(row_col, (self.d, self.d))
            # playing board -> padded board
            r_id = row + self.pad
            c_id = col + self.pad
            # saves p_id as last piece placed
            self.last_p_id = p_id

            # POV is cycled in increasing order of player id, e.g. active_player = 2; pl_pov = [2, 3, 0, 1]
            pl_pov = self.active_pl
            # active_pl_from_pov is cycled in decreasing order of player id, e.g. active_player = 2; active_pl_from_pov = [0, 3, 2, 1]
            active_pl_from_pov = 0

            # computes next state from the POV of every player
            for _ in range(self.n_pl):
                self.invalid, self.pad_board[:, :, pl_pov], _, self.player_hands[:, :, pl_pov] = next_state(
                    r_id, c_id, p_id, var_id, self.pad_board[:,
                                                             :, pl_pov], active_pl_from_pov,
                    self.player_hands[:, :, pl_pov], self.n_pl, *self.piece_data)

                if self.invalid:
                    # invalid move played
                    if self.kill_if_invalid_move:
                        # active_player is dead due to an invalid move
                        self.dead[self.active_pl] = True
                    break

                # rotates 90 deg counter-clockwise row and col
                r_id, c_id = rot90_row_col(r_id, c_id, self.d + 2 * self.pad)
                # rotates 90 deg counter-clockwise piece variant
                var_id = self.rot90_variant[var_id]
                # updates pl_pov to next player
                pl_pov = (pl_pov + 1) % self.n_pl
                # when the player POV increases to next player the active player consequently appears to roll back by 1
                active_pl_from_pov = (active_pl_from_pov - 1) % self.n_pl

            # in case of 'discrete_masked' action mode and in case of valid move (with action masking should always be the case, except if a player is dead)
            if self.action_mode == 'discrete_masked' and not self.invalid:
                # updates players' valid actions, the following boolean masks are used in this order:
                #   (1) invalid -> maybe valid (only for the active_player), must be done for each square before the others, to avoid overwriting
                #   (2) action of placed piece -> invalid (only for the active_player, the one who placed the piece)
                #   (3) valid, maybe valid -> invalid (only for the active player)
                #   (4) valid, maybe valid -> invalid (for every player)
                #   (5) valid, maybe valid -> invalid from placement history (for every player, includes the (2), (3), (4) of past moves, and always_invalid)
                # this is done for every square of the piece placed in the last action
                position_square = self.piece_data[0]
                count_pos_squares = self.piece_data[1]
                # extract the playing board coordinates of the square of the places piece
                c_sq = count_pos_squares[p_id, var_id]
                row_squares = position_square[p_id, var_id, 0:c_sq, 0] + row
                col_squares = position_square[p_id, var_id, 0:c_sq, 1] + col
                # (1)
                for row_r, col_r in zip(row_squares, col_squares):
                    # set to True the previously invalid actions where invalid_to_maybe_valid for this [row_r, col_r] is True
                    self.valid_act_mask[self.active_pl, np_and(~self.valid_act_mask[self.active_pl, :],
                                                               self.inv_to_val[row_r, col_r,
                                                                               :])] = True  # (1)
                # (2), (3), (4), (5)
                for row_r, col_r in zip(row_squares, col_squares):
                    for _ in range(self.n_pl):
                        # updates boolean masks for action validity
                        self.update_masks(p_id, pl_pov, row_r, col_r)
                        # rotates 90 deg counter-clockwise row_r and col_r when changing POV
                        row_r, col_r = rot90_row_col(row_r, col_r, self.d)
                        # updates pl_pov to next player
                        pl_pov = (pl_pov + 1) % self.n_pl

                # check if the players are dead
                self.dead = ~np.any(self.valid_act_mask, axis=-1)

        # calculates reward before updating active player and move count
        reward = self._get_reward()
        
        # in case of a dead active player or a valid move
        if self.dead[self.active_pl] or not self.invalid:

            # updates active player and move count
            # 0 -> 1, ..., 3 -> 0
            self.active_pl = (self.active_pl + 1) % self.n_pl
            self.move_count += 1            

            # renders only valid moves or moves of dead players
            if not self.kill_if_invalid_move:
                if self.render_mode == 'human':
                    self._render_frame()

        # get return information after updating the active player
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def update_masks(self, p_id, pl_pov, row_r, col_r):
        # updates the boolean masks used for action validity check given piece, player POV, placed square coordinates
        if pl_pov == self.active_pl:
            # (2)
            self.invalid_history[pl_pov, self.p_flat_id[p_id]] = True
            # (3)
            self.invalid_history[pl_pov, self.val_to_inv_act_pl[row_r, col_r, :]] = True
        # (4)
        self.invalid_history[pl_pov, self.val_to_inv[row_r, col_r, :]] = True
        # (5)
        self.valid_act_mask[pl_pov, self.invalid_history[pl_pov, :]] = False

    # @timeit
    def random_step(self, seed=None):
        # perform a random step checking all possible actions;
        # these actions are generated as the cartesian product (combinations)
        # of masked tiles, pieces and orientations;
        # if no action were valid the player is dead
        
        # action mode must be 'multi_discrete'
        assert self.action_mode in 'multi_discrete'            
        
        if self.dead[self.active_pl]:            
            # if active_pl is dead it does not take an action and is skipped 
            self.active_pl = (self.active_pl + 1) % self.n_pl
            self.move_count += 1
            return

        m = self.action_masks()
        # possible actions
        tile = np.argwhere(m[:400]).tolist()
        pieces = np.argwhere(m[400:421]).tolist()
        orientations = np.argwhere(m[421:429]).tolist()
        # randomly shuffles the arrays, otherwise cartesian product is always the same
        tile = random.sample(tile, len(tile))
        pieces = random.sample(pieces, len(pieces))
        orientations = random.sample(orientations, len(orientations))
        
        self.kill_if_invalid_move = False  # temporary disable killings
        for t, p, o in product(tile, pieces, orientations):
            # calling step method of the current class (not child's methods)
            _, _, _, wrong, _ = __class__.step(self, np.concatenate([t, p, o], dtype=np.int64))
            if not wrong:
                break
        else:
            # not a single valid move is found, thus the active_pl is dead
            self.dead[self.active_pl] = True
            self.active_pl = (self.active_pl + 1) % self.n_pl
            self.move_count += 1
        self.kill_if_invalid_move = True  # enable killings again    
        
        return

    def reset(self, seed=None):
        # resets the environment and returns the first observation and info

        # seeds self.np_random
        super().reset(seed=seed)
        # resets end game flag
        self.end_game = False
        # resets invalid move flag to valid move
        self.invalid = 0
        # resets move count to 0
        self.move_count = 0
        # resets active player
        self.active_pl = int(self.move_count / self.n_pl)
        # resets dead players, resuscitates them if you prefer
        self.dead = np.zeros((self.n_pl, 1), dtype='bool')
        # resets player hands to all pieces available
        self.player_hands = np.ones(
            (self.n_pl, self.n_pieces, self.n_pl), dtype='bool')
        # resets game board with padding
        self.pad_board = np.ones((self.d + 2 * self.pad, self.d + 2 * self.pad, self.n_pl),
                                 dtype='int') * 5  # (unplayable area marked with 5)
        self.pad_board[self.pad:-self.pad, self.pad:-self.pad,
                       :] = 0  # only internal 20 x 20 is playable (marked with 0)
        # player id: 1, 2, 3, 4 starting attachment point (corner outside of 20 x 20 playing board)
        for i in range(self.n_pl):
            # places first attachment points in the corners just outside the playing board (player_color = player_id + 1)
            # start attachment point player 1, for each POV
            self.pad_board[self.pad - 1, self.pad - 1, i] = 1
            # start attachment point player 2, for each POV
            self.pad_board[self.pad - 1, self.d + self.pad, i] = 2
            self.pad_board[
                self.d + self.pad, self.d + self.pad, i] = 3  # start attachment point player 3, for each POV
            # start attachment point player 4, for each POV
            self.pad_board[self.d + self.pad, self.pad - 1, i] = 4
        # id of the last placed piece, mainly used for rewarding
        self.last_p_id = -1

        if self.action_mode == 'discrete_masked':
            # initialize valid starting action for each player, from their POV only (this means they are the same for all players)
            self.valid_act_mask[:, :] = self.val_at_start
            # initialize hystory of invali actions for each player, from their POV only (this means they are the same for all players)
            self.invalid_history[:, :] = self.alw_inv

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info

    def show_piece(self, p_id, var_id):
        # displays the variant var_id of piece p_id
        p_mat = np.zeros((11, 11))
        p_mat[self.piece_data[0][p_id, var_id, :, 0] + 5,
              self.piece_data[0][p_id, var_id, :, 1] + 5] = 2
        p_mat[5, 5] = 1
        plt.matshow(p_mat)

    def show_boards(self, axs, use_given_axis=True, draw_edge=False):
        # displays the board from the POV of each player
        # if used outside of this class axs can be set to [] and use_given_axis to False
        if not use_given_axis:
            px = 1 / plt.rcParams['figure.dpi']
            fig, axs = plt.subplots(2, 2, figsize=(
                self.window_w * px, self.window_h * px))
        else:
            plt.ioff()  # prevents the display of plots

        plt.suptitle('Move count : %d - Active player: %d' % (self.move_count, self.active_pl))
        for i, ax in zip([0, 1, 3, 2], axs.flat):
            # compute current score (without win bonuses)
            placed_pieces_id = np.where(~self.player_hands[i, :, 0])
            count_pos_squares = self.piece_data[1]
            score = np.sum(count_pos_squares[placed_pieces_id, 0])
            # compute placed pieces
            tot_pieces = self.n_pieces - sum(self.player_hands[i, :, 0])
            # plot
            plot_board = self.pad_board[self.pad-1: -
                                        (self.pad-1), self.pad-1: -(self.pad-1), i].copy()
            plot_board[[0, 0, -1, -1], [0, -1, 0, -1]] = 5
            if draw_edge:
                ax.pcolormesh(plot_board, edgecolors=np.array(
                    [222, 217, 210])/255, cmap=self.cmap[i])
                ax.invert_yaxis()
            else:
                bkg = (np.indices(np.shape(plot_board)).sum(axis=0) % 2)*5
                rgb_bkg = [[255, 255, 255, 0], [222, 217, 210, 255]]
                rgb_bkg = np.array(rgb_bkg) / 255
                rgb_bkg[-1, :] = rgb_bkg[-1, :]*0.4 + \
                    np.ones_like(rgb_bkg[-1, :])*0.6
                c_bkg = ListedColormap(rgb_bkg)
                ax.matshow(bkg, cmap=c_bkg)
                ax.matshow(plot_board, cmap=self.cmap[i])
            ax.set_aspect('equal', adjustable='box')
            ax.set_title('Pl %d POV - Score: %d - Pieces: %d - Dead: %d' %
                         (i, score, tot_pieces, self.dead[i]), fontsize=12)
            ax.axis('off')

    def render(self):
        # used to return the pygame rgb array
        if self.render_mode == 'rgb_array':
            return self._render_frame()

    def _render_frame(self):
        # renders a new pygame frame
        if self.window is None and self.render_mode == 'human':
            plt_use('Agg')  # turn off matplotlib gui
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_w, self.window_h))
        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        # prepares boards image using matplotlib
        px = 1 / plt.rcParams['figure.dpi']
        fig, axs = plt.subplots(2, 2, figsize=(
            self.window_w * px, self.window_h * px))
        canvas_plt = FigureCanvas(fig)
        self.show_boards(axs=axs)

        canvas_plt.draw()  # draw the canvas, cache the renderer
        image_flat = np.frombuffer(
            canvas_plt.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
        plt.close('all')  # closes all open figures
        # reversed converts (W, H) from get_width_height to (H, W)
        # (H, W, 3) usable with plt.imshow()
        image = image_flat.reshape(*reversed(canvas_plt.get_width_height()), 3)
        # pygame surface must be transposed (first dimension is horizontal-x, second is vertical-y)
        surface_image = pygame.surfarray.make_surface(image.transpose(1, 0, 2))

        if self.render_mode == 'human':
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(surface_image, surface_image.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # needed to ensure that human-rendering occurs at the predefined framerate
            self.clock.tick(self.metadata['render_fps'])

        else:  # rgb_array
            return image

    def close(self):
        # closes matlab plots and pygame
        plt.close('all')
        plt_use('TkAgg')  # turn on matplotlib gui
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


def rot90_row_col(row, col, d):
    # rotates counterclockwise row and column indexes of a single point in a square matrix of order d
    # returns rotated row, rotated col
    return d - col - 1, row


# @timeit
def next_state(row, col, p_id, var_id, padded_board, player_id, player_hands, n_players,
               position_square, count_pos_squares, position_attach, count_pos_attach, position_forbid,
               count_pos_forbid):
    """
    :param row: [5 - 24] row of the piece origin in the playing board
    :param col: [5 - 24] column of the piece origin in the playing board
    :param p_id: [0 - 20] id of the blokus piece, see 'Blokus Pieces.xlsx'
    :param var_id: [0 - 7] id of the blokus piece variation, see 'Blokus Pieces.xlsx'
    :param padded_board: 30 x 30 numpy array of the playing board, including padding
    :param player_id: [0 - 3] id of the current player
    :param player_hands: 4 x 21 numpy array of the pieces in hand, boolean
    :param n_players: number of players, 4 by default
    :param position_square: see preprocessing_id.py
    :param count_pos_squares: see preprocessing_id.py
    :param position_attach: see preprocessing_id.py
    :param count_pos_attach: see preprocessing_id.py
    :param position_forbid: see preprocessing_id.py
    :param count_pos_forbid: see preprocessing_id.py
    :return: invalid, padded_board, player_id, player_hands
    invalid: is > 0 if the move is invalid move, 0 for valid moves
    padded_board: updated board
    player_id:updated active player
    player_hands: updated player hands
    """

    # input parameters with typical ranges:
    #   count_pos_forbid:   see preprocessing_id.py
    #
    # output parameters:
    #   invalid:            is > 0 if the move is invalid move, 0 for valid moves
    #   padded_board:       updated board
    #   player_id:          updated active player
    #   player_hands:       updated player hands
    #
    # returns same board, same player_id, same player_hands if move is not valid,
    # otherwise returns updated board, updated player_id, updated player_hands

    # checks if piece is available
    if player_hands[player_id, p_id] == 0:
        # if piece already has been used -> invalid move
        return 4, padded_board, player_id, player_hands

    # checks overlap with other pieces
    # occurs if at least one of the squares of the piece is different than zero
    c_sq = count_pos_squares[p_id, var_id]
    row_squares = position_square[p_id, var_id, 0:c_sq, 0]
    col_squares = position_square[p_id, var_id, 0:c_sq, 1]
    if np.any(padded_board[row + row_squares, col + col_squares]):
        # if there is overlap (any non-zero) -> invalid move
        return 3, padded_board, player_id, player_hands

    # checks attachment
    # occurs if no attachment point overlap with pieces of the current player
    c_att = count_pos_attach[p_id, var_id]
    row_attach = position_attach[p_id, var_id, 0:c_att, 0]
    col_attach = position_attach[p_id, var_id, 0:c_att, 1]
    if not np.any(padded_board[row + row_attach, col + col_attach] == (player_id + 1)):
        # if no attachment points match with player_id squares -> invalid move
        return 2, padded_board, player_id, player_hands

    # checks adjacency with other current player pieces in forbidden zones
    # occurst if at least one of the forbidden squares of the piece overlap with pieces of the current player
    c_forb = count_pos_forbid[p_id, var_id]
    row_forbid = position_forbid[p_id, var_id, 0:c_forb, 0]
    col_forbid = position_forbid[p_id, var_id, 0:c_forb, 1]
    if np.any(padded_board[row + row_forbid, col + col_forbid] == (player_id + 1)):
        # if any forbidden zone corresponds to current player pieces -> invalid move
        return 1, padded_board, player_id, player_hands

    # if the move is valid...
    # place block and update board
    padded_board[row + row_squares, col + col_squares] = player_id + 1
    # update hand
    player_hands[player_id, p_id] = 0
    # update active player
    player_id = (player_id + 1) % n_players  # 0 -> 1, ..., 3 -> 0

    return 0, padded_board, player_id, player_hands

