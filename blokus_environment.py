import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from preprocessing_id import preprocess_id
from matplotlib import use as plt_use
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# TODO: player_id to player_color and viceversa

class BlokusEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 8}
    
    def __init__(self, render_mode=None, d_board=20, win_width=640*2, win_height=480*2):
        
        # computes the blokus pieces data
        self.piece_data = preprocess_id()
        
        # constants
        self.pad = 5 # pentominos need extra 5 spaces over the board for generic placement
        self.d = d_board # playing board edge dimension
        self.n_pieces = 21 # number of blokus pieces for each player
        self.n_variant = 8 # number of vairant for each piece, 4 rotation times 2 flip states by default
        self.n_pl = 4 # number of players during a game, by default is always 4
        self.rot90_variant = [1, 2, 3, 0, 5, 6, 7, 4] # next variant when rotating 90 deg counter-clockwise, see preprocess_id.py
        
        # resettable
        # invalid last move [0 - 4], valid move = 0, invalid move = [1 - 3], see next_state for details
        self.invalid = 0
        # number of moves completed, counting all players
        self.move_count = 0
        # active player number, can be 0,1,2,3 with the default 4 players
        self.active_pl = int(self.move_count/self.n_pl)
        # player hand, 4 x 21 boolean array (1 = in hand, 0 = already placed), as seen by the 4 player
        self.player_hands = np.ones((self.n_pl, self.n_pieces, self.n_pl), dtype='bool')
        # game board, with padding = 5 to avoid checking for out of bounds exceptions, one for each player
        self.padded_board = np.ones((self.d + 2*self.pad, self.d + 2*self.pad, self.n_pl), dtype=int)
        
        # observation:
        #   board is the playing board where 0 = empty and 1-4 = player square
        #   hands represent the available pieces in the hands of every player
        #   turn is the turn number of the current player, derives from move_count
        self.observation_space = spaces.Dict(
            {
                'board': spaces.Box(0, 4, (self.d, self.d), dtype=int),
                'hands': spaces.MultiBinary((self.n_pl, self.n_pieces)),
                'turn': spaces.Box(0, self.n_pieces - 1, dtype=int),
                'invalid': spaces.Box(0, 4, dtype=int)
            }
        )
        
        # action:
        # array representation of a multidimensional space
        #   row position of the origin of the piece to play [0-d]
        #   column position of the origin of the piece to play [0-d]
        #   piece to play [0-n]
        #   variant of the piece to play [0-7]
        self.action_dim = self.d*self.d*self.n_pieces*self.n_variant
        self.action_space = spaces.MultiBinary((self.action_dim,))
        
        # pygame and rendering attributes
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_w = win_width # pygame window width
        self.window_h = win_height # pygame window height
        # white, red, green, blue, yellow, grey from: https://service.mattel.com/instruction_sheets/BJV44-Eng.pdf
        rgb_col = [[255,255,255], [215,29,36], [0,161,75], [0,90,170], [253,184,19], [222,217,210]]
        rgb_col = np.array(rgb_col)/255
        self.cmap = [ListedColormap(np.vstack((rgb_col[0,:], np.roll(rgb_col[1:self.n_pl+1], -i, axis=0), rgb_col[-1,:]))) for i in range(self.n_pl)]
        
        
    def _get_obs(self):
        # return the game board and hands, as seen by the active_player
        return {'board': self.padded_board[self.pad : self.d+self.pad, self.pad : self.d+self.pad, self.active_pl],
                'hands': self.player_hands[:,:, self.active_pl],
                'turn': int(self.move_count/self.n_pl),
                'invalid': self.invalid
            }
        
    
    def _get_info(self):        
        # return some additional game state information
        return {'move_count': self.move_count}
        
    
    def step(self, action):
        # computes a simulation step, given an action and returns observation and info
        
        # decode action, must be a boolean vector with a single element equal to 1, returns (row, col, piece, variant)
        (row, col, p_id, var_id) = np.unravel_index(np.argmax(action), (self.d, self.d, self.n_pieces, self.n_variant))
        r_id = row + self.pad
        c_id = col + self.pad        
        
        # POV is cycled in increasing order of player id, e.g. active_player = 2; pl_pov = [2, 3, 0, 1]
        pl_pov = self.active_pl
        # active_pl_from_pov is cycled in decrasing oreder of player id, e.g. active_player = 2; active_pl_from_pov = [0, 3, 2, 1]
        active_pl_from_pov = 0
        
        first_valid = False
        # compute next state from the POV of every player
        for _ in range(self.n_pl):
            self.invalid, self.padded_board[:,:,pl_pov], next_pl, self.player_hands[:,:,pl_pov] = next_state(
                    r_id, c_id, p_id, var_id, self.padded_board[:,:,pl_pov], active_pl_from_pov, self.player_hands[:,:,pl_pov], self.n_pl, *self.piece_data)
            if self.invalid:
                if first_valid:
                    print('PROBLEM')
                    pass
                break # invalid move, should be negatively rewarded
            else:
                first_valid = True
                # print("HERE")                
                # self.show_piece(p_id, var_id)
                # self.show_boards()
                # plt.show()
                # plt.close('all')
                pass 
            
            # rotate 90 deg counter-clockwise row and col
            r_id, c_id = rot90_row_col(r_id, c_id, self.d + 2*self.pad)
            # rotate 90 deg counter-clockwise piece variant
            var_id = self.rot90_variant[var_id]
            # update pl_pov to next player
            pl_pov = (pl_pov + 1) % self.n_pl
            # when the player POV increases to next player the active player consequently appear to roll back by 1
            active_pl_from_pov = (active_pl_from_pov - 1) % self.n_pl
         
        if not self.invalid: 
            # if action is admissible update active and move count
            self.active_pl = (self.active_pl + 1) % self.n_pl # 0 -> 1, ..., 3 -> 0
            self.move_count += 1
            # render only valid moves
            if self.render_mode == 'human':
                self._render_frame()
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    
    def reset(self, seed=None):
        # resets the environment and returns the first observation and info
                
        # seeds self.np_random
        super().reset(seed=seed)        
        # reset invalid move flag to valid move
        self.invalid = 0
        # reset move count to 0
        self.move_count = 0 
        # reset active player
        self.active_pl = int(self.move_count/self.n_pl)
        # reset player hands to all pieces available
        self.player_hands = np.ones((self.n_pl, self.n_pieces, self.n_pl), dtype='bool')
        # reset game board with padding
        self.padded_board = np.ones((self.d + 2*self.pad, self.d + 2*self.pad, self.n_pl), dtype=int)*5 # (unplayable area marked with 5)
        self.padded_board[self.pad:-self.pad,self.pad:-self.pad,:] = 0 # only internal 20 x 20 is playable (marked with 0)
        # player id: 1, 2, 3, 4 starting attachment point (corner outside of 20 x 20 playing board)      
        for i in range(self.n_pl):
            # place first attachment points in the corners just outside the playing board (player_color = player_id + 1)
            self.padded_board[self.pad-1,       self.pad-1,         i] = 1 # start attachment point player 1, for each POV
            self.padded_board[self.pad-1,       self.d+self.pad,    i] = 2 # start attachment point player 2, for each POV
            self.padded_board[self.d+self.pad,  self.d+self.pad,    i] = 3 # start attachment point player 3, for each POV
            self.padded_board[self.d+self.pad,  self.pad-1,         i] = 4 # start attachment point player 4, for each POV

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()

        return observation, info
    
    
    def show_piece(self, p_id, var_id):
        # display the variant var_id of piece p_id
        p_mat = np.zeros((11,11))
        p_mat[self.piece_data[0][p_id, var_id, :, 0] + 5, self.piece_data[0][p_id, var_id, :, 1] + 5] = 2
        p_mat[5,5] = 1
        plt.matshow(p_mat)
        
    
    def show_boards(self, axs, use_given_axis=True):
        # display the board from the POV of each player
        # if used outside of this class axs can be set to [] and use_given_axis to False
        if not use_given_axis:
            px = 1/plt.rcParams['figure.dpi'] 
            fig, axs = plt.subplots(2,2, figsize=(self.window_w*px, self.window_h*px))
        else:            
            plt.ioff() # prevents the display of plots
        
        for i, ax in zip([0, 1, 3, 2], axs.flat):
            ax.matshow(self.padded_board[:,:,i], cmap=self.cmap[i])
            ax.set_title('Player %d POV' % i, fontsize=12)
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
                (self.window_w, self.window_h)
            )
        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        # prepare boards image using matplotlib
        px = 1/plt.rcParams['figure.dpi']
        fig, axs = plt.subplots(2,2, figsize=(self.window_w*px, self.window_h*px))
        canvas_plt = FigureCanvas(fig)
        self.show_boards(axs=axs)

        canvas_plt.draw()  # draw the canvas, cache the renderer
        image_flat = np.frombuffer(canvas_plt.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
        # reversed converts (W, H) from get_width_height to (H, W)
        image = image_flat.reshape(*reversed(canvas_plt.get_width_height()), 3)  # (H, W, 3) usable with plt.imshow()
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
        plt_use('TkAgg') # turn on matplotlib gui
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        
        
def rot90_row_col(row, col, d):       
    # rotate counterclockwise row and column indexes of a single point in a square matrix of order d
    # returns rotated row, rotated col
    return d-col-1, row


def next_state(row, col, p_id, p_var, padded_board, player_id, player_hands, n_players,
               position_square, count_pos_squares, position_attach, count_pos_attach, position_forbid, count_pos_forbid):
    
    # input parameters with typical ranges:
    #   row:                [5 - 24] row of the piece origin in the playing board 
    #   col:                [5 - 24] column of the piece origin in the playing board 
    #   p_id:               [0 - 20] id of the blokus piece, see 'Blokus Pieces.xlsx'
    #   p_var:              [0 - 7] id of the blokus piece variation, see 'Blokus Pieces.xlsx'
    #   padded_board:       30 x 30 numpy array of the playing board, including padding
    #   player_id:          [0 - 3] id of the current player
    #   player_hands:       4 x 21 numpy array of the pieces in hand, boolean
    #   n_players:          number of players, 4 by default
    #   position_square:    see preprocessing_id.py
    #   count_pos_squares:  see preprocessing_id.py
    #   position_attach:    see preprocessing_id.py
    #   count_pos_attach:   see preprocessing_id.py
    #   position_forbid:    see preprocessing_id.py
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
    
    # check if piece is available
    if player_hands[player_id, p_id] == 0:
        # if piece already has been used -> invalid move
        return 4, padded_board, player_id, player_hands
    
    # check overlap with other pieces
    c_sq = count_pos_squares[p_id, p_var]
    row_squares = position_square[p_id, p_var, 0:c_sq, 0]
    col_squares = position_square[p_id, p_var, 0:c_sq, 1]
    if np.any(padded_board[row + row_squares, col + col_squares]):
        # if there is overlap (any non-zero) -> invalid move
        return 3, padded_board, player_id, player_hands
    
    # check attachment
    c_att = count_pos_attach[p_id, p_var]
    row_attach = position_attach[p_id, p_var, 0:c_att, 0]
    col_attach = position_attach[p_id, p_var, 0:c_att, 1]
    if not np.any(padded_board[row + row_attach, col + col_attach] == (player_id+1)):
        # if no attachment points match with player_id squares -> invalid move
        return 2, padded_board, player_id, player_hands
    
    # check adjacency with other current player pieces in forbidden zones
    c_forb = count_pos_forbid[p_id, p_var]
    row_forbid = position_forbid[p_id, p_var, 0:c_forb, 0]
    col_forbid = position_forbid[p_id, p_var, 0:c_forb, 1]
    if np.any(padded_board[row + row_forbid, col + col_forbid] == (player_id+1)):
        # if any forbidden zone corresponds to current player pieces -> invalid move
        return 1, padded_board, player_id, player_hands
    
    # if the move is valid...
    # place block and update board
    padded_board[row + row_squares, col + col_squares] = player_id+1
    # update hand
    player_hands[player_id, p_id] = 0
    # update active player
    player_id = (player_id + 1) % n_players # 0 -> 1, ..., 3 -> 0
        
    return 0, padded_board, player_id, player_hands