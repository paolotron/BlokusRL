import os
import pickle as pk
import numpy as np


def preprocess_action(padded_board, pad, n_p, n_v,
        position_square, count_pos_squares, position_attach, count_pos_attach, position_forbid, count_pos_forbid):
    
    # preprocess all the possible action, each square placed inside the playing board can:
    #   - turn an invalid action of the active player from invalid to potentially valid
    #   - turn an action of each player to invalid    
        
    d_row = np.shape(padded_board)[0] - 2*pad
    d_col = np.shape(padded_board)[1] - 2*pad
    
    # boolean mask of all of the 67.2k actions (normal game) indicating actions valid on first turn of every player
    valid_at_start = np.zeros((d_row*d_col*n_p*n_v), dtype='bool') 
    # to be applied in each pos when a new squares is placed there, 26.88M values (normal game), only for active_player:
    invalid_to_maybe_valid = np.zeros((d_row, d_col, d_row*d_col*n_p*n_v), dtype='bool') 
    # to be applied in each pos when a new squares is placed there, 26.88M values (normal game):
    valid_to_invalid = np.zeros((d_row, d_col, d_row*d_col*n_p*n_v), dtype='bool')
    # to be applied in each pos when a new squares is placed there, 26.88M values (normal game), only for active_player::
    valid_to_invalid_act_pl = np.zeros((d_row, d_col, d_row*d_col*n_p*n_v), dtype='bool')
    # boolean mask of all of the 67.2k actions (normal game) indicating actions always invalid, to be applied after valid_to_invalid
    always_invalid = np.zeros((d_row*d_col*n_p*n_v), dtype='bool')
    
    # place a single piece in every one of the 67.2k board (normal game)
    for a_id in range(len(valid_at_start)):
        
        # decode action, must be a boolean vector with a single element equal to 1, returns (row, col, piece, variant)
        (r_id, c_id, p_id, var_id) = np.unravel_index(a_id, (d_row, d_col, n_p, n_v))
        
        # block squares (marked with 1)
        c_sq = count_pos_squares[p_id, var_id]
        row_squares = position_square[p_id, var_id, 0:c_sq, 0] + r_id
        col_squares = position_square[p_id, var_id, 0:c_sq, 1] + c_id
        # check if piece squares go out of bounds -> action always invalid
        row_out_bounds = np.logical_or(row_squares < 0, row_squares >= d_row)
        col_out_bounds = np.logical_or(col_squares < 0, col_squares >= d_col)
            
        if np.any(np.logical_or(row_out_bounds, col_out_bounds)):
            # if action is always invalid it has no effect on invalid_to_maybe_valid, valid_to_invalid
            # TODO: dictionary from valid action to (r_id, c_id, p_id, var_id), with reduced dimension of action space
            always_invalid[a_id] = True
        else:
            
            # block attachment points (marked with 5), for invalid -> potentially valid actions
            c_att = count_pos_attach[p_id, var_id]
            row_attach = position_attach[p_id, var_id, 0:c_att, 0] + r_id
            col_attach = position_attach[p_id, var_id, 0:c_att, 1] + c_id
            # keep only positions inside playing board
            id_keep = np.logical_and(np.logical_and(row_attach >= 0, row_attach < d_row), np.logical_and(col_attach >= 0, col_attach < d_col))
            invalid_to_maybe_valid[row_attach[id_keep], col_attach[id_keep], a_id] = True
            
            # block squares (marked with 1), for valid -> invalid actions
            c_sq = count_pos_squares[p_id, var_id]
            row_squares = position_square[p_id, var_id, 0:c_sq, 0] + r_id
            col_squares = position_square[p_id, var_id, 0:c_sq, 1] + c_id
            # keep only positions inside playing board
            id_keep = np.logical_and(np.logical_and(row_squares >= 0, row_squares < d_row), np.logical_and(col_squares >= 0, col_squares < d_col))
            valid_to_invalid[row_squares[id_keep], col_squares[id_keep], a_id] = True
            
            # block forbidden positions (marked with 6), for valid -> invalid actions
            c_forb = count_pos_forbid[p_id, var_id]
            row_forbid = position_forbid[p_id, var_id, 0:c_forb, 0] + r_id
            col_forbid = position_forbid[p_id, var_id, 0:c_forb, 1] + c_id
            # keep only positions inside playing board
            id_keep = np.logical_and(np.logical_and(row_forbid >= 0, row_forbid < d_row), np.logical_and(col_forbid >= 0, col_forbid < d_col))                
            valid_to_invalid_act_pl[row_forbid[id_keep], col_forbid[id_keep], a_id] = True
            
            # check if is a valid starting action:
            # - must not be an always_invalid action
            # - must have an attachment point in [-1, -1] in row_attach, col_attach which corresponds to
            #       the top left corner, starting attachment point of active_player in the playing board (not padded)
            if np.any(np.logical_and(row_attach == -1, col_attach == -1)):
                valid_at_start[a_id] = True
                
    return always_invalid, invalid_to_maybe_valid, valid_to_invalid, valid_to_invalid_act_pl, valid_at_start
                
"""
    if not os.path.isfile('actions_data.pkl'):      
    
    # code here
              
        with open('actions_data.pkl', 'wb') as handle:
            pk.dump((always_invalid, invalid_to_maybe_valid, valid_to_invalid, valid_at_start), handle)
        pass
    
    else:
        # load pieces data
        with open('actions_data.pkl', 'rb') as handle:
            (always_invalid, invalid_to_maybe_valid, valid_to_invalid, valid_at_start) = pk.load(handle)
    
"""


