import os
import openpyxl as xl
import numpy as np
import pickle as pk


def preprocess_id():
    
    # must be run after preprocessing_excel.py
    # calculates indexes of pieces position, relative to their origin
    #   - indexes of piece squares
    #   - indexes of attachment points
    #   - indexes of forbidden positions
    
    if not os.path.isfile('environments\pieces_data.pkl'):
        
        if not os.path.isfile('environments\Blokus_Pieces.xlsx'):
            print('Error: file named Blokus_Pieces.xlsx cannot be found')
            return None
        
        wb = xl.load_workbook('environments\Blokus_Pieces.xlsx', data_only=True) # 
        ws = wb['Pieces'] # worksheet
        d_cell = 11 # dimension of excel cell used
        n_pieces = 21 # number of blokus pieces (standard game)
        n_variant = 8 # max number of flip and rotation for each piece
        # squares relative position coord: n_pieces x n_variant x 5 (max number of square for piece) x 2 (row, column)
        # row = 0, col = 0 is the relative coord of the origin square, always defined for every block
        position_square = np.zeros((n_pieces, n_variant, 5, 2), dtype=np.int8)
        count_pos_squares = np.zeros((n_pieces, n_variant), dtype=np.int8) # counter of square coord, used for indexing

        # attachment squares relative position coord: n_pieces x n_variant x 8 (max number of attachment squares, cross piece) x 2 (row, column)
        position_attach = np.zeros((n_pieces, n_variant, 8, 2), dtype=np.int8)
        count_pos_attach = np.zeros((n_pieces, n_variant), dtype=np.int8) # counter of attachment coord, used for indexing
        
        # forbidden squares relative position coord: n_pieces x n_variant x 12 (max number of attachment squares, long straight piece) x 2 (row, column)
        position_forbid = np.zeros((n_pieces, n_variant, 12, 2), dtype=np.int8)
        count_pos_forbid = np.zeros((n_pieces, n_variant), dtype=np.int8) # counter of forbidden coord, used for indexing

        for p_id in range(n_pieces):
            for v_id in range(n_variant):
                
                start_xl_col = 2 + p_id*d_cell # starting column on excel
                start_xl_row = 2 + v_id*d_cell # starting row on excel
                
                # read cells
                curr_mat = np.zeros((d_cell, d_cell))
                for i in range(d_cell): # rows
                    for j in range(d_cell): # columns
                        curr_mat[i,j] = ws.cell(row=start_xl_row+i , column=start_xl_col+j).value    
                        # center of the matrix "curr_mat" is in [5,5] and always contains the origin of the piece
                        rel_row = i - 5
                        rel_col = j - 5
                            
                        if curr_mat[i,j] == 1 or curr_mat[i,j] == 2: # 1 = origin square of piece, 2 = other squares
                            count = count_pos_squares[p_id, v_id]
                            # updating the position_square array
                            position_square[p_id, v_id, count, 0] = rel_row
                            position_square[p_id, v_id, count, 1] = rel_col
                            # updating counter
                            count_pos_squares[p_id, v_id] = count + 1
                            
                        elif curr_mat[i,j] == 5: # 5 = attachment points
                            count = count_pos_attach[p_id, v_id]
                            # updating the position_attach array
                            position_attach[p_id, v_id, count, 0] = rel_row
                            position_attach[p_id, v_id, count, 1] = rel_col
                            # updating counter
                            count_pos_attach[p_id, v_id] = count + 1
                        
                        elif curr_mat[i,j] == 6: # 6 = forbidden squares
                            count = count_pos_forbid[p_id, v_id]
                            # updating the position_attach array
                            position_forbid[p_id, v_id, count, 0] = rel_row
                            position_forbid[p_id, v_id, count, 1] = rel_col
                            # updating counter
                            count_pos_forbid[p_id, v_id] = count + 1
        
        # saving pieces data using pickle
        piece_data_tuple = (position_square, count_pos_squares, position_attach, count_pos_attach, position_forbid, count_pos_forbid)
        with open('environments\pieces_data.pkl', 'wb') as handle:
            pk.dump(piece_data_tuple, handle)
        pass
    
    else:
        # load pieces data
        with open('environments\pieces_data.pkl', 'rb') as handle:
            piece_data_tuple = pk.load(handle)
    
    # return pieces data
    return piece_data_tuple
        