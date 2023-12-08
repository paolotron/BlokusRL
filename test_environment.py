import time as tm
t_0 = tm.time() # for timing
from environments import blokus_environment as be
import numpy as np
import matplotlib.pyplot as plt
import time as tm
# time estimation
elapsed = tm.time() - t_0
print('Elapsed time for import: %f s\n' % elapsed)

def main():
    
    # ----- INPUT -----
    
    # 1 for human rendering and 0 for no rendering
    human_mode = 0
    # number of consecutive games to play
    n_games = 1
    # square playing board dimension
    d_board = 20
    # rendering window dimension (px)
    win_width = 640*2
    win_height = 480*2
    # numpy random seed for random number generator
    np_seed = 0

    # ----- MAIN -----

    # seeds the numpy random number generator
    np.random.seed(np_seed)

    t_0 = tm.time() # for timing
    # initializes the environment
    if human_mode:
        blokus_game = be.BlokusEnv(render_mode='human', d_board=d_board, win_width=win_width, win_height=win_height)
    else:
        blokus_game = be.BlokusEnv(render_mode=None, d_board=d_board)
    # time estimation
    elapsed = tm.time() - t_0
    print('Elapsed time for init: %f s\n' % elapsed)

    for _ in range(n_games):
        
        t_0 = tm.time() # for timing
        # resets the environment
        obs, info = blokus_game.reset(seed=np_seed)
        # time estimation
        elapsed = tm.time() - t_0
        print('Elapsed time for reset: %f s\n' % elapsed)
        
        valid_masks = info['valid_masks'] # boolean mask of each player's valid action
        active_pl = info['active_player'] # active player

        state_list = [] # list of action validity
        # max iterations
        n_iter = int(blokus_game.n_pieces*blokus_game.n_pl)
        n_valid = np.zeros((n_iter, 6)) # counter for invalid actions
        t_0 = tm.time() # for timing
        term = False # terminated condition

        for i in range(n_iter*4):
            
            # admissible actions ids
            act_id = np.where(valid_masks[active_pl,:] == True)[0]
            # number of admissible actions
            n_act = len(act_id)
            if n_act > 0:
                # random admissible action
                adm_a_id = np.random.randint(0, n_act)
                # actual action id
                action = act_id[adm_a_id]
            else:
                action = None
            
            # simulation step
            obs, rew, term, trunc, info = blokus_game.step(action)
            valid_masks = info['valid_masks'] # boolean mask of each player's valid action
            active_pl = info['active_player'] # active player
            
            # saving state
            state = obs['invalid']
            state_list.append(state)
            # saving validity counters
            if i != 0:
                n_valid[i,:] = n_valid[i-1,:]
                n_valid[i,state] += 1
            else:
                n_valid[i,state] = 1
            
            # exit condition
            if term:
                break
            
        # time estimation
        elapsed = tm.time() - t_0
        print('Elapsed time for %d iterations: %f s\n' % (i, elapsed))
        print('Average time for single iteration: %f ms' % (1000*elapsed/i))

    # closes environment
    blokus_game.close()

    # ----- POST-PROCESSING -----

    # state plot
    plt.figure()
    plt.plot(state_list,'o')
    plt.title('Move validity plot')

    # move outcome percentage plot
    plt.figure()
    lbl = ['valid', 'adjacent edges', 'no corner connected', 'overlapping', 'already used']
    n_step = np.arange(0,n_iter,1)
    for i in range(5):
        plt.plot(n_step, 100*n_valid[:,i]/(n_step+1), '-', label=lbl[i])
    plt.xlabel('Step number')
    plt.ylabel(r'% of step outcome')
    plt.legend()
    plt.grid(True)

    # move outcome cumulative plot
    plt.figure()
    lbl = ['valid', 'adjacent edges', 'no corner connected', 'overlapping', 'already used']
    n_step = np.arange(0,n_iter,1)
    for i in range(5):
        plt.plot(n_step, n_valid[:,i], '-', label=lbl[i])
    plt.xlabel('Step number')
    plt.ylabel(r'Number of step outcome')
    plt.legend()
    plt.grid(True)

    # final state, from each POV
    blokus_game.show_boards([], False)

    plt.show()


if __name__ == '__main__':
    main()