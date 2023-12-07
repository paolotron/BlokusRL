from environments import blokus_environment as be
import numpy as np
import matplotlib.pyplot as plt
import time as tm

pass
def main():
    # ----- INPUT -----
    human_mode = 0 # 1 for rendering and 0 for no rendering
    # plays n_iter moves
    n_iter = int(1e5)
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

    # initializes the environment
    if human_mode:
        blokus_game = be.BlokusEnv(render_mode='human', d_board=d_board, win_width=win_width, win_height=win_height)
    else:
        blokus_game = be.BlokusEnv(render_mode=None, d_board=d_board)

# boolean mask of the action space for each player, indicating valid actions from the active_player's POV
valid_act_mask = np.zeros((blokus_game.n_pl, blokus_game.action_dim), dtype='bool')
# player action
action = np.zeros((blokus_game.action_dim,), dtype='bool')

# resets the environment
obs, info = blokus_game.reset(seed=np_seed)
valid_masks = info['valid_masks'] # boolean mask of each player's valid action
active_pl = info['active_player'] # active player

state_list = [] # list of action validity
n_valid = np.zeros((n_iter,5)) # counter for invalid actions
t_0 = tm.time() # for timing

for i in range(n_iter):
    
    # admissible actions ids
    act_id = np.where(valid_masks[active_pl,:] == True)[0]
    # number of admissible actions
    n_act = len(act_id)
    # random admissible action
    adm_a_id = np.random.randint(0, n_act)
    # actual action id
    a_id = act_id[adm_a_id]
    # action array (all elements are False except for a_id)
    action[a_id] = 1
    
    # simulation step
    obs, info = blokus_game.step(action)
    valid_masks = info['valid_masks'] # boolean mask of each player's valid action
    active_pl = info['active_player'] # active player
    
    # resetting action
    action[a_id] = 0
    
    # saving state
    state = obs['invalid']
    state_list.append(state)
    # saving validity counters
    if i != 0:
        n_valid[i,:] = n_valid[i-1,:]
        n_valid[i,state] += 1
    else:
        n_valid[i,state] = 1
    
# time estimation
elapsed = tm.time() - t_0
print('Elapsed time for %d iterations: %f s\n' % (n_iter, elapsed))
print('Average time for single iteration: %f ms' % (1000*elapsed/n_iter))

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