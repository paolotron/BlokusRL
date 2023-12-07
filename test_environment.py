from environments import blokus_environment as be
import numpy as np
from random import randrange
import matplotlib.pyplot as plt
import time as tm

pass
def main():
    # ----- INPUT -----
    human_mode = 0 # 1 for rendering and 0 for no rendering
    # play n_iter moves
    n_iter = int(1e5)
    # square playing board dimension
    d_board = 20
    # rnedering window dimension (px)
    win_width = 640*2
    win_height = 480*2

    # ----- MAIN -----

    # initializes the environment
    if human_mode:
        blokus_game = be.BlokusEnv(render_mode='human', d_board=d_board, win_width=win_width, win_height=win_height)
    else:
        blokus_game = be.BlokusEnv(render_mode=None, d_board=d_board)

    # resets the environment
    obs, info = blokus_game.reset()

    state_list = []
    valid_p = 0
    n_valid = np.zeros((n_iter,5))
    t_0 = tm.time()
    action = np.zeros((blokus_game.action_dim), dtype='bool')
    for i in range(n_iter):

        # random action
        a_id = randrange(0, blokus_game.action_dim)
        action[a_id] = 1
        # simulation step
        obs, info = blokus_game.step(action)
        # resetting action
        action[a_id] = 0

        # saving state
        state = obs['invalid']
        state_list.append(state)
        # saving validity %
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