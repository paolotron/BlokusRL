import matplotlib.pyplot as plt
import numpy as np
from environments import blokus_environment as be
import time as tm
t_0 = tm.time()  # for timing
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
    # action mode ('discrete_masked' or 'multi_discrete')
    action_mode = 'discrete_masked'
    # action_mode = 'multi_discrete'

    # ----- MAIN -----

    # seeds the numpy random number generator
    # np.random.seed(np_seed)

    t_0 = tm.time()  # for timing
    # initializes the environment
    if human_mode:
        blokus_game = be.BlokusEnv(render_mode='human', action_mode=action_mode,
                                   d_board=d_board, win_width=win_width, win_height=win_height)
    else:
        blokus_game = be.BlokusEnv(
            render_mode=None, action_mode=action_mode, d_board=d_board)
    # time estimation
    elapsed = tm.time() - t_0
    print('Elapsed time for init: %f s\n' % elapsed)

    for _ in range(n_games):

        t_0 = tm.time()  # for timing
        # resets the environment
        obs, info = blokus_game.reset(seed=np_seed)
        # time estimation
        elapsed = tm.time() - t_0
        print('Elapsed time for reset: %f s\n' % elapsed)

        # boolean mask of each player's valid action
        valid_masks = info['valid_masks']
        active_pl = info['active_player']  # active player

        # max iterations
        if action_mode == 'discrete_masked':
            n_iter = int(blokus_game.n_pieces*blokus_game.n_pl)
        elif action_mode == 'multi_discrete':
            n_iter = 1000
            state_list = []  # list of action validity
            n_valid = np.zeros((n_iter, 6))  # counter for invalid actions

        t_0 = tm.time()  # for timing
        term = False  # terminated condition
        rew_tot = np.zeros((int(n_iter/4), 4))  # reward sum

        for i in range(n_iter):

            if action_mode == 'discrete_masked':
                # admissible actions ids
                act_id = np.where(valid_masks[active_pl, :] == True)[0]
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
                
                # boolean mask of each player's valid action
                valid_masks = info['valid_masks']
                active_pl = info['active_player']  # active player

                # update reward
                if i < 4:
                    rew_tot[int(i/4), (active_pl - 1) % 4] = rew
                else:
                    rew_tot[int(i/4), (active_pl - 1) %
                            4] = rew_tot[int(i/4) - 1, (active_pl - 1) % 4] + rew

                # exit condition
                if term:
                    print('Last player: ',  (active_pl - 1) %
                        4, '    Last Reward: ', rew)
                    rew_tot[int(i/4), rew_tot[int(i/4), :] ==
                            0] = rew_tot[int(i/4) - 1, rew_tot[int(i/4), :] == 0]
                    break

            elif action_mode == 'multi_discrete':
                # totally random action
                a_mask = np.int8(blokus_game.action_masks())
                mask = (a_mask[0:d_board**2], a_mask[d_board**2:d_board**2 + 21], a_mask[d_board**2 + 21:])
                action = blokus_game.action_space.sample(mask=mask)
                # resuscitate dead players
                # blokus_game.dead[blokus_game.active_pl] = False

                # simulation step
                blokus_game.random_step()
                
                if blokus_game.all_dead():
                    break
             

        # time estimation
        elapsed = tm.time() - t_0
        print('Elapsed time for %d iterations: %f s\n' % (i, elapsed))
        print('Average time for single iteration: %f ms' % (1000*elapsed/i))

    # closes environment
    blokus_game.close()

    # ----- POST-PROCESSING -----

    # final state, from each POV
    blokus_game.show_boards([], False, draw_edge=True)
    blokus_game.show_boards([], False, draw_edge=False)

    plt.figure()
    lbl = ['pl 0', 'pl 1', 'pl 2', 'pl 3']
    n_step = np.arange(0, int(i/4) + 1, 1)
    for p in range(4):
        plt.plot(n_step, rew_tot[:int(i/4) + 1, p], '-',
                 label=lbl[p], color=blokus_game.rgb_col[p+1, :])
    plt.xlabel('Step number')
    plt.ylabel('Reward cumulative')
    plt.legend()
    plt.title('Cumulative reward for 4 random player agents')
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    main()
