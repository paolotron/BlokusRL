from environments.blokus_environment import BlokusEnv

class SingleplayerBlokusEnv(BlokusEnv):

    def __init__(self, *args, **kwargs):
        # only 'multi_discrete' allowed
        super().__init__(*args, **kwargs)
        assert self.action_mode == 'multi_discrete'

    def step(self, action):
        output = super().step(action)

        # render frame here due to 'kill_if_invalid_move' set to False
        if self.render_mode == 'human':
            self._render_frame()

        # rolls back active player
        self.active_pl = (self.active_pl - 1) % self.n_pl
        return output
        
        
    def reset(self, seed=None):
        output = super().reset(seed)
        return output
