# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import pickle
import torch
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
# from policy_value_net_numpy import PolicyValueNetNumpy
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet  # Keras


class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    n = 5
    # width, height = 8, 8
    # model_file = 'best_policy_8_8_5.model'
    width, height = 6, 6
    model_file = 'best_policy.model'

    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        # ############### human VS AI ###################

        # Original pickle-based code (commented out)
        # try:
        #     policy_param = pickle.load(open(model_file, 'rb'))
        # except:
        #     policy_param = pickle.load(open(model_file, 'rb'),
        #                                encoding='bytes')  # To support python3
        # best_policy = PolicyValueNet(width, height, policy_param)

        # New torch-based loading (your model was saved with state_dict)
        state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        best_policy = PolicyValueNet(width, height)
        best_policy.policy_value_net.load_state_dict(state_dict)

        mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)

        # human player, input your move in the format: 2,3
        human = Human()

        # set start_player=0 for human first
        game.start_play(human, mcts_player, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')



if __name__ == '__main__':
    run()
