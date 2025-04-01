import zmq
import zmq.asyncio
import asyncio
import numpy as np
import pickle
import time
import torch
import torch.nn.functional as F
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet

class AlphaZeroActor:
    def __init__(self, board_width=6, board_height=6, n_in_row=4):
        # ZMQ setup
        self.ctx = zmq.asyncio.Context()
        
        # SUB socket to receive model weights
        self.sub_socket = self.ctx.socket(zmq.SUB)
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.sub_socket.connect("tcp://localhost:5557")
        
        # PUSH socket to send experiences
        self.push_socket = self.ctx.socket(zmq.PUSH)
        self.push_socket.connect("tcp://localhost:5556")
        
        # Model and game setup
        self.board_width = board_width
        self.board_height = board_height
        self.n_in_row = n_in_row
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = Game(self.board)
        
        # Training params
        self.temp = 1.0
        self.n_playout = 400
        self.c_puct = 5
        
        # Shared state
        self.model_weights = None
        self.model_hash = None
        self.policy_value_net = None
        self.mcts_player = None
        
        # Statistics
        self.game_count = 0
        self.started_at = time.time()
    
    def extract_weights_and_hash(self, message):
        """Extract weights and hash from a received message"""
        weights = np.frombuffer(message[:-32], dtype=np.float32)
        hash_val = message[-32:]
        return weights, hash_val
    
    def load_weights_into_model(self, flat_weights):
        """Load flat weights into the PyTorch model"""
        with torch.no_grad():
            flat_tensor = torch.tensor(flat_weights, dtype=torch.float32)
            idx = 0
            for param in self.policy_value_net.policy_value_net.parameters():
                numel = param.numel()
                param.copy_(flat_tensor[idx:idx + numel].view(param.shape))
                idx += numel
    
    def create_model_and_mcts(self):
        """Create the policy-value network and MCTS player"""
        self.policy_value_net = PolicyValueNet(self.board_width, self.board_height)
        self.mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=1
        )
    
    def get_equi_data(self, play_data):
        """Augment the data set by rotation and flipping"""
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                  np.flipud(equi_mcts_prob).flatten(),
                                  winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                  np.flipud(equi_mcts_prob).flatten(),
                                  winner))
        return extend_data
    
    async def receive_weights(self):
        """Receive and update model weights"""
        while True:
            message = await self.sub_socket.recv()
            self.model_weights, self.model_hash = self.extract_weights_and_hash(message)
            
            # Create model on first weights receipt
            if self.policy_value_net is None:
                self.create_model_and_mcts()
                print(f"[Actor] Received initial model weights. Hash: {self.model_hash[:8].decode()}")
            
            # Load weights into model
            # print(self.model_weights.shape)
            self.load_weights_into_model(self.model_weights)
    
    async def generate_self_play_games(self):
        """Generate self-play games and send experiences"""
        # Wait for initial weights
        while self.model_weights is None:
            await asyncio.sleep(0.1)
        
        while True:
            start_time = time.time()
            
            # Run a self-play game
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            
            # Process the game data
            play_data = list(play_data)[:]
            augmented_data = self.get_equi_data(play_data)
            
            # Send experience data to server
            experience_bytes = pickle.dumps(augmented_data)
            experience_with_hash = experience_bytes + self.model_hash
            await self.push_socket.send(experience_with_hash)
            
            # Update statistics
            self.game_count += 1
            duration = time.time() - start_time
            elapsed = time.time() - self.started_at
            game_per_sec = self.game_count / elapsed
            
            # Print stats
            result = "Win" if winner == 1 else "Loss" if winner == 2 else "Draw"
            print(f"[Actor] Game {self.game_count}: {result} | Moves: {len(play_data)} | Time: {duration:.2f}s | Games/sec: {game_per_sec:.2f}")
            
            # Small sleep to avoid overwhelming the system
            await asyncio.sleep(0.01)
    
    async def run(self):
        """Run the actor's tasks"""
        await asyncio.gather(
            self.receive_weights(),
            self.generate_self_play_games(),
        )

if __name__ == "__main__":
    actor = AlphaZeroActor()
    asyncio.run(actor.run())