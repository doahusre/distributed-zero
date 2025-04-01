import zmq
import zmq.asyncio
import asyncio
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import time
from policy_value_net_pytorch import PolicyValueNet

class AlphaZeroLearner:
    def __init__(self, board_width=6, board_height=6):
        # ZMQ setup
        self.ctx = zmq.asyncio.Context()
        
        # Socket to send weights to server
        self.dealer_socket = self.ctx.socket(zmq.DEALER)
        self.dealer_socket.connect("tcp://localhost:5555")
        
        # Socket to receive batches from server
        self.pull_socket = self.ctx.socket(zmq.PULL)
        self.pull_socket.connect("tcp://localhost:5558")
        
        # Model setup
        self.board_width = board_width
        self.board_height = board_height
        self.policy_value_net = PolicyValueNet(board_width, board_height)
        
        # Training parameters
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.batch_size = 512
        self.epochs = 5
        self.kl_targ = 0.02
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_value_net.policy_value_net.parameters(),
            lr=self.learn_rate * self.lr_multiplier,
            weight_decay=1e-4
        )
        
        # Monitoring
        self.batch_count = 0
        self.total_loss = 0
        self.best_loss = float('inf')
        self.start_time = time.time()
    
    def get_flat_weights(self):
        """Get flattened model weights as bytes"""
        with torch.no_grad():
            all_params = [p.data.cpu().numpy().flatten() for p in 
                          self.policy_value_net.policy_value_net.parameters()]
            flat = np.concatenate(all_params).astype(np.float32)
            return flat.tobytes()
    
    async def send_initial_weights(self):
        """Send initial weights to the server"""
        weights = self.get_flat_weights()
        await self.dealer_socket.send(weights)
        print("[Learner] Sent initial weights")
    
    async def send_updated_weights(self):
        """Send updated weights to the server"""
        weights = self.get_flat_weights()
        await self.dealer_socket.send(weights)
        self.batch_count += 1
        
        # Print statistics
        elapsed = time.time() - self.start_time
        batches_per_sec = self.batch_count / elapsed
        avg_loss = self.total_loss / max(1, self.batch_count)
        print(f"[Learner] Sent updated weights | Batches: {self.batch_count} ({batches_per_sec:.2f}/s) | Avg Loss: {avg_loss:.5f}")
    
    async def train(self):
        """Main training loop"""
        await self.send_initial_weights()
        
        while True:
            # Receive a batch from the server
            batch_bytes = await self.pull_socket.recv()
            data_blob, model_hash = batch_bytes[:-32], batch_bytes[-32:]
            
            try:
                # Load and process the batch
                mini_batch = pickle.loads(data_blob)
                loss, entropy = await self.policy_update(mini_batch)
                
                # Track metrics
                self.total_loss += loss
                
                # Send updated weights back to server
                await self.send_updated_weights()
                
            except Exception as e:
                print(f"[Learner] Training error: {e}")
    
    async def policy_update(self, mini_batch):
        """Update the policy-value network"""
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        
        # Convert to PyTorch tensors
        state_batch = torch.FloatTensor(np.array(state_batch))
        mcts_probs_batch = torch.FloatTensor(np.array(mcts_probs_batch))
        winner_batch = torch.FloatTensor(np.array(winner_batch))
        
        # Get old predictions for KL divergence calculation
        old_probs, old_v = self.policy_value_net.policy_value_net(state_batch)
        
        # Optimization loop
        for i in range(self.epochs):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            log_act_probs, value = self.policy_value_net.policy_value_net(state_batch)
            
            # Loss calculation
            value_loss = F.mse_loss(value.view(-1), winner_batch)
            policy_loss = -torch.mean(torch.sum(mcts_probs_batch * log_act_probs, 1))
            loss = value_loss + policy_loss
            
            # Backward and optimize
            loss.backward()
            self.optimizer.step()
            
            # Calculate KL divergence
            new_probs, new_v = self.policy_value_net.policy_value_net(state_batch)
            kl = torch.mean(torch.sum(old_probs * (torch.log(old_probs + 1e-10) - torch.log(new_probs + 1e-10)), 1))
            
            if kl > self.kl_targ * 4:
                # Early stopping if KL diverges too much
                break
        
        # Adjust learning rate based on KL divergence
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        
        # Update learning rate
        param_groups = self.optimizer.param_groups
        for param_group in param_groups:
            param_group['lr'] = self.learn_rate * self.lr_multiplier
        
        # Calculate explained variance for monitoring
        explained_var_old = 1 - torch.var(winner_batch - old_v.view(-1)) / torch.var(winner_batch)
        explained_var_new = 1 - torch.var(winner_batch - new_v.view(-1)) / torch.var(winner_batch)
        
        # Calculate entropy for monitoring
        entropy = -torch.mean(torch.sum(torch.exp(log_act_probs) * log_act_probs, 1))
        
        print(f"[Learner] KL: {kl:.5f}, LR: {self.lr_multiplier:.3f}, Loss: {loss.item():.5f}, " 
              f"Policy: {policy_loss.item():.5f}, Value: {value_loss.item():.5f}, Entropy: {entropy.item():.5f}")
        
        # Save best model
        if loss.item() < self.best_loss:
            self.best_loss = loss.item()
            self.policy_value_net.save_model('./best_policy.model')
            print(f"[Learner] 💾 Saved new best model (loss: {self.best_loss:.5f})")
        
        return loss.item(), entropy.item()
    
    async def run(self):
        """Run the learner"""
        await self.train()

if __name__ == "__main__":
    # Fix for missing F import
    import torch.nn.functional as F
    
    learner = AlphaZeroLearner()
    asyncio.run(learner.run())