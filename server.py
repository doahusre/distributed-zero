import zmq
import zmq.asyncio
import asyncio
import numpy as np
import pickle
import hashlib
import time
from collections import deque

# Configuration
BUFFER_SIZE = 10000
BATCH_SIZE = 512
PUBLISH_FREQUENCY = 0.05  # seconds

class AlphaZeroServer:
    def __init__(self):
        self.ctx = zmq.asyncio.Context()
        
        # Socket for receiving experiences from actors
        self.actor_socket = self.ctx.socket(zmq.PULL)
        self.actor_socket.bind("tcp://0.0.0.0:5556")
        
        # Socket for pushing batches to learner
        self.learner_push_socket = self.ctx.socket(zmq.PUSH)
        self.learner_push_socket.bind("tcp://0.0.0.0:5558")
        
        # Socket for receiving weights from learner
        self.learner_socket = self.ctx.socket(zmq.ROUTER)
        self.learner_socket.bind("tcp://0.0.0.0:5555")
        
        # Socket for publishing weights to actors
        self.pub_socket = self.ctx.socket(zmq.PUB)
        self.pub_socket.bind("tcp://0.0.0.0:5557")
        
        # Storage
        self.data_buffer = deque(maxlen=BUFFER_SIZE)
        self.model_weights = None
        self.model_hash = b"0" * 32
        
        # Stats
        self.episode_count = 0
        self.started_at = time.time()
    
    def generate_hash(self, data):
        """Generate a hash for the model weights"""
        return hashlib.md5(data.tobytes()).hexdigest().encode("utf-8")
    
    async def handle_learner_weights(self):
        """Receive updated weights from the learner and publish to actors"""
        while True:
            frames = await self.learner_socket.recv_multipart()
            identity, weights_bytes = frames[0], frames[1]
            
            # Update model weights
            self.model_weights = np.frombuffer(weights_bytes, dtype=np.float32)
            self.model_hash = self.generate_hash(self.model_weights)
            
            print(f"[Server] Received weights from learner. Hash: {self.model_hash[:8].decode()}")
            
            # Publish the weights to all actors
            await self.pub_socket.send(self.model_weights.tobytes() + self.model_hash)
    
    async def handle_experience(self):
        """Receive gameplay experiences from actors"""
        while True:
            message = await self.actor_socket.recv()
            experience_blob, received_hash = message[:-32], message[-32:]
            
            # Skip if we're still waiting for initial weights
            if self.model_hash == b"0" * 32:
                print("[Server] Discarding experience (waiting for weights)")
                continue
            
            # Skip if the experience was generated with outdated weights
            if received_hash != self.model_hash:
                print("[Server] Ignored outdated experience")
                continue
            
            try:
                experiences = pickle.loads(experience_blob)
                self.data_buffer.extend(experiences)
                self.episode_count += 1
                
                # Print statistics
                elapsed = time.time() - self.started_at
                eps_per_sec = self.episode_count / elapsed
                print(f"[Server] Buffer: {len(self.data_buffer)}/{BUFFER_SIZE} | Episodes: {self.episode_count} ({eps_per_sec:.2f}/s)")
                
                # Send a batch to the learner if we have enough data
                if len(self.data_buffer) >= BATCH_SIZE:
                    await self.send_batch()
            
            except Exception as e:
                print(f"[Server] Failed to process experience: {e}")
    
    async def send_batch(self):
        """Send a batch of experiences to the learner"""
        if len(self.data_buffer) < BATCH_SIZE:
            return
        
        try:
            # Sample a mini-batch
            mini_batch = list(self.data_buffer)
            if len(mini_batch) > BATCH_SIZE:
                indices = np.random.choice(len(mini_batch), BATCH_SIZE, replace=False)
                mini_batch = [mini_batch[i] for i in indices]
            
            # Send to learner
            batch_bytes = pickle.dumps(mini_batch)
            await self.learner_push_socket.send(batch_bytes + self.model_hash)
            print(f"[Server] Sent mini-batch to learner. Batch size: {len(mini_batch)}")
        
        except Exception as e:
            print(f"[Server] Failed to send batch: {e}")
    
    async def publish_weights_periodically(self):
        """Periodically publish weights to ensure actors stay updated"""
        while self.model_weights is None:
            await asyncio.sleep(0.1)
        
        while True:
            await asyncio.sleep(PUBLISH_FREQUENCY)
            await self.pub_socket.send(self.model_weights.tobytes() + self.model_hash)
    
    async def run(self):
        """Run all server tasks"""
        await asyncio.gather(
            self.handle_learner_weights(),
            self.handle_experience(),
            self.publish_weights_periodically(),
        )

if __name__ == "__main__":
    server = AlphaZeroServer()
    asyncio.run(server.run())