

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from flask_socketio import SocketIO, emit #to connect python backend and javascript frontend
from collections import deque
import random
from memory_brain import DQN, ReplayMemory
import matplotlib.pyplot as plt
from time import sleep
from threading import Lock
from queue import Queue
import threading
import sys

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#, Manager # to run function after the server starts
#I use manager to share the variables between the processes, now they are not shared!


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #gpu for faster batch processing

# Hyperparameters
gamma = 0.9 #how much agent values future rewards over immediate rewards
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32
target_update_interval = 100
memory_capacity = 10000

# Environment specifics
state_dim = 14  # Example input dimension (depends on your environment)
action_dim = 4 # Example output dimension (depends on your environment)

# Initialize networks
policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())  # Start with the same weights
target_net.eval()

# Optimizer
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# Replay Memory
memory = ReplayMemory(memory_capacity)

#############################################################################################################
# Hyperparameters 2
gamma2 = 0.9
epsilon2 = 1.0
epsilon_min2 = 0.01
epsilon_decay2 = 0.995
learning_rate2 = 0.001
batch_size2 = 32
target_update_interval2 = 100
memory_capacity2 = 10000

# Environment specifics 2
state_dim2 = 14  # Example input dimension (depends on your environment)
action_dim2 = 4 # Example output dimension (depends on your environment)

# Initialize networks 2
policy_net2 = DQN(state_dim, action_dim).to(device)
target_net2 = DQN(state_dim, action_dim).to(device)
target_net2.load_state_dict(policy_net2.state_dict())  # Start with the same weights
target_net2.eval()

# Optimizer 2
optimizer2 = optim.Adam(policy_net2.parameters(), lr=learning_rate2)

# Replay Memory 2
memory2 = ReplayMemory(memory_capacity2)


# Ensure tensor-to-scalar conversion is handled correctly
def select_action(state, policy, action_dim, epsilon):
    if random.random() < epsilon:
        return random.randrange(action_dim)  # Random action
    
    with torch.no_grad():
        state = state.unsqueeze(0).to(device)  # Add batch dimension: [1, state_dim]
        q_values = policy(state)  # Forward pass through the network
        
        # Ensure you're working with a scalar value
        return q_values.max(1)[1].item()  # Convert to a Python scalar


def optimize_model(memory, policy_net ,target_net, optimizer, gamma, batch_size):
    

    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = list(zip(*transitions))

  
    state_batch = torch.tensor(torch.stack(batch[0]), dtype=torch.float).to(device)
    
    action_batch = torch.tensor(batch[1], dtype=torch.long).unsqueeze(1).to(device)
  
    reward_batch = torch.tensor(batch[2], dtype=torch.float).to(device)

    next_state_batch = torch.tensor(torch.stack(batch[3]), dtype=torch.float).to(device)

    done_batch = torch.tensor(batch[4], dtype=torch.float).to(device)
   
    current_q_values = policy_net(state_batch).gather(1, action_batch)
 
    with torch.no_grad():
        next_q_values = target_net(next_state_batch).max(1)[0]
        expected_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))
 
    loss = F.mse_loss(current_q_values.squeeze(), expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item() #for plot

def game_logic(action, action2, velocityX, velocityY, velocity2X, velocity2Y):

    # Control snake 1 with model
    if (action == 0 and velocityY != 1): # up
        velocityY = -1
        velocityX = 0

    elif (action == 1 and velocityY != -1):  # down
        velocityY = 1
        velocityX = 0

    elif(action == 2 and velocityX!= 1):  # left
        velocityX = -1
        velocityY = 0

    elif (action == 3 and velocityX!= -1):  # right
        velocityX = 1
        velocityY = 0

    # Control snake 2 with model2
    if (action2 == 0 and velocity2Y != 1): #up
        velocity2Y = -1
        velocity2X = 0

    elif (action2  == 1 and velocity2Y != -1): # down
        velocity2Y = 1
        velocity2X = 0

    elif (action2  == 2 and velocity2X!= 1): #left
        velocity2X = -1
        velocity2Y = 0
    
    elif (action2  == 3 and velocity2X!= -1): #right
        velocity2X = 1
        velocity2Y = 0
    return velocityX, velocityY, velocity2X, velocity2Y


from flask import Flask, send_from_directory

#initialize the flask app
app = Flask(__name__, static_folder='.')

#make websoccent connection to transfer data between python and javascript
socketio = SocketIO(app)

#just load the page with html,css and javascript
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')
@app.route('/<path:path>')
def serve_file(path):
    return send_from_directory('.', path)

def plot_metrics(avg_reward1, avg_reward2):
    plt.figure(figsize=(10, 5))

    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(avg_reward1, label="Snake 1")
    plt.plot(avg_reward2, label="Snake 2")
    plt.xlabel('Games')
    plt.ylabel('Reward')
    plt.title('Rewards per 10 Games') 
    plt.legend()
    plt.savefig('rewards.png')
    #plt.show()

def plot_losses(losses1, losses2):
    plt.figure(figsize=(10, 5))

    plt.plot(losses1, label="Loss DQN1")
    plt.plot(losses2, label="Loss DQN2")
    plt.xlabel('games')
    plt.ylabel('Loss')
    plt.title('AVG Loss per 10 games')
    
    plt.legend()
    plt.savefig('rewards.png')
    #plt.show()





def train_snakes(shared_data, emit_queue):
    global epsilon, epsilon2
    
    total_rewards1 = []
    total_rewards2 = []
    episode_reward1 = 0
    episode_reward2 = 0
    games_played = 0
    steps_done = 0
    num_episodes = 10000
    losses1=[]
    losses2=[]
    

    try:
        for i in range(num_episodes):

            while not shared_data['state_received']:
                sleep(0.1)
            shared_data['state_received'] = False
            special_lock.acquire()
            data = shared_data['state']  # State before action
            special_lock.release()
            # Process the state and get actions
            data_feed = [data['snake1']['x'], data['snake1']['y'], data['snake1']['body'],
                         data['snake1dir']['x'], data['snake1dir']['y'], data['reward1'],
                         data['snake2']['x'], data['snake2']['y'], data['snake2']['body'],
                         data['snake2dir']['x'], data['snake2dir']['y'],
                         data['food']['x'], data['food']['y'], data['reward2']]
            velocityX = data['snake1dir']['x']
            velocityY = data['snake1dir']['y']
            velocity2X = data['snake2dir']['x']
            velocity2Y = data['snake2dir']['y']
           
            data_feed = torch.tensor(data_feed, dtype=torch.float).to(device)
         
            action = select_action(data_feed, policy_net, action_dim, epsilon)
            action2 = select_action(data_feed, policy_net2, action_dim2, epsilon2)
         
            velocityX, velocityY, velocity2X, velocity2Y = game_logic(action, action2, velocityX, velocityY, velocity2X, velocity2Y)
            
            emit_queue.put({
                'velocityX': velocityX, 
                'velocityY': velocityY,
                'velocity2X': velocity2X,
                'velocity2Y': velocity2Y
            })

            sleep(0.07)  # the problem is , my new state is not updated yet, so I need to wait for it (according to a debugger)

            while not shared_data['new_state_received']:
                    sleep(0.1) # Wait for new state to be received
            
            shared_data['new_state_received'] = False
            special_lock.acquire()
            new_data = shared_data['new_state']  # State after action
            special_lock.release()
            new_data_feed = [
                new_data['snake1']['x'], new_data['snake1']['y'], new_data['snake1']['body'],
                new_data['snake1dir']['x'], new_data['snake1dir']['y'], new_data['reward1'],
                new_data['snake2']['x'], new_data['snake2']['y'], new_data['snake2']['body'],
                new_data['snake2dir']['x'], new_data['snake2dir']['y'],
                new_data['food']['x'], new_data['food']['y'], new_data['reward2']
            ]
          
            new_data_feed = torch.tensor(new_data_feed, dtype=torch.float).to(device)
            
            reward1 = new_data['reward1']
            reward2 = new_data['reward2']
            done1 = new_data['gameover']
            done2 = new_data['gameover']


            episode_reward1 += reward1
            episode_reward2 += reward2
            
            memory.push(data_feed, action, reward1, new_data_feed, done1)
            memory2.push(data_feed, action2, reward2, new_data_feed, done2)
           
            loss1=optimize_model( memory, policy_net, target_net, optimizer, gamma, batch_size)
            
            loss2= optimize_model( memory2, policy_net2, target_net2, optimizer2, gamma2, batch_size2)

            losses1.append(loss1)
            losses2.append(loss2) 
            
            # print('Action 1 ',action ,'Reward 1 ',reward1  )
            # print('Action 2 ',action2 ,'Reward 2 ',reward2 )
           
            if done1:
                games_played += 1
                total_rewards1.append(episode_reward1)
                episode_reward1 = 0

            if done2:
                total_rewards2.append(episode_reward2)
                episode_reward2 = 0

            if games_played % 10 == 0 and games_played > 0 and loss1:
                avg_reward1 = np.mean(total_rewards1[-10:])
                avg_reward2 = np.mean(total_rewards2[-10:])
                avg_loss1 = np.mean(losses1[-10:])
                avg_loss2 = np.mean(losses2[-10:])
                print(f"Game {games_played}: Avg Reward1: {avg_reward1}, Avg Reward2: {avg_reward2}")
                print(f"Game {games_played}: Avg loss1: {avg_loss1}, Avg loss2: {avg_loss2}")
                print(f"Epsilon1: {epsilon}, Epsilon2: {epsilon2}")
                plot_metrics(total_rewards1[-10:], total_rewards2[-10:])
                plot_losses(losses1[-10:],losses2[-10:])
                games_played=0

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            if epsilon2 > epsilon_min2:
                epsilon2 *= epsilon_decay

            if steps_done % target_update_interval == 0:
                target_net.load_state_dict(policy_net.state_dict())
             

            if steps_done % target_update_interval2 == 0:
                target_net2.load_state_dict(policy_net2.state_dict())

            steps_done += 1
        save_models()
    except Exception as e:
        print(f"Error handling game state: {e}")

from time import time
def handle_emit(emit_queue):
    emit_interval = 1.0 / 10
    while True:
        start_time = time()
        message = emit_queue.get()
        if message is None:
            break
        
            # print(f"Time since last emit: {elapsed_time:.3f} seconds")
            # sys.stdout.flush()
        
        socketio.emit('action', message)
        #print('Emitting action')
        #sys.stdout.flush()
        elapsed_time = time() - start_time
        sleep_time = emit_interval - elapsed_time
        if sleep_time > 0:
                sleep(sleep_time)

# Flask-SocketIO configuration
@socketio.on('connect')
def handle_connect():
    logger.info("Client connected, starting the training process.")
    socketio.start_background_task(train_snakes, shared_data, emit_queue)

@socketio.on('game_state_after_action')
def handle_game_newstate(data):
   
        try:
            special_lock.acquire()
            shared_data['new_state'] = data
            shared_data['new_state_received'] = True
            special_lock.release()
        except Exception as e:
            print(f"Error updating new state: {e}")

@socketio.on('game_state_before_action')
def accept(data):
        
        try:
            special_lock.acquire()
            shared_data['state'] = data
            shared_data['state_received'] = True
            special_lock.release()
        except Exception as e:
            print(f"Error updating state: {e}")

special_lock=Lock()
def save_models():
    torch.save(policy_net.state_dict(), 'policy_net.pth')
    torch.save(target_net.state_dict(), 'target_net.pth')
    torch.save(policy_net2.state_dict(), 'policy_net2.pth')
    torch.save(target_net2.state_dict(), 'target_net2.pth')

if __name__ == '__main__':
    shared_data = {
        'state': None,
        'new_state': None,
        'state_received': False,
        'new_state_received': False,
    }

    emit_queue = Queue()

    # Start the server with debug mode enabled
    emit_thread = threading.Thread(target=handle_emit, args=(emit_queue,))
    emit_thread.start()

    socketio.run(app, host='0.0.0.0', debug=True)

    emit_queue.put(None)
    emit_thread.join()
