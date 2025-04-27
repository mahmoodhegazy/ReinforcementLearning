import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse
import json
import ale_py

gym.register_envs(ale_py)

class Q_Network(nn.Module):
    """Neural network to approximate Q-function."""
    
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=2):
        """
        Initialize Q-Network with configurable architecture.
        
        Args:
            input_dim (int): Dimension of the input state
            output_dim (int): Number of actions (output dimension)
            hidden_dim (int): Size of hidden layers
            num_layers (int): Number of hidden layers
        """
        super(Q_Network, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        self.init_weights()
        
    def init_weights(self):
        """Initialize all linear layers uniformly between -0.001 and 0.001."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-0.001, b=0.001)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, a=-0.001, b=0.001)
                    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)

class ReplayBuffer:
    """Optimized experience replay buffer to store and sample transitions."""
    
    def __init__(self, capacity):
        """
        Initialize replay buffer with pre-allocated arrays.
        
        Args:
            capacity (int): Maximum number of transitions to store
        """
        self.capacity = capacity
        self.position = 0
        self.size = 0
        # Pre-allocate memory for better performance, will initialize state arrays on first add
        self.buffer = {
            'states': None,  # Will initialize on first add when we know the state shape
            'actions': np.zeros(capacity, dtype=np.int64),
            'rewards': np.zeros(capacity, dtype=np.float32),
            'next_states': None,  # Will initialize on first add
            'dones': np.zeros(capacity, dtype=np.float32)
        }
        
    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Initialize state arrays if this is the first add
        if self.buffer['states'] is None:
            state_shape = np.array(state).shape
            self.buffer['states'] = np.zeros((self.capacity, *state_shape), dtype=np.float32)
            self.buffer['next_states'] = np.zeros((self.capacity, *state_shape), dtype=np.float32)
            
        # Store transition
        self.buffer['states'][self.position] = state
        self.buffer['actions'][self.position] = action
        self.buffer['rewards'][self.position] = reward
        self.buffer['next_states'][self.position] = next_state
        self.buffer['dones'][self.position] = float(done)
        
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones)
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.buffer['states'][indices],
            self.buffer['actions'][indices],
            self.buffer['rewards'][indices],
            self.buffer['next_states'][indices],
            self.buffer['dones'][indices]
        )
    
    def __len__(self):
        """Return the current size of the buffer."""
        return self.size

def set_seeds(seed):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def preprocess_state(state):
    """
    Preprocess state for neural network input.
    
    Args:
        state: Raw state from environment
        
    Returns:
        np.ndarray: Preprocessed state as float32 array
    """
    if isinstance(state, np.ndarray) and state.dtype == np.uint8:
        return state.astype(np.float32) / 255.0
    return np.array(state, dtype=np.float32)

def select_action(state, q_network, epsilon, env, device):
    """
    Select action using epsilon-greedy policy.
    
    Args:
        state: Current state
        q_network: Q-function neural network
        epsilon: Exploration rate
        env: Environment
        device: Device to run computations on
        
    Returns:
        int: Selected action
    """
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = q_network(state_tensor)
        return q_values.argmax(1).item()

def update_q_network_no_replay(q_network, optimizer, state, action, reward, next_state, done, 
                    gamma, epsilon, algorithm, device):
    """
    Update Q-network for a single transition (no replay).
    
    Args:
        q_network: Q-function neural network
        optimizer: Optimizer for the Q-network
        state: Current state
        action: Action taken
        reward: Reward received
        next_state: Next state
        done: Whether episode is done
        gamma: Discount factor
        epsilon: Exploration rate
        algorithm: 'q_learning' or 'expected_sarsa'
        device: Device to run computations on
    """
    q_network.train()
    current_state = torch.FloatTensor(state).view(1, -1).to(device)
    action_values = q_network(current_state)
    current_q_estimate = action_values[0, action].view(1)
    
    subsequent_state = torch.FloatTensor(next_state).view(1, -1).to(device)
    with torch.no_grad():
        subsequent_q_values = q_network(subsequent_state)
    
    if done:
        q_target = reward
    else:
        if algorithm == 'q_learning':
            # Q-learning uses the maximum Q-value for the next state
            q_target = reward + gamma * subsequent_q_values.max().item()
        elif algorithm == 'expected_sarsa':
            # Expected SARSA uses the expected Q-value under the policy
            action_count = subsequent_q_values.shape[1]
            optimal_action = subsequent_q_values.argmax().item()
            
            # Calculate the weighted sum of Q-values based on epsilon-greedy policy
            policy_expectation = 0.0
            non_greedy_prob = epsilon / action_count
            greedy_prob = 1 - epsilon + non_greedy_prob
            
            # Calculate expected value based on epsilon-greedy policy
            for potential_action in range(action_count):
                # Probability of selecting action a under ε-greedy policy
                action_probability = greedy_prob if potential_action == optimal_action else non_greedy_prob
                policy_expectation += action_probability * subsequent_q_values[0, potential_action].item()
                
            q_target = reward + gamma * policy_expectation
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    target_tensor = torch.FloatTensor([q_target]).to(device)
    error = nn.MSELoss()(current_q_estimate, target_tensor)
    
    optimizer.zero_grad()
    error.backward()
    optimizer.step()

def update_q_network_replay(q_network, optimizer, states, actions, rewards, next_states, dones, 
                          gamma, epsilon, algorithm, device):
    """
    Optimized batch update for Q-network.
    
    Args:
        q_network: Q-function neural network
        optimizer: Optimizer for the Q-network
        states: Batch of states
        actions: Batch of actions
        rewards: Batch of rewards
        next_states: Batch of next states
        dones: Batch of episode termination flags
        gamma: Discount factor
        epsilon: Exploration rate
        algorithm: 'q_learning' or 'expected_sarsa'
        device: Device to run computations on
    """
    q_network.train()
    
    # Convert numpy arrays to tensors and move to device in one operation
    states_tensor = torch.FloatTensor(states).to(device)
    actions_tensor = torch.LongTensor(actions).to(device)
    rewards_tensor = torch.FloatTensor(rewards).to(device)
    next_states_tensor = torch.FloatTensor(next_states).to(device)
    dones_tensor = torch.FloatTensor(dones).to(device)
    
    # Get Q-values for the batch
    q_values = q_network(states_tensor)
    q_value = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
    
    with torch.no_grad():
        next_q_values = q_network(next_states_tensor)
    
    if algorithm == 'q_learning':
        # Q-learning uses max Q-value for next states (vectorized)
        next_q_max = next_q_values.max(1)[0]
        target = rewards_tensor + gamma * (1 - dones_tensor) * next_q_max
    elif algorithm == 'expected_sarsa':
        # Expected SARSA uses expected Q-value under the policy (vectorized)
        num_actions = next_q_values.shape[1]
        best_actions = next_q_values.argmax(dim=1)
        
        # Initialize probabilities for ε-greedy policy
        probs = torch.ones_like(next_q_values) * (epsilon / num_actions)
        
        # Use advanced indexing to update probabilities for best actions
        batch_indices = torch.arange(best_actions.size(0), device=device)
        probs[batch_indices, best_actions] = 1 - epsilon + epsilon / num_actions
        
        # Calculate expected Q-value (vectorized)
        expected_q = (next_q_values * probs).sum(dim=1)
        target = rewards_tensor + gamma * (1 - dones_tensor) * expected_q
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    loss = nn.MSELoss()(q_value, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_episode(env, q_network, optimizer, epsilon, gamma, algorithm, replay_buffer=None,
                 batch_size=64, device='cpu', update_frequency=4):
    """
    Train for one episode with optimized updates.
    
    Args:
        env: Gym environment
        q_network: Q-function neural network
        optimizer: Optimizer for Q-network
        epsilon: Exploration rate
        gamma: Discount factor
        algorithm: 'q_learning' or 'expected_sarsa'
        replay_buffer: Replay buffer (None means no replay)
        batch_size: Mini-batch size for replay updates
        device: Device to run computations on
        update_frequency: How often to update the network (in steps)
        
    Returns:
        float: Total reward for the episode
    """
    # Handle different gym versions for reset
    reset_result = env.reset()
    if isinstance(reset_result, tuple):  # New gym version returns (obs, info)
        state = reset_result[0]
    else:  # Old gym version returns just obs
        state = reset_result
        
    state = preprocess_state(state)
    total_reward = 0
    done = False
    step_count = 0
    
    while not done:
        # Select action using epsilon-greedy policy
        action = select_action(state, q_network, epsilon, env, device)
        
        # Take action in environment (handle different gym versions)
        step_result = env.step(action)
        
        if len(step_result) == 4:  # Old gym: obs, reward, done, info
            next_state, reward, done, _ = step_result
        else:  # New gym: obs, reward, terminated, truncated, info
            next_state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
            
        next_state = preprocess_state(next_state)
        total_reward += reward
        step_count += 1
        
        if replay_buffer is None:
            # Direct update (no replay)
            update_q_network_no_replay(q_network, optimizer, state, action, reward, next_state, done,
                            gamma, epsilon, algorithm, device)
        else:
            # Add to replay buffer
            replay_buffer.add(state, action, reward, next_state, done)
            
            # Update from replay less frequently for better efficiency
            if step_count % update_frequency == 0 and len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                update_q_network_replay(q_network, optimizer, states, actions, rewards, next_states,
                                     dones, gamma, epsilon, algorithm, device)
        
        state = next_state
    
    return total_reward

def run_experiment(env_name, algorithm, use_replay, epsilon, step_size, num_episodes=1000,
                  seed=0, num_layers=2, hidden_dim=256, gamma=0.99, batch_size=256, 
                  update_frequency=4, device='cpu'):
    """
    Run a complete experiment with one configuration and optimized parameters.
    
    Args:
        env_name: Name of the gym environment
        algorithm: 'q_learning' or 'expected_sarsa'
        use_replay: Whether to use experience replay
        epsilon: Exploration rate
        step_size: Learning rate
        num_episodes: Number of episodes to train
        seed: Random seed
        num_layers: Number of hidden layers in Q-network
        hidden_dim: Hidden dimension size
        gamma: Discount factor
        batch_size: Mini-batch size for replay updates
        update_frequency: How often to update the network
        device: Device to run computations on
        
    Returns:
        list: Episode rewards
    """
    # Create environment and set seeds
    env = gym.make(env_name)
    set_seeds(seed)
    
    # Get dimensions for Q-network
    if hasattr(env.observation_space, 'shape'):
        input_dim = env.observation_space.shape[0]
    else:
        input_dim = env.observation_space.n
    
    output_dim = env.action_space.n
    
    # Initialize Q-network
    q_network = Q_Network(input_dim, output_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    
    # Initialize optimizer with adaptive learning rate
    optimizer = optim.Adam(q_network.parameters(), lr=step_size)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=50, verbose=False
    )
    
    # Initialize replay buffer with optimized capacity
    replay_buffer = ReplayBuffer(int(1e6)) if use_replay else None
    
    # Track episode rewards
    episode_rewards = []
    
    # Run training loop
    for episode in range(num_episodes):
        reward = train_episode(
            env=env,
            q_network=q_network,
            optimizer=optimizer,
            epsilon=epsilon,
            gamma=gamma,
            algorithm=algorithm,
            replay_buffer=replay_buffer,
            batch_size=batch_size,
            update_frequency=update_frequency,
            device=device
        )
        episode_rewards.append(reward)
        
        # Update learning rate based on performance every 10 episodes
        if episode % 10 == 0 and episode > 0:
            avg_reward = np.mean(episode_rewards[-10:])
            scheduler.step(avg_reward)
    
    env.close()
    return episode_rewards

def plot_results(results, env_name, use_replay, epsilon, step_size, algorithms):
    """
    Plot training curves for a specific configuration.
    
    Args:
        results (dict): Dictionary containing results
        env_name (str): Environment name
        use_replay (bool): Whether replay buffer was used
        epsilon (float): Exploration rate
        step_size (float): Learning rate
        algorithms (list): List of algorithms to plot
    """
    plt.figure(figsize=(10, 6))
    episodes = np.arange(1, results[env_name][use_replay][epsilon][step_size][algorithms[0]].shape[1] + 1)
    
    # Define colors and line styles
    colors = ['green', 'red']
    line_styles = ['-', '--']
    
    # Plot data for each algorithm
    for i, alg in enumerate(algorithms):
        data = results[env_name][use_replay][epsilon][step_size][alg]
        mean_reward = data.mean(axis=0)
        std_reward = data.std(axis=0)
        
        plt.plot(episodes, mean_reward, color=colors[i], linestyle=line_styles[i % len(line_styles)], 
                label=f"{alg}")
        plt.fill_between(episodes, mean_reward - std_reward, mean_reward + std_reward,
                        color=colors[i], alpha=0.2)
    
    # Configure plot appearance
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    replay_text = "Replay" if use_replay else "No Replay"
    plt.title(f"{env_name} | {replay_text} | ε={epsilon} | step size={step_size}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"plots/{env_name}_{'replay' if use_replay else 'noreplay'}_eps{epsilon}_step{step_size}.png"
    plt.savefig(plot_filename, dpi=300)
    plt.close()

def run_all_experiments(env_names, algorithms, epsilons, learning_rates, use_replay_options,
                      num_seeds=50, num_episodes=1000, num_layers=2, hidden_dim=256,
                      gamma=0.99, batch_size=256, update_frequency=4, device='cpu'):
    """
    Run experiments for all configurations with optimized parameters.
    
    Args:
        env_names: List of environment names
        algorithms: List of algorithms to run
        epsilons: List of exploration rates
        learning_rates: List of learning rates
        use_replay_options: List of booleans for replay usage
        num_seeds: Number of random seeds
        num_episodes: Number of training episodes per run
        num_layers: Number of hidden layers in Q-network
        hidden_dim: Hidden dimension size
        gamma: Discount factor
        batch_size: Mini-batch size for replay updates
        update_frequency: How often to update the network
        device: Device to run computations on
        
    Returns:
        dict: Results for all configurations
    """
    results = {}
    
    for env_name in env_names:
        print(f"\n=== Running experiments for {env_name} ===")
        results[env_name] = {}
        
        for use_replay in use_replay_options:
            replay_text = "With replay buffer" if use_replay else "Without replay buffer"
            print(f"\n--- {replay_text} ---")
            results[env_name][use_replay] = {}
            
            for epsilon in epsilons:
                results[env_name][use_replay][epsilon] = {}
                
                for step_size in learning_rates:
                    results[env_name][use_replay][epsilon][step_size] = {}
                    
                    for algorithm in algorithms:
                        print(f"Running {algorithm} with ε={epsilon}, step size={step_size}")
                        all_rewards = []
                        
                        # Create a progress bar for the seeds
                        for seed in tqdm(range(num_seeds), desc=f"{algorithm} seeds"):
                            rewards = run_experiment(
                                env_name=env_name,
                                algorithm=algorithm,
                                use_replay=use_replay,
                                epsilon=epsilon,
                                step_size=step_size,
                                num_episodes=num_episodes,
                                seed=seed,
                                num_layers=num_layers,
                                hidden_dim=hidden_dim,
                                gamma=gamma,
                                batch_size=batch_size,
                                update_frequency=update_frequency,
                                device=device
                            )
                            all_rewards.append(rewards)
                        
                        # Store results as numpy array
                        results[env_name][use_replay][epsilon][step_size][algorithm] = np.array(all_rewards)
                        
                        # Plot after each algorithm to see progress
                        temp_algorithms = [algorithm]
                        if set(algorithms) <= set(results[env_name][use_replay][epsilon][step_size].keys()):
                            temp_algorithms = algorithms
                        plot_results(results, env_name, use_replay, epsilon, step_size, temp_algorithms)
    
    return results

def plot_results_from_json(json_file, epsilons=[0.1, 0.2, 0.3], learning_rates=[0.01, 0.001, 0.0001]):
    """
    Generate comprehensive visualization of Q-learning and Expected SARSA results.
    
    
    This function creates a grid of plots for each combination of epsilon and learning rate,
    showing the performance of both algorithms with standard deviation bands.
    
    Args:
        json_file (str): Path to the JSON results file
        epsilons (list): Exploration rates to plot (default: [0.1, 0.2, 0.3])
        learning_rates (list): Learning rates to plot (default: [0.01, 0.001, 0.0001])
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import json
    import os
    
    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Process each environment
    for env_name in data:
        # Process with and without replay buffer separately
        for replay_setting in ['with_replay', 'without_replay']:
            if replay_setting not in data[env_name]:
                continue
                
            # Create figure with subplots in a grid
            fig, axes = plt.subplots(len(epsilons), len(learning_rates), figsize=(15, 10), 
                                    sharex=True, sharey=True)
            
            # Handle single row/column cases
            if len(epsilons) == 1 and len(learning_rates) == 1:
                axes = np.array([[axes]])
            elif len(epsilons) == 1:
                axes = axes.reshape(1, -1)
            elif len(learning_rates) == 1:
                axes = axes.reshape(-1, 1)
            
            # Plot each combination of epsilon and learning rate
            for i, epsilon in enumerate(epsilons):
                for j, lr in enumerate(learning_rates):
                    ax = axes[i, j]
                    
                    try:
                        # Get data for this configuration
                        eps_key = f"epsilon_{epsilon}"
                        lr_key = f"step_size_{lr}"
                        config_data = data[env_name][replay_setting][eps_key][lr_key]
                        
                        # Extract data for Q-learning
                        q_mean = config_data['q_learning']['mean']
                        q_std = config_data['q_learning']['std']
                        
                        # Extract data for Expected SARSA
                        sarsa_mean = config_data['expected_sarsa']['mean']
                        sarsa_std = config_data['expected_sarsa']['std']
                        
                        # Create x-axis for episodes
                        episodes = np.arange(1, len(q_mean) + 1)
                        
                        # Plot Q-learning (green)
                        ax.plot(episodes, q_mean, color='green', label='Q-Learning')
                        ax.fill_between(episodes, np.array(q_mean) - np.array(q_std), 
                                      np.array(q_mean) + np.array(q_std), color='green', alpha=0.2)
                        
                        # Plot Expected SARSA (red)
                        ax.plot(episodes, sarsa_mean, color='red', label='Expected SARSA')
                        ax.fill_between(episodes, np.array(sarsa_mean) - np.array(sarsa_std), 
                                      np.array(sarsa_mean) + np.array(sarsa_std), color='red', alpha=0.2)
                    except KeyError as e:
                        print(f"Warning: Missing data for {env_name}, {replay_setting}, ε={epsilon}, α={lr}: {e}")
                        ax.text(0.5, 0.5, 'Data not available', 
                              horizontalalignment='center', verticalalignment='center', 
                              transform=ax.transAxes)
                    
                    # Set title and labels for this subplot
                    ax.set_title(f"ε={epsilon}, α={lr}")
                    if i == len(epsilons) - 1:  # Bottom row
                        ax.set_xlabel("Episode")
                    if j == 0:  # Leftmost column
                        ax.set_ylabel("Return")
                    
                    # Add legend to the first subplot only
                    if i == 0 and j == 0:
                        ax.legend()
                    
                    # Add grid for better readability
                    ax.grid(True, linestyle='--', alpha=0.6)
            
            # Add overall title for the entire figure
            replay_text = "with Replay Buffer" if replay_setting == 'with_replay' else "without Replay Buffer"
            fig.suptitle(f"{env_name} {replay_text}", fontsize=16)
            
            # Optimize layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.92)
            
            # Save the figure to a file
            os.makedirs("figures", exist_ok=True)
            filename = f"{env_name.replace('/', '_')}_{replay_setting}.png"
            plt.savefig(os.path.join("figures", filename), dpi=300)
            plt.close()
            
            print(f"Generated visualization for {env_name} {replay_text}")

def main(args):
    """Main function to run the experiments with optimized settings."""
    # Create directory for plots
    os.makedirs("plots", exist_ok=True)
    
    # Set device with improved logic
    if torch.cuda.is_available() and args.device == "cuda":
        device = torch.device("cuda")
        # Set for better performance
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and args.device == "mps":
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Define experiment configurations
    env_names = ["ALE/Assault-ram-v5"] if args.environment == "assault" else ["Acrobot-v1"]
    algorithms = ['q_learning', 'expected_sarsa']
    epsilons = [0.1, 0.2, 0.3]  # Equivalent to the required epsilon values
    learning_rates = [0.01, 0.001, 0.0001]  # Approximates 1/4, 1/8, 1/16 for the Adam optimizer
    use_replay_options = [True] if args.replay_only else [False, True]
    
    # Optimized parameters
    batch_size = args.batch_size
    update_frequency = args.update_frequency
    
    # Run experiments
    results = run_all_experiments(
        env_names=env_names,
        algorithms=algorithms,
        epsilons=epsilons,
        learning_rates=learning_rates,
        use_replay_options=use_replay_options,
        num_seeds=args.num_seeds,
        num_episodes=args.num_episodes,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        batch_size=batch_size,
        update_frequency=update_frequency,
        device=device
    )
    
    # Final plotting
    for env_name in env_names:
        for use_replay in use_replay_options:
            for epsilon in epsilons:
                for step_size in learning_rates:
                    plot_results(results, env_name, use_replay, epsilon, step_size, algorithms)
    
    print("\nExperiments completed. All plots saved in the 'plots' directory.")
    
    # Optionally, save numerical results as JSON
    if args.save_results:
        os.makedirs("results", exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for env_name in env_names:
            json_results[env_name] = {}
            for use_replay in use_replay_options:
                replay_key = 'with_replay' if use_replay else 'without_replay'
                json_results[env_name][replay_key] = {}
                for epsilon in epsilons:
                    json_results[env_name][replay_key][f'epsilon_{epsilon}'] = {}
                    for step_size in learning_rates:
                        json_results[env_name][replay_key][f'epsilon_{epsilon}'][f'step_size_{step_size}'] = {}
                        for algorithm in algorithms:
                            data = results[env_name][use_replay][epsilon][step_size][algorithm]
                            mean = data.mean(axis=0).tolist()
                            std = data.std(axis=0).tolist()
                            json_results[env_name][replay_key][f'epsilon_{epsilon}'][f'step_size_{step_size}'][algorithm] = {
                                'mean': mean,
                                'std': std
                            }
        
        with open('results/numerical_results.json', 'w') as f:
            json.dump(json_results, f)
        print("Numerical results saved to 'results/numerical_results.json'")
        plot_results_from_json('results/numerical_results.json')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run optimized RL experiments with Q-learning and Expected SARSA')
    parser.add_argument('--num_seeds', type=int, default=50, help='Number of random seeds to run')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes per run')
    parser.add_argument('--num_layers', type=int, default=2, choices=[2, 3], help='Number of hidden layers (2 or 3)')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--batch_size', type=int, default=256, help='Mini-batch size for replay updates')
    parser.add_argument('--update_frequency', type=int, default=4, help='How often to update the network (in steps)')
    parser.add_argument('--device', type=str, default='mps', choices=['cuda', 'mps', 'cpu'], help='Device to run on')
    parser.add_argument('--save_results', action='store_true', help='Save numerical results as JSON')
    parser.add_argument('--environment', type=str, default='assault', choices=['assault', 'acrobot'], help='Environment to run')
    parser.add_argument('--replay_only', action='store_true', help='Only run experiments with replay buffer')
    
    args = parser.parse_args()
    main(args)