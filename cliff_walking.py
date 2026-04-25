import numpy as np
import matplotlib.pyplot as plt

# Environment properties
ROWS = 4
COLS = 12
START = (3, 0)
GOAL = (3, 11)

# Actions: 0: Up, 1: Right, 2: Down, 3: Left
ACTIONS = [0, 1, 2, 3]

def step(state, action):
    r, c = state
    if action == 0:
        r = max(0, r - 1)
    elif action == 1:
        c = min(COLS - 1, c + 1)
    elif action == 2:
        r = min(ROWS - 1, r + 1)
    elif action == 3:
        c = max(0, c - 1)
        
    new_state = (r, c)
    
    # Check cliff
    if new_state[0] == 3 and 1 <= new_state[1] <= 10:
        return START, -100, False
    
    # Check goal
    if new_state == GOAL:
        return new_state, -1, True
        
    return new_state, -1, False

def get_action(state, q_value, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(ACTIONS)
    else:
        values = q_value[state[0], state[1], :]
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])

def q_learning(episodes=500, alpha=0.1, epsilon=0.1, gamma=0.9):
    q_value = np.zeros((ROWS, COLS, len(ACTIONS)))
    rewards = np.zeros(episodes)
    
    for ep in range(episodes):
        state = START
        done = False
        total_reward = 0
        
        while not done:
            action = get_action(state, q_value, epsilon)
            next_state, reward, done = step(state, action)
            
            best_next_action = np.argmax(q_value[next_state[0], next_state[1], :])
            td_target = reward + gamma * q_value[next_state[0], next_state[1], best_next_action]
            td_error = td_target - q_value[state[0], state[1], action]
            q_value[state[0], state[1], action] += alpha * td_error
            
            state = next_state
            total_reward += reward
            
        rewards[ep] = total_reward
        
    return q_value, rewards

def sarsa(episodes=500, alpha=0.1, epsilon=0.1, gamma=0.9):
    q_value = np.zeros((ROWS, COLS, len(ACTIONS)))
    rewards = np.zeros(episodes)
    
    for ep in range(episodes):
        state = START
        done = False
        total_reward = 0
        
        action = get_action(state, q_value, epsilon)
        
        while not done:
            next_state, reward, done = step(state, action)
            next_action = get_action(next_state, q_value, epsilon)
            
            td_target = reward + gamma * q_value[next_state[0], next_state[1], next_action]
            td_error = td_target - q_value[state[0], state[1], action]
            q_value[state[0], state[1], action] += alpha * td_error
            
            state = next_state
            action = next_action
            total_reward += reward
            
        rewards[ep] = total_reward
        
    return q_value, rewards

def run_experiments(runs=50, episodes=500, alpha=0.1, epsilon=0.1, gamma=0.9):
    q_rewards_all = np.zeros((runs, episodes))
    sarsa_rewards_all = np.zeros((runs, episodes))
    
    # Q-Tables for finding the greedy path after training
    # For a stable path, we will just keep the final Q-table from the final run
    # Or an average Q-table over all runs
    q_q_final = np.zeros((ROWS, COLS, len(ACTIONS)))
    sarsa_q_final = np.zeros((ROWS, COLS, len(ACTIONS)))
    
    for r in range(runs):
        q_q, q_rewards = q_learning(episodes, alpha, epsilon, gamma)
        sarsa_q, sarsa_rewards = sarsa(episodes, alpha, epsilon, gamma)
        
        q_rewards_all[r] = q_rewards
        sarsa_rewards_all[r] = sarsa_rewards
        
        q_q_final += q_q / runs
        sarsa_q_final += sarsa_q / runs
            
    q_avg_rewards = np.mean(q_rewards_all, axis=0)
    sarsa_avg_rewards = np.mean(sarsa_rewards_all, axis=0)
    
    return q_avg_rewards, sarsa_avg_rewards, q_q_final, sarsa_q_final

def get_optimal_path(q_value):
    state = START
    path = [state]
    done = False
    
    # Generate path using purely greedy policy (epsilon = 0)
    steps = 0
    while state != GOAL and steps < 100:
        action = np.argmax(q_value[state[0], state[1], :])
        state, _, done = step(state, action)
        path.append(state)
        if done and state == START:
           break
        steps += 1
    return path

def plot_path(path, q_value, title, filename):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    
    for x in range(COLS + 1):
        ax.axvline(x, color='black', linewidth=1)
    for y in range(ROWS + 1):
        ax.axhline(y, color='black', linewidth=1)
        
    for c in range(1, 11):
        rect = plt.Rectangle((c, 0), 1, 1, facecolor='lightblue', edgecolor='black')
        ax.add_patch(rect)
        ax.text(c + 0.5, 0.5, 'Cliff', ha='center', va='center', fontsize=12)
        
    ax.text(0.5, 0.5, 'Start', ha='center', va='center', fontsize=12)
    ax.text(11.5, 0.5, 'Goal', ha='center', va='center', fontsize=12)
    
    # Draw policy arrows
    for r in range(ROWS):
        for c in range(COLS):
            state = (r, c)
            if state == START or state == GOAL or (r == 3 and 1 <= c <= 10):
                continue
                
            action = np.argmax(q_value[r, c, :])
            y = 3 - r + 0.5
            x = c + 0.5
            
            dx, dy = 0, 0
            if action == 0: dy = 0.3
            elif action == 1: dx = 0.3
            elif action == 2: dy = -0.3
            elif action == 3: dx = -0.3
            
            ax.annotate("", xy=(x + dx, y + dy), xycoords='data',
                        xytext=(x, y), textcoords='data',
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5, alpha=0.4))
                        
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        
        y1 = 3 - r1 + 0.5
        x1 = c1 + 0.5
        
        y2 = 3 - r2 + 0.5
        x2 = c2 + 0.5
        
        ax.annotate("",
                    xy=(x2, y2), xycoords='data',
                    xytext=(x1, y1), textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="purple", 
                                    lw=2))
        
    ax.set_title(title)
    ax.axis('off')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    runs = 50
    episodes = 500
    alpha = 0.1
    epsilon = 0.1
    gamma = 0.9 # Using 0.9 as specified in assignment
    
    print("Running experiments...")
    q_rewards, sarsa_rewards, q_q, sarsa_q = run_experiments(runs, episodes, alpha, epsilon, gamma)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sarsa_rewards, label='SARSA', color='c')
    plt.plot(q_rewards, label='Q-learning', color='r')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.title('SARSA vs Q-learning on Cliff Walking (Epsilon=0.1, Alpha=0.1, Avg 50 runs)')
    plt.legend()
    # Typical rewards range for this problem
    plt.ylim([-100, -10]) 
    plt.grid(True)
    plt.savefig('learning_curve.png')
    print("Saved learning_curve.png")
    
    q_path = get_optimal_path(q_q)
    sarsa_path = get_optimal_path(sarsa_q)
    
    plot_path(q_path, q_q, "Q-Learning Optimal Path", 'q_learning_path.png')
    plot_path(sarsa_path, sarsa_q, "SARSA Optimal Path", 'sarsa_path.png')
    print("Saved path images.")
