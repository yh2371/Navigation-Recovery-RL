from arg_utils import get_args
from recovery_rl.experiment import Experiment
import dill as pickle
import os
import torch
import numpy as np

def torchify(x): 
    if torch.is_tensor(x):
        return torch.FloatTensor(x.clone()).to("cuda")
    return torch.FloatTensor(x).to("cuda")

def get_nav_rollout(exp):
        def recovery_thresh(state, action):
            critic_val = exp.agent.safety_critic.get_value(
                torchify(state).unsqueeze(0),
                torchify(action).unsqueeze(0))

            if critic_val > exp.exp_cfg.eps_safe:
                return True
            return False

        state = exp.env.reset()
        print(state)
        waypoints = [state]
        episode_steps = 0
        episode_reward = 0
        done = False
        while not done:
            action = exp.agent.select_action(state, eval=True)  # Sample action from policy
            if recovery_thresh(state, action):
                recovery = True
                real_action = exp.agent.safety_critic.select_action(state)
            else:
                recovery = False
                real_action = np.copy(action)
            next_state, reward, done, info = exp.env.step(real_action)  # Step

            done = done or episode_steps == exp.env.max_episode_steps

            episode_reward += reward
            episode_steps += 1
            state = next_state
            waypoints.append(state)
        print("Recovery", recovery)
        print("Reward:",reward)

        return waypoints

if __name__ == "__main__":
    with open("./trained/model.pkl", "rb") as f:
        print("Loading")
        experiment = torch.load(f, pickle_module = pickle)
    print("Loaded")
    waypoints = get_nav_rollout(experiment)
    print(waypoints)
    new_waypoints = np.array(waypoints)
    pickle.dump(new_waypoints, open("./trained/trajectory.pkl", "wb"))


