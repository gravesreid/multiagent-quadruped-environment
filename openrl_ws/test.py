import isaacgym

from openrl.envs.common import make
from openrl.modules.common import PPONet
from openrl.runners.common import PPOAgent
from utils import make_mqe_env

env = make_mqe_env("go1gate")
net = PPONet(env, device="cuda")  # Create neural network.
agent = PPOAgent(net)  # Initialize the agent.
agent.load("./result/module_20m.pt")

agent.set_env(env)  # The agent requires an interactive environment.
obs = env.reset()  # Initialize the environment to obtain initial observations and environmental information.
while True:
    action, _ = agent.act(obs)  # The agent predicts the next action based on environmental observations.
    # The environment takes one step according to the action, obtains the next observation, reward, whether it ends and environmental information.
    obs, r, done, info = env.step(action)