
###

## Multi-agent reinforcement learning
# Collaboration and competition

<img src="img/grab.jpg" width=20%>

### Context

This project demonstrates an approach to solve continuous space problem on the example of the Reacher environment. In the continuos action space the agent can take an action with any value, contrary to a discrete action-state space. For this project the Deep Deterministic Policy Gradient methods was used as an algorithm to train the agent(s).

#### Deep Deterministic Policy Gradient (DDPG)

DDPG belongs to the group of actor-critic methods, which leverage the strengths of both policy-based and value-based methods. It uses a stochastic behaviour policy for exploration and uses an estimate deterministic target policy.

The DDPG method relies on two networks: a “critic”, that estimates the value function or state-value, and an “actor” that updates the policy distribution in the direction suggested by the critic. The actor directly maps states to actions instead of returning a probability distribution across a discrete action space. The actor represents the application of the Policy Gradient method. The DDPG uses deterministic policy gradients on contrary to stochastic policy methods: "In the stochastic case, the policy gradient integrates over both state and action spaces, whereas in the deterministic case it only integrates over the state space." [Silver et al. (2014)](http://proceedings.mlr.press/v32/silver14.pdf).

##### DDPG pseudo-algorithm
After [Lillicrap et al. (2016)](https://arxiv.org/abs/1509.02971):
- random initialization of actor and critic networks
- initialize target network
- initialize replay buffer
- for all episodes:
	- initialize a random process
	- receive initial observation state
	- for all time instances:
		- select action considering current policy and noise
		- execute action and collect reward and get new states
		- store the experience tuple to the replay buffer
		- sample an experience tuple from replay buffer - at random
		- set the target
		- update critic using the loss function
		- update the actor policy using the sampled policy gradient
		- update the target network



### Approach

#### Environment

#### DDPG from previous project

### 3. Environment solutions

#### Hyperparameters used

The table list out the best set of parameters after many experiments:

| parameter | value |
| --- | --- |
| N nodes critic  | 256, 256|
| N nodes actor | 256, 256|
|BUFFER_SIZE replay buffer size | int(1e6)  |
|BATCH_SIZE minibatch size | 128        |
|GAMMA discount factor | 0.99            |
|TAU soft update of target parameters | 1e-3              |
|LR_ACTOR  learning rate of the actor | 1e-3         |
|LR_CRITIC learning rate of the critic| 1e-3     |
|WEIGHT_DECAY L2 weight decay | 0     |
|EPSILON  epsilon noise | 1.0        |
|EPSILON_DECAY decay rate for noise| 1e-6 |   
| sigma OUNoise | 0.2 |
| theta OUNoise | 0.15 |
| mu OUNoise | 0 |

### Future Work / Improvement points
Points to consider:
- share experiences between the two agents OR define to separate agents that are not sharing experiences between each other

- Prioritized experience replay
The problem of reproducibility of a successful training session (generalization)
Random sampling of the replay buffer - the `sample` method of the Replay Buffer class of the ddpg_agent is using random sampling of the stored experience tuples. It has been proven by [Schaul el al. (2016)](https://arxiv.org/pdf/1511.05952.pdf) that prioritization of the experiences has very benefitial effect on the generalization of the environment solution and successful training of the agents.
