# Reinforcement-Learning

This is meant to be a collection of notes o  the basics of Reinforcement Learning slowly building up to more advanced concepts. The notes are based on the book [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) by Richard S. Sutton and Andrew G. Barto, the video lectures by [David Silver](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) and Spinning Up in Deep RL by OpenAI.

## <strong>Table of Contents</strong>
1. [Basic Terms and Notation](#chap1)
2. [The RL Problem](#chap2)
3. [Bellman Equations](#chap3)
4. [Types of RL Algorithms](#chap4)
5. [Policy Optimization](#chap5)


---
---
## Basic Terms and Notation <a href="chap1"></a>

The main components of RL are agents and the environment. The environment is essentially the world that the agent interacts with. 

<strong>States and Observations: </strong> A state $s$ is a descrription of the state of the world. $s_{t}$ represents the state of the world at some time step t. On the other hand an  <strong>Observation</strong> $o$ only contains partial information about the world.
\
\
<strong>Action Space: </strong> Essentially the set of possible actions that your agent can take. For example movement of a body in a grid world, the action space could be $(up, down, left, right)$. 
\
\
<strong>Policy: </strong> The rule used by the agent to decide which action to take. It can either be deterministic or stochastic. In the case that it's deterministic it can be written as: 

$a_{t} = \mu(s_{t})$

And in the case that it's stochastic it can be written as:

$a_{t} = \pi(.|s_{t})$

In deep RL the policies are parameterized. In simpler terms the policy is a function whose output depends on a set of parameters like the weights of a neural network. In this case the policy is represented as follows:

$a_{t} = \mu_{\theta}(s_{t})$
\
$a_{t} \thicksim \pi_{\theta}(.|s_{t})$

<strong>Trajectories: </strong> A trajectory $\tau$  is a sequence of actions and states in the world:

$\tau = (s_{0}, a_{0}, s_{1}, a_{1},...)$

The initial state of the world $s_{0}$ is randomly sampled from the initial state distribution $s_{0} \thicksim \rho_{0}(.)$

The state transition can be written as $s_{t+1} \thicksim P(.|s_{t}, a_{t})$ where $P$ is the state transition probability distribution.

<strong>Rewards: </strong> The reward function is the feedback signal that informs the agent how good a certain state and action pair are for reaching its goal. It can be written as:

$r_{t} = R(s_{t}, a_{t}, s_{t+1})$ 

But is also commonly written just as: 

$r_{t} = R(s_{t}, a_{t})$

<strong>Return: </strong> The goal of RL is to maximize some form of the cumulative return. The return $G_{t}$ is the total discounted reward from time step $t$ onwards. It can be written as:

$G_{t} = \sum_{k=0}^{\infty} \gamma^{k}r_{t+k+1}$

Where $\gamma$ is the discount factor. The discount factor is a value between 0 and 1 that determines how much the agent values future rewards. A discount factor of 0 means that the agent only cares about the immediate reward and a discount factor of 1 means that the agent cares about all future rewards equally.

---
---
## The RL Problem <a href="chap2"></a>

The goal as stated earlier, is to maximize the expected return. Value functions are exactly that, the expected return. There are four types of value functions:

1. State-Value Function: $V_{\pi}(s) = E_{\pi}[G_{t}|s_{t} = s]$

2. Action-Value Function: $Q_{\pi}(s, a) = E_{\pi}[G_{t}|s_{t} = s, a_{t} = a]$

3. Optimal State-Value Function: $V_{*}(s) = max_{\pi}V_{\pi}(s)$

4. Optimal Action-Value Function: $Q_{*}(s, a) = max_{\pi}Q_{\pi}(s, a)$

<strong>The optimal Q-function and action:</strong>

By definiton $Q_{*}(s, a)$ is the maximum expected return achievable by following any policy, after seeing some state $s$ and taking some action $a$, and thereafter following the optimal policy. Therefore, it makes sense to choose the action that maximizes the Q-function:

$a_{*}(s) = argmax_{a}Q_{*}(s, a)$


\
<strong>The Markov Property:</strong>

The Markov property states that the future is independent of the past given the present. Mathematically it can be written as:

$P[s_{t+1}|s_{t}] = P[s_{t+1}|s_{t}, s_{t-1}, s_{t-2},.....s_{1}]$

We can write the state transition in the form of the state transition matrix $P$:

$P_{ss'} = P[s_{t+1} = s'|s_{t} = s]$

The state transition matrix can be written as:

$$
\begin{pmatrix}
  P_{11}       & P_{12}     & P_{13}     & \cdots  & P_{1n}    \\
  P_{21}        & P_{22}    & P_{23}     & \cdots  & P_{2n}    \\
  \vdots  & \vdots  & \vdots  & \ddots  & \vdots \\
  P_{n1}       & P_{n2}    & P_{n3}     & \cdots  & P_{nn}   \\
\end{pmatrix}
$$
\
<strong>Markov Reward Process: </strong> A Markov Reward Process is a tuple $(S, P, R, \gamma)$ where $S$ is a finite set of states, $P$ is the state transition matrix, $R$ is the reward function and $\gamma$ is the discount factor. The return $G_{t}$ is the total discounted reward from time step $t$ onwards.

In the previous section we touched upon the definiton of the value function and the different types of value funtions. We can further expand the basic equation as follows:

$V_{\pi}(s) = E_{\pi}[G_{t}|s_{t} = s] = E_{\pi}[\sum_{k=0}^{\infty} \gamma^{k}r_{t+k+1}|s_{t} = s] \\ \forall s \ \epsilon \ S $

<strong>Note: The return is stochastic while the value is not stochastic</strong>

The state value function can also be written in terms of the action value function:

$V_{\pi}(s) = \sum_{a} \pi(a|s)Q_{\pi}(s, a)$

---
---
## Bellman Equations <a href="chap3"></a>

The Bellman equations are a set of equations that describe the relationship between the value of a state and the value of its successor states. The Bellman equations are the key to solving the RL problem.
\
\
It helps us find the optimal policies and value functions. The optimal value function is the maximum value function over all policies. The optimal policy is the policy that maximizes the value function. We can decompse the Bellman equation into two parts:

1. The immediate reward
2. The discounted value of the successor state

The Bellman equation can be defined as:

$$V_{\pi}(s) = E_{\pi}[G_{t}|s_{t} = s] = E_{\pi}[r_{t+1} + \gamma V_{\pi}(s_{t+1})|s_{t} = s]$$

The Bellman equation can also be written in terms of the action value function:

$$Q_{\pi}(s, a) = E_{\pi}[r_{t+1} + \gamma Q_{\pi}(s_{t+1}, a_{t+1})|s_{t} = s, a_{t} = a]$$

To expand upon this, consider an example. Suppose there is a robot in state $s$ and then moves to another state $s'$. The question is how good is it for the robot to be in state $s$. Using the Bellman equation we can find that it is the expectation of the reward it got on leaving state $s$ plus the value of state $s'$.

$$v(s) = R_{s} + \gamma \sum_{s' \in S} P_{ss'}v(s')$$

The above equation can be written in matrix form as:

$$v = R + \gamma Pv $$

Where $P_{s}$ is the state transition matrix for state $s$.

We can further modify the above equation as follows

$$(1-\gamma P)v = R$$
$$v = (1 - \gamma P)^{-1}R$$

<strong>Bellman Equation for state value function (in terms of state-action value function):</strong>

$$V_{\pi}(s) = \sum_{a} \pi(a|s)\sum_{s'}P^{a}_{ss'}(r(s,a) + \gamma V_{\pi}(s'))$$

<strong>Bellman Equation for action value function (in terms of state value function):</strong>

$$Q_{\pi}(s,a) = \sum_{s'} P^{a}_{ss'}(r(s,a) + \gamma \sum_{a'} \pi(a'|s')Q_{\pi}(s',a'))$$

This can further be simplified to:

$$Q_{\pi}(s,a) = \sum_{s'} P^{a}_{ss'}(r(s,a) + \gamma V_{\pi}(s'))$$

<strong>Bellman Optimality Equations: </strong> The Bellman optimality equations are the Bellman equations for the optimal value functions. The optimal value functions are the maximum value functions over all policies. The optimal policy is the policy that maximizes the value function. The Bellman optimality equations are as follows:

$$V_{*}(s) = max_{a} \sum_{s'} P^{a}_{ss'}(r(s,a) + \gamma V_{*}(s'))$$

$$Q_{*}(s,a) = \sum_{s'} P^{a}_{ss'}(r(s,a) + \gamma max_{a'}Q_{*}(s',a'))$$

---
---
## Types of RL Algorithms <a href="chap4"></a>

RL algorithms are classified as either <strong>model-free</strong> or <strong>model-based</strong> algorithms. A model of the environment is essentially a function which can predict the state transitions and the rewards. 

The main advantage of model-based algorithms is that it allows the agent to plan ahead. However the ground-truth model of the environment is usually not avialable to the agent and hence it must learn this model from experience. 

<strong>Policy Optimization: </strong> In this method the parameters $\theta$ of the policy $\pi_{\theta}(a|s)$ are optimized to maximize the expected return $J(\pi_{\theta})$ through either gradient ascent or indirectly by maximizing a local approximation of $J(\pi_{\theta})$. It also usually involves learning a value function $V_{\phi}(s)$ or $Q_{\phi}(s,a)$ to reduce the variance of the policy gradient estimator, which gets used in figuring out how to update the policy.

<strong>Q-Learning: </strong> The agent learns the optimal action-value function $Q^{*}(s,a)$ directly without learning the policy $\pi(a|s)$. The agent uses the action-value function to select the optimal action at each step. The agent learns the optimal action-value function by using the Bellman optimality equation as an iterative update. An approximator $Q_{\theta}(s,a)$ is learned for $Q_{*}(s,a)$ and the action taken by the Q-learning agent is given by:

$$a_{t} = argmax_{a}Q_{\theta}(s_{t},a)$$

---
---
## Policy Optimization <a href="chap5"></a>

The aim of policy optimization is to maximize the the expected return $J(\pi_{\theta}) = E_{\tau \thicksim \pi _{\theta}}[R(\tau)]$

We can optimize the policy by gradient ascent as foollows: 

$$\theta_{t+1} = \theta_{t} + \alpha \nabla_{\theta}J(\pi_{\theta})$$

Where $\alpha$ is the learning rate.

Some important notations, theorems and formulae that will be used are as follows:

1. <strong>Probability of Trajectory: </strong> The probability of a trajectory $\tau$ is given by:

$$P(\tau) = P(s_{0})\prod_{t=0}^{T}\pi_{\theta}(a_{t}|s_{t})P(s_{t+1}|s_{t},a_{t})$$

2. <strong>The Log Probability Trick: </strong> Making use of the chain rule we can write:

$$\nabla_{\theta}P(\tau) = P(\tau)\nabla_{\theta}logP(\tau)$$

3. <strong>Gradients of Environment Functions: </strong> The environment has no dependence on $\theta$ and hence the gradients of $\rho(s_0), P(t+1|s_t,a_t)$ and $R(\tau)$ are zero.

4. <strong>The Log Probability of a Trajectory: </strong> 

$$log P(\tau|\theta) = log P(s_{0}) + \sum_{t=0}^{T}(log\pi_{\theta}(a_{t}|s_{t}) + logP(s_{t+1}|s_{t},a_{t}))$$

5. <strong>Gradient Log Probability of a Trajectory: </strong> 

$$\nabla_{\theta}logP(\tau|\theta) = \sum_{t=0}^{T}\nabla_{\theta}log\pi_{\theta}(a_{t}|s_{t})$$

Using the above formulae we can derive the basic policy gradient algorithm as follows:

$$\nabla_{\theta}J(\pi_{\theta}) = \nabla_{\theta}E_{\tau \thicksim \pi_\theta}[R(\tau)]$$
$$ = \nabla_{\theta}\int_{\tau}P(\tau)R(\tau)$$
$$ = \int_{\tau}\nabla_{\theta}P(\tau)R(\tau)$$
$$ = \int_{\tau}P(\tau)\nabla_{\theta}logP(\tau)R(\tau)$$
$$ = E_{\tau \thicksim \pi_\theta}[\nabla_{\theta}logP(\tau)R(\tau)]$$

Since we are calculating the expectation, we can estimate it using a sample mean. We can collect a set of trajectories $D = \{\tau_i\}_{i=1,2,...N}$

Hence the policy gradient update rule becomes:

$$\nabla_{\theta}J(\pi_{\theta}) = \frac{1}{|D|}\sum_{\tau \epsilon D}\sum_{t=0}^{T}\nabla_{\theta}log\pi_{\theta}(a_{t}|s_{t})R(\tau)$$


