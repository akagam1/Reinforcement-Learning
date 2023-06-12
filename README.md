# Reinforcement-Learning

This is meant to be a collection of notes o  the basics of Reinforcement Learning slowly building up to more advanced concepts. The notes are based on the book [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) by Richard S. Sutton and Andrew G. Barto, the video lectures by [David Silver](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) and Spinning Up in Deep RL by OpenAI.

## <strong>Table of Contents</strong>
1. [Basic Terms and Notation](#chap1)
2. [The RL Problem](#chap2)
3. [Bellman Equations](#chap3)


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

$$V_{\pi}(s) = \sum_{a} \pi(a|s)\sum_{s'}P^{a}_{ss'}(r(s,a) + \gamma V_{\pi}(s')$$

