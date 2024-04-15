# 1. Introduction
Some environments have a large number of possible actions. For example, consider playing Go on a $19\times19$ grid board. In this case, the number of possible initial moves is $19\times19 = 361$, and as each player takes turns, the number of possible moves decreases. Let's imagine creating an agent to play Go using a deep neural network. Then, how should we determine the state-action values or probabilities for each action based on where a stone is placed?

A simple approach would be to output a value for each of the $19\times19$ positions, representing the probability of placing a stone there and the value of placing a stone there. While straightforward, this approach has some drawbacks in the environment we've envisioned. Firstly, there's the issue of how to handle illegal actions. There are various ways to handle illegal actions, such as penalizing them or masking them to prevent them from being chosen. Secondly, always outputting values for all actions can be inefficient. How can we overcome these challenges? In this text, we'll address this by computing values for each action state pair that satisfies a one-to-one correspondence, thus resolving the issues. However, there are still some problems with this approach, particularly regarding learning methods and efficiency. Nonetheless, these issues can be addressed by combining policy function approximation with existing policies, defining action states, and exploring solutions for them.

# 2. Action State
Let's denote the set of all states as $\mathcal{S}$ and the set of all actions as $\mathcal{A}$. Then, we can define the following function for state-action pairs:

$\eta: \mathcal{S}\times \mathcal{A} \rightarrow \overline{S}$, where $\eta$ is bijective. We then define this function as the action-state map.

With this, we can define functions $\overline{Q}$ and $\overline{\pi}$ that satisfy the following relationships through the action value function $Q$ and policy function $\pi$:

$Q(s_t, a_t) = \overline{Q} \circ \eta(s_t, a_t)$

$\pi(s_t, a_t) = \overline{\pi} \circ \eta(s_t, a_t)$

Using the action-state map offers several advantages:

1. Provides useful information about actions given knowledge about the environment.

For example, Go is a deterministic environment, so if we define $\eta(s_t, a_t)$ as the action taken from state $s_t$ by taking action $a_t$, then $\overline{Q}(s_t, a_t) = r(s,  a) + \alpha V(s_{t+1})$. Even in non-deterministic environments, it can provide additional information about the outcome of taking an action. For instance, consider the card game BigTwo. Although it's not deterministic, by adding information about the remaining cards in hand after discarding, we can construct action-states. By including such information, even though we can't predict the next state, we gain more advantages than having no information at all.

![go](https://github.com/jh2525/jh2525.github.io/assets/160830734/634b2ab8-cbbc-43f0-b63c-9ee0d1c85880)

2. Handling various action shapes and illegal actions becomes easier.

Consider the Big Two game environment mentioned earlier. With numerous possible hand combinations, calculating logits and action values for every possible action poses challenges, especially with illegal actions.

In BigTwo, there are thousands of possible hand combinations. Hence, computing logits and state-action values for all combinations in every scenario is highly inefficient. However, utilizing the action-state map allows the agent to compute logits and state-action values only for actions it can take. Thus, regardless of the action shape or the existence of illegal actions, by focusing on feasible actions within the action state, the agent can easily handle the gap between feasible and total actions.

# 3. Approximation of $\pi$
when we try to train a policy using the action-state map, we face the challenge of having to recompute values for all action-states to obtain $\pi(a_i, s_i)$. However, this becomes resource-intensive since it involves computing values for actions not taken. If, during training, reinforcing one action doesn't significantly alter the logits of other actions, then during episodes, we can calculate the probability of new policy at that action-state using only the taken action-states.

Let's denote the logit when taking action $a$ in state $s$ as $l(s, a)$. Then, the probability using softmax to obtain $\pi(s, a)$ when the state is $s$ can be expressed as:

$\pi(s, a) = \frac{e^{l(s, a)}}{\sum_{i=1}^{n} e^{l(s, a_i)}}$

The issue we face now is having to compute logits for actions not taken. To overcome this, we'll store the value $\sum_{i=1}^{n} e^{l(s, a_i)} - e^{l(s,a)} = \rho(s, a)$ during inference and utilize it. Then, the probability can be calculated as:

$\pi(s, a) = \frac{e^{l(s, a)}}{\rho(s, a) + e^{l(s, a)}}$

Now, let's denote the policies and logits corresponding to two parameters $\theta_1$ and $\theta_2$ as $\pi_{\theta_1}$, $\pi_{\theta_2}$, $l_{\theta_1}$, and $l_{\theta_2}$, respectively. Similarly, let $\sum_{i=1}^{n} e^{l_{\theta_i}(s, a_i)} - e^{l_{\theta_i}(s,a)} = \rho_{\theta_i}(s, a)$ for $i = 1, 2$. Then, we define the policy $\pi_{\theta_1}^{\theta_2}$ for $\theta_1$ regarding $\theta_2$ as follows:

$$\pi_{\theta_1}^{\theta_2}(s, a) = \frac{e^{l_{\theta_2}(s, a)}}{\rho_{\theta_1}(s, a) + e^{l_{\theta_2}(s, a)}}$$

Then, the following holds:

**Theorem 3.1**
$$\left\vert\frac{\pi_{\theta_2}}{\pi_{\theta_1}^{\theta_2}} - 1\right\vert \leq \left\vert\frac{\rho_{\theta_1}}{\rho_{\theta_2}} -1\right\vert$$

***Proof)*** Since
$\displaystyle{
 \frac{\pi_{\theta_2}}{\pi_{\theta_1}^{\theta_2}} = 
 \frac{e^{l_{\theta_2}}}{\rho_{\theta_2} + e^{l_{\theta_2}}} \cdot \frac{\rho_{\theta_1} + e^{l_{\theta_2}}}{e^{l_{\theta_2}}} =
 \frac{\rho_{\theta_1} + e^{l_{\theta_2}}}{\rho_{\theta_2} + e^{l_{\theta_2}}} =
 \frac{\rho_{\theta_1} - \rho_{\theta_2}}{\rho_{\theta_2} + e^{l_{\theta_2}}} + 1 
}$ and $e^{l_{\theta_2}} > 0$,

$\displaystyle{
 -\frac{|\rho_{\theta_1} - \rho_{\theta_2}|}{\rho_{\theta_2}} = -\left\vert \frac{\rho_{\theta_1}}{\rho_{\theta_2}} - 1\right\vert
 \leq -\frac{|\rho_{\theta_1} - \rho_{\theta_2}|}{\rho_{\theta_2} + e^{l_{\theta_2}}}
 \leq\frac{\pi_{\theta_2}}{\pi_{\theta_1}^{\theta_2}} - 1 \leq
 \frac{|\rho_{\theta_1} - \rho_{\theta_2}|}{\rho_{\theta_2} + e^{l_{\theta_2}}} \leq
 \frac{|\rho_{\theta_1} - \rho_{\theta_2}|}{\rho_{\theta_2}} = \left\vert \frac{\rho_{\theta_1}}{\rho_{\theta_2}} - 1\right\vert
}$

Therefore, $\displaystyle{\left\vert\frac{\pi_{\theta_2}}{\pi_{\theta_1}^{\theta_2}} - 1\right\vert \leq \left\vert\frac{\rho_{\theta_1}}{\rho_{\theta_2}} -1\right\vert}$

The crucial point here is that $\rho$ is independent of the logit value chosen for our action. Therefore, if we assume that during training, we mainly alter the logits of actions taken, then using $\pi_{\theta_1}^{\theta_2}$ instead of $\pi_{\theta_2}$ during training is a good approximation. Hence, we only need to compute logits for the actions we take. This makes training approximately $n$ times more efficient if there are $n$ actions to choose from on average.

## Algorithm 3.1: Calculate action probability by using $\rho$

**Set** $\theta_1, \theta_2$, parameters of two policies $\pi_{\theta_1}, \pi_{\theta_2}$  
**Set** $\eta$, action state map  
**Set** $s$, state  
**Set** $\mathcal{A}$, all available actions in $s$  
**Set** $N$, the number of all available actions  
**Set** $a$, action in $\mathcal{A}$  
**Set** $\rho_{\theta_1} = 0$  
 
**For** $i = 1, 2, \dots, N$ do  
ㅤㅤcalculate $l_{\theta_1}(s, a_i)$ using $\eta(s, a_i)$  

**Set** $l_{\text{max}}$, the maximum of logits for $a_i$  
**Set** $\rho_{\theta_1} = \sum_{a_i \neq a}{e^{l_{\theta_1}(s, a_i) - l_{\text{max}}}}$  
$\pi_{\theta_2} \approx \pi_{\theta_1}^{\theta_2} = \frac{e^{l_{\theta_2}(s, a) - l_{\text{max}}}}{\rho_{\theta_1} + e^{l_{\theta_2}(s, a) - l_{\text{max}}}}$

# 4. Max Advantage Action Incentive

In cases where we do not use an approximation of $\pi$, there is a straightforward method to motivate the reduction of the gap between the maximum state-action value and the action with the maximum probability.

We can simply employ the '**max advantage action incentive**' defined as:

$$\text{max action incentive} = \beta\mathbb{E}[ A^{\pi}(s, \text{argmax}_aQ(s, a)) \log{\pi(s, \text{argmax}_aQ(s, a))}]$$

where $\beta > 0$.

There are some notable points:
- If $\pi$ is optimal, the max advantage action incentive is always 0.
- If the max advantage action incentive increases, $\pi(s, \text{argmax}_a Q(s, a))$ tends to increase.

# 5. Discussion

- **Why do we estimate state-action values instead of state values?**  
To calculate state values, only the state is required. However, in the case of action-states, they depend not only on the state but also on the action. Therefore, the simplest structure is to estimate state-action values rather than state values. Moreover, by estimating state-action values using this method, we can still calculate the policy probabilities, which is an advantage as it allows us to obtain state values. It is also possible to calculate state values using common states. For more details, refer to **Appendix B: State-value based action-state algorithm**.

- **Is it necessary to approximate the policy using Algorithm 3.1?**  
This algorithm is a method for fast computation in environments with many actions, but it is not always preferred. Especially when each action-state has a similar distance, $\rho$ tends to change accordingly. If accurate probability calculation is more important than computational efficiency, there is no need to use this algorithm.

- **Is the max advantage action incentive necessary?**  
Experimental results have shown that agents using learned policies perform better than greedy policies. In other words, state-action values can be overestimated. However, by providing the max advantage action incentive, we can further explore overestimated actions and more accurately determine if those actions truly have such values.

# 6. Results
I have been utilizing this algorithm in BigTwo and achieved superhuman performance despite having relatively small parameters. For further details, please refer to [BigTwo](https://github.com/jh2525/BigTwo). (but not use approximation of $\pi$)

# Appendix

## A. PPO with Action-State Map Algorithm

**Set** $\epsilon \geq 0$, the clipping variable  
**Set** $\epsilon_p \geq 0$, the probability clipping variable  
**Set** $\epsilon_{\pi}$, the greedy epsilon variable  
**Set** $K$, the number of epochs  
**Set** $N$, the number of actors  
**Set** $\mathcal{M}$, the replay buffer  
Randomly **initialize** the actor and critic parameters $\theta_1, \theta_2$  
**Initialize** the old actor network $\theta_{A_\text{old}}$  
**For** $i = 1, 2, \dots, N$ do  
ㅤㅤ**Set** $\theta_{A_\text{old}} = \theta_A$  
ㅤㅤ**Let** $\pi$ be the actor-based epsilon-greedy policy of $\pi_{\theta}$ as $\epsilon_p$   
ㅤㅤ**While** not truncated nor terminated do  
ㅤㅤㅤㅤ**Step** using policy $\pi$  
ㅤㅤㅤㅤ**Stack** $(\eta(s, a), r, v = \sum{\pi_{\theta}(s, a)Q(s, a)}, \log{\pi_{\theta}(s, a)}, \rho(s, a), l_\text{max})$ in $\mathcal{M}$  
ㅤㅤ**For** $j = 1, 2, \dots, K$ do  
ㅤㅤㅤㅤ**Sample** train data in $\mathcal{M}$  
ㅤㅤㅤㅤ**Optimize** $\theta_C$ by minimizing $(Q_{\theta_C}(s, a) - T(s, a))^2$  
ㅤㅤㅤㅤ**Calculate** probability ratio $r_p = \pi_{\theta}(s, a) / \pi_{\theta_{\text{old}}}(s, a) \approx \pi^{\theta}_{\theta_\text{old}}(s, a) / \pi_{\theta_{\text{old}}}(s, a)$ using Algorithm 5.1  
ㅤㅤㅤㅤ**Optimize** $\theta_A$ by minimizing $-\min(Ar_p, \text{clip}(Ar_p, 1-\epsilon, 1+\epsilon))$  
ㅤㅤ**Clear** $\mathcal{M}$

## B. State-Value Based Action-State Algorithm
The state-value based action-state algorithm calculates state values and action probabilities using two components instead of predicting state-action values as follows:

- Common state: Used solely for computing state values and together with action state for calculating action probabilities.
- Action state: Used in conjunction with common state for computing action probabilities.

![statevaluebasealgorithm](https://github.com/jh2525/jh2525.github.io/assets/160830734/24c7c07c-cca9-439b-9230-1eb7fd55810d)
## C. Recurrent Q-Network
The motivation behind recurrent Q-network is to train a neural network to learn the action-state map $\eta$. So, what is the most effective way to handle the action-state map $\eta$? The most basic idea is to predict the state when taking action $a$ in state $s$, denoted by $\eta(s, a)$. However, there are two problems with this approach. The first is that it is difficult to predict in non-deterministic environments, and the second is that it is difficult to define the distance between two states. Therefore, we decide to predict the encoded next state when $\eta(s, a)$ encodes the next state (i.e., we train an action-state encoder). This method effectively addresses the first problem, allowing for accurate predictions of the next state even in non-deterministic environments. If $\eta$ is well-trained, it will predict the encoded state containing the possibilities of the next state. However, the second problem has not yet been resolved. If we simply set the target value to be the encoded next state, the model will tend to minimize the encoding. Therefore, to address the second problem, we ensure that the same critic layer produces the same output when given the same input. In other words, if the state resulting from taking action $a$ in state $s$ is $s'$, we penalize such that $\overline{Q}(\eta(s, a)) = r(s,a) + \alpha V(s')$. This way, the model learns to produce the same value for the encoded $\eta(s, a)$ and $s'$, encouraging consistency in values.

![recurrentQ](https://github.com/jh2525/jh2525.github.io/assets/160830734/998f612c-4bdd-4c73-8f43-78b02ecbc39b)

# References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. 
2. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction (2nd ed.). The MIT Press.
3. Ravichandiran, S. (2020). Deep reinforcement learning in Python: A hands-on guide. Pearson Education.
