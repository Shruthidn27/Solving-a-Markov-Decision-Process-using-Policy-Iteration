# Solving a Markov Decision Process using Policy Iteration

## Aim

To implement the Policy Iteration algorithm for solving a finite Markov Decision Process using the Gymnasium FrozenLake-v1 environment, by repeatedly performing policy evaluation and policy improvement to obtain the optimal value function and optimal policy.

---

## Problem Statement

In this experiment, the `FrozenLake-v1` environment is solved using the **Policy Iteration** algorithm.

The agent starts from the start state and must reach the goal state without falling into holes. The environment is represented as a finite Markov Decision Process. Policy Iteration is used to repeatedly evaluate the current policy and improve it until the policy becomes stable.

The objective is to find:

1. The optimal state-value function $V^*(s)$
2. The optimal policy $pi^*(s)$

---

## Software Requirements

```bash
pip install gymnasium numpy
```

---

## Environment Description

The experiment uses the Gymnasium `FrozenLake-v1` environment.

FrozenLake is a grid-world environment where the agent moves over frozen tiles and tries to reach the goal without falling into holes.

For the default 4 × 4 FrozenLake map:

| Component | Description |
|---|---|
| Environment | `FrozenLake-v1` |
| Map size | 4 × 4 |
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching the goal, 0 otherwise |
| Terminal states | Goal and hole states |

---

## Theory

Policy Iteration is a Dynamic Programming method used to find the optimal policy of a Markov Decision Process.

It consists of two major steps:

1. **Policy Evaluation**
2. **Policy Improvement**

These two steps are repeated until the policy becomes stable.

---

## Policy Evaluation

Policy evaluation estimates the value function for the current policy.

The Bellman expectation equation is:

$$
V^\pi(s) =
\sum_a \pi(a \mid s)
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $pi(a \mid s)$ | Probability of taking action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $pi$ |

---

## Policy Improvement

Policy improvement updates the policy greedily with respect to the current value function.

The improved policy is obtained as:

$$
\pi'(s) =
\arg\max_a
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

If the improved policy is the same as the old policy, the policy is considered stable.

---

## Algorithm

1. Create the Gymnasium `FrozenLake-v1` environment.
2. Initialize a random policy.
3. Repeat until the policy becomes stable:
   - Evaluate the current policy using iterative policy evaluation.
   - Improve the policy greedily using the current value function.
   - Compare the old policy and the new policy.
4. Stop when the policy does not change.
5. Display the optimal value function and optimal policy.

---

## Python Program

```python
import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=True)
env = env.unwrapped


actions = {
    0: "Left",
    1: "Down",
    2: "Right",
    3: "Up"
}


def display_policy(policy, title):

    print("\n" + title)

    policy_grid = np.argmax(policy, axis=1).reshape(4,4)

    for row in policy_grid:
        print([actions[a] for a in row])
# -------------------------------------------------
# Policy Evaluation
# -------------------------------------------------
def policy_evaluation(env, policy, gamma=0.5, theta=1e-8):
    n_states = env.observation_space.n
    V = np.zeros(n_states)
    while True:
        delta = 0
        for state in range(n_states):
            value = 0
            for action, prob_action in enumerate(policy[state]):
                for prob, next_state, reward, done in env.P[state][action]:
                    value += prob_action * prob * (
                        reward + gamma * V[next_state] * (not done)
                    )
            delta = max(delta, abs(value - V[state]))
            V[state] = value
        if delta < theta:
            break
    return V
# -------------------------------------------------
# Policy Improvement
# -------------------------------------------------
def policy_improvement(env, V, gamma=0.5):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    improved_policy = np.zeros((n_states,n_actions))
    for state in range(n_states):
        action_values = np.zeros(n_actions)
        for action in range(n_actions):
            for prob, next_state, reward, done in env.P[state][action]:
                action_values[action] += prob * (
                    reward + gamma * V[next_state] * (not done)
                )
        best_action = np.argmax(action_values)
        improved_policy[state][best_action] = 1
    return improved_policy

#-------------------------------------------------
# Policy Iteration
# -------------------------------------------------
def policy_iteration(env):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    fixed_action = 2
    policy = np.zeros((n_states,n_actions))
    for state in range(n_states):
        policy[state][fixed_action] = 1
    iteration = 0
    while True:
        iteration += 1
        print("\n================================")
        print("POLICY ITERATION :", iteration)
        print("================================")
        display_policy(
            policy,
            "Policy Before Evaluation"
        )
        V = policy_evaluation(
            env,
            policy
        )
        print("\nValue Function After Evaluation:")
        print(np.round(V.reshape(4,4),3))
        new_policy = policy_improvement(
            env,
            V
        )
        display_policy(
            new_policy,
            "Policy After Improvement"
        )
        if np.array_equal(policy,new_policy):
            print("\nPolicy Stable")
            break
        policy = new_policy
    return policy,V



optimal_policy, optimal_value = policy_iteration(env)
print("\n\n========== FINAL OUTPUT ==========")
display_policy(
    optimal_policy,
    "Optimal Policy"
)
print("\nOptimal Value Function:")
print(
    np.round(
        optimal_value.reshape(4,4),
        3
    )
)
env.close()




```

## Output

```
================================
POLICY ITERATION : 1
================================

Policy Before Evaluation
['Right', 'Right', 'Right', 'Right']
['Right', 'Right', 'Right', 'Right']
['Right', 'Right', 'Right', 'Right']
['Right', 'Right', 'Right', 'Right']

Value Function After Evaluation:
[[0.    0.    0.002 0.   ]
 [0.001 0.    0.012 0.   ]
 [0.005 0.027 0.071 0.   ]
 [0.    0.088 0.414 0.   ]]

Policy After Improvement
['Down', 'Up', 'Left', 'Down']
['Left', 'Left', 'Left', 'Left']
['Up', 'Down', 'Left', 'Left']
['Left', 'Right', 'Down', 'Left']

================================
POLICY ITERATION : 2
================================

Policy Before Evaluation
['Down', 'Up', 'Left', 'Down']
['Left', 'Left', 'Left', 'Left']
['Up', 'Down', 'Left', 'Left']
['Left', 'Right', 'Down', 'Left']

Value Function After Evaluation:
[[0.    0.001 0.003 0.001]
 [0.001 0.    0.013 0.   ]
 [0.006 0.029 0.077 0.   ]
 [0.    0.089 0.418 0.   ]]

Policy After Improvement
['Right', 'Up', 'Left', 'Up']
['Left', 'Left', 'Left', 'Left']
['Up', 'Down', 'Left', 'Left']
['Left', 'Right', 'Down', 'Left']

================================
POLICY ITERATION : 3
================================

Policy Before Evaluation
['Right', 'Up', 'Left', 'Up']
['Left', 'Left', 'Left', 'Left']
['Up', 'Down', 'Left', 'Left']
['Left', 'Right', 'Down', 'Left']

Value Function After Evaluation:
[[0.    0.001 0.003 0.001]
 [0.001 0.    0.013 0.   ]
 [0.006 0.029 0.077 0.   ]
 [0.    0.089 0.418 0.   ]]

Policy After Improvement
['Right', 'Up', 'Right', 'Up']
['Left', 'Left', 'Left', 'Left']
['Up', 'Down', 'Left', 'Left']
['Left', 'Right', 'Down', 'Left']

================================
POLICY ITERATION : 4
================================

Policy Before Evaluation
['Right', 'Up', 'Right', 'Up']
['Left', 'Left', 'Left', 'Left']
['Up', 'Down', 'Left', 'Left']
['Left', 'Right', 'Down', 'Left']

Value Function After Evaluation:
[[0.    0.001 0.003 0.001]
 [0.001 0.    0.013 0.   ]
 [0.006 0.029 0.077 0.   ]
 [0.    0.089 0.418 0.   ]]

Policy After Improvement
['Down', 'Up', 'Right', 'Up']
['Left', 'Left', 'Left', 'Left']
['Up', 'Down', 'Left', 'Left']
['Left', 'Right', 'Down', 'Left']

================================
POLICY ITERATION : 5
================================

Policy Before Evaluation
['Down', 'Up', 'Right', 'Up']
['Left', 'Left', 'Left', 'Left']
['Up', 'Down', 'Left', 'Left']
['Left', 'Right', 'Down', 'Left']

Value Function After Evaluation:
[[0.    0.001 0.003 0.001]
 [0.001 0.    0.013 0.   ]
 [0.006 0.029 0.077 0.   ]
 [0.    0.089 0.418 0.   ]]

Policy After Improvement
['Down', 'Up', 'Right', 'Up']
['Left', 'Left', 'Left', 'Left']
['Up', 'Down', 'Left', 'Left']
['Left', 'Right', 'Down', 'Left']

Policy Stable


========== FINAL OUTPUT ==========

Optimal Policy
['Down', 'Up', 'Right', 'Up']
['Left', 'Left', 'Left', 'Left']
['Up', 'Down', 'Left', 'Left']
['Left', 'Right', 'Down', 'Left']

Optimal Value Function:
[[0.    0.001 0.003 0.001]
 [0.001 0.    0.013 0.   ]
 [0.006 0.029 0.077 0.   ]
 [0.    0.089 0.418 0.   ]]


```



---

## Result

```
The Policy Iteration algorithm was successfully implemented for the FrozenLake-v1 environment.A fixed initial policy was evaluated, improved, and iteratively
updated until the policy became stable.The optimal state-value function and the optimal policy for the FrozenLake environment were obtained successfully.



```
---

## Inference
```
The Policy Iteration algorithm was successfully implemented for the FrozenLake-v1 environment using a fixed initial policy.The algorithm first evaluated the given policy using the Bellman expectation equation to calculate the state-value function. Then, the policy improvement step updated the policy by selecting the
optimal action based on the calculated value function.By repeatedly performing policy evaluation and policy improvement,the algorithm converged to a stable optimal policy. This experiment demonstrates that Policy Iteration can efficiently solve finite Markov Decision Processes by finding the optimal value function and
optimal strategy for the agent.


```
---

