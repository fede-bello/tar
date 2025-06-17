import numpy as np


def q_learning_algorithm(
    S=10,
    theta=0.5,
    C=100,
    gamma=0.9,
    alpha=0.1,
    epsilon=0.1,
    max_iterations=50000,
    s0=10,
    decreasing_params=False,
):
    """
    Ejecuta el algoritmo Q-learning para el problema de reemplazo de máquinas.

    Args:
        S (int): Número de estados.
        theta (float): Probabilidad de degradación de la máquina.
        C (float): Costo de reemplazo.
        gamma (float): Factor de descuento.
        alpha (float): Tasa de aprendizaje.
        epsilon (float): Probabilidad de exploración para política epsilon-greedy.
        max_iterations (int): Número máximo de iteraciones a ejecutar.
        tolerance (float): Tolerancia de convergencia para valores Q.
        s0 (int): Estado inicial.
        decreasing_params (bool): Si es True, usa alpha y epsilon decrecientes.

    Returns:
        tuple: Una tupla que contiene:
            - q_history (np.ndarray): Historial de tablas Q en cada iteración.
            - actions_taken (list): Secuencia de acciones tomadas durante el aprendizaje.
    """
    q_table = np.zeros((S, 2))
    q_history = [q_table.copy()]
    actions_taken = []

    current_state = s0

    for k in range(max_iterations):
        current_alpha = alpha / (k + 1) if decreasing_params else alpha
        current_epsilon = epsilon / (k + 1) ** (2 / 3) if decreasing_params else epsilon

        if np.random.rand() < current_epsilon:
            current_action = np.random.choice([1, 2])
        else:
            q_values = q_table[current_state - 1, :]
            current_action = np.argmin(q_values + np.random.rand(2) * 1e-6) + 1

        actions_taken.append(current_action)

        if current_action == 1:
            cost = C
            next_state = S
        else:
            cost = (S - current_state) / (S - 1.0)
            if np.random.rand() < theta:
                next_state = max(1, current_state - 1)
            else:
                next_state = current_state

        q_current_val = q_table[current_state - 1, current_action - 1]
        min_q_next = np.min(q_table[next_state - 1, :])

        new_q_val = q_current_val + current_alpha * (
            cost + gamma * min_q_next - q_current_val
        )
        q_table[current_state - 1, current_action - 1] = new_q_val

        q_history.append(q_table.copy())

        current_state = next_state

        # no use tolerancia porque me convergia muy rapido con los valores de la letra

    return np.array(q_history), actions_taken


def value_iteration(S=10, theta=0.5, C=100, gamma=0.9, max_iter=10000, tol=1e-5):
    """
    Calcula la función de valor óptima usando iteración de valor.
    """
    V = np.zeros(S)
    policy = np.zeros(S, dtype=int)
    actions_taken = []

    for k in range(max_iter):
        cost_keep = (S - np.arange(1, S + 1)) / (S - 1.0)

        q_replace = C + gamma * V[S - 1]

        q_keep = np.zeros(S)
        for s in range(1, S + 1):
            s_idx = s - 1
            v_next_if_kept = theta * V[max(1, s - 1) - 1] + (1 - theta) * V[s_idx]
            q_keep[s_idx] = cost_keep[s_idx] + gamma * v_next_if_kept

        V = np.minimum(q_replace, q_keep)

        action = 1 if q_replace < q_keep[S - 1] else 2
        actions_taken.append(action)

    q_replace_final = C + gamma * V[S - 1]
    policy = np.argmin(np.vstack([np.full(S, q_replace_final), q_keep]), axis=0) + 1

    return V, policy, actions_taken


import matplotlib.pyplot as plt

S = 10
THETA = 0.5
C = 10
GAMMA = 0.9
ALPHA = 0.04
EPSILON = 0.1
MAX_ITER = 20000
TOL = 0.0000001

v_star, pi_star, vi_actions = value_iteration(
    S=S, theta=THETA, C=C, gamma=GAMMA, max_iter=MAX_ITER, tol=TOL
)

q_history, ql_actions = q_learning_algorithm(
    S=S,
    theta=THETA,
    C=C,
    gamma=GAMMA,
    alpha=ALPHA,
    epsilon=EPSILON,
    max_iterations=MAX_ITER,
    tolerance=TOL,
    s0=S,
)

v_from_q = np.zeros((len(q_history), S))
for i, q_table in enumerate(q_history):
    v_from_q[i, :] = np.min(q_table, axis=1)

plt.figure(figsize=(12, 7))

for i, state_to_plot in enumerate([1, 4, 8]):
    colors = ["red", "blue", "green"]
    plt.plot(
        v_from_q[:, state_to_plot - 1],
        label=f"Q-Learning V(s={state_to_plot})",
        color=colors[i],
    )
    plt.axhline(
        y=v_star[state_to_plot - 1],
        linestyle="--",
        label=f"Optimal V*(s={state_to_plot})",
        color=colors[i],
    )

plt.title("Q-Learning Value Function Trajectory vs. Optimal Value")
plt.xlabel("Iterations (k)")
plt.ylabel("Value V(s)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

plt.figure(figsize=(12, 7))
plt.plot(ql_actions[::10], label="Q-Learning", color="blue", alpha=0.5)
plt.plot(vi_actions[::10], label="Value Iteration", color="red", alpha=0.5)
plt.title("Actions Taken by Q-Learning and Value Iteration")
plt.xlabel("Iteration")
plt.ylabel("Action")
plt.legend()
plt.show()

q_history_decr, ql_actions_decr = q_learning_algorithm(
    S=S,
    theta=THETA,
    C=C,
    gamma=GAMMA,
    alpha=50,
    epsilon=0.5,
    max_iterations=MAX_ITER,
    tolerance=TOL,
    s0=S,
    decreasing_params=True,
)

v_from_q_decr = np.zeros((len(q_history_decr), S))
for i, q_table in enumerate(q_history_decr):
    v_from_q_decr[i, :] = np.min(q_table, axis=1)

plt.figure(figsize=(12, 7))
for i, state_to_plot in enumerate([1, 4, 8]):
    colors = ["red", "blue", "green"]
    plt.plot(
        v_from_q[:, state_to_plot - 1],
        label=f"Q-Learning V(s={state_to_plot})",
        color=colors[i],
        linestyle="-",
    )
    plt.plot(
        v_from_q_decr[:, state_to_plot - 1],
        label=f"Q-Learning Decreasing V(s={state_to_plot})",
        color=colors[i],
        linestyle=":",
    )
    plt.axhline(
        y=v_star[state_to_plot - 1],
        linestyle="--",
        label=f"Optimal V*(s={state_to_plot})",
        color=colors[i],
    )

plt.title("Q-Learning Value Function Trajectory vs. Optimal Value")
plt.xlabel("Iterations (k)")
plt.ylabel("Value V(s)")
plt.ylim(0, 12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


def simulate_cost(policy, S, theta, C, gamma, T=500, num_sims=10):
    """
    Simula el costo descontado para una política.
    """
    costs = []
    for _ in range(num_sims):
        s, cost, discount = S, 0, 1
        for _ in range(T):
            a = policy[s - 1]
            if a == 1:
                c, s = C, S
            else:
                c = (S - s) / (S - 1)
                s = max(1, s - 1) if np.random.rand() < theta else s
            cost += discount * c
            discount *= gamma
        costs.append(cost)
    return np.mean(costs)


print("\nExercise 4: Threshold Policy Analysis")
print("=" * 45)

threshold_costs = {}
for s_star in range(1, S + 1):
    policy = np.where(np.arange(1, S + 1) < s_star, 1, 2)
    cost = simulate_cost(policy, S, THETA, C, GAMMA)
    threshold_costs[s_star] = cost
    print(f"s*={s_star}: Cost={cost:.2f}")

best_threshold = min(threshold_costs, key=threshold_costs.get)
best_cost = threshold_costs[best_threshold]

ql_policy = np.argmin(q_history[-1], axis=1) + 1
ql_cost = simulate_cost(ql_policy, S, THETA, C, GAMMA)

print(f"\nResults:")
print(f"Best Threshold (s*={best_threshold}): {best_cost:.2f}")
print(f"Q-Learning: {ql_cost:.2f}")
print(f"Winner: {'Threshold' if best_cost < ql_cost else 'Q-Learning'}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(list(threshold_costs.keys()), list(threshold_costs.values()), "bo-")
plt.axhline(ql_cost, color="r", linestyle="--", label="Q-Learning")
plt.xlabel("Threshold s*")
plt.ylabel("Cost")
plt.title("Threshold Policy Performance")
plt.legend()

plt.subplot(1, 2, 2)
states = np.arange(1, S + 1)
best_policy = np.where(states < best_threshold, 1, 2)
plt.plot(states, best_policy, "g-o", label=f"Best Threshold (s*={best_threshold})")
plt.plot(states, ql_policy, "r-s", label="Q-Learning")
plt.xlabel("State")
plt.ylabel("Action")
plt.title("Policy Comparison")
plt.legend()
plt.tight_layout()
plt.show()
