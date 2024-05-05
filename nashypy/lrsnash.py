from fractions import Fraction
import warnings
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import importlib.resources as pkg_resources
from . import games


class NashGame:
    def __init__(self, payoffs_A, payoffs_B, game_name):
        self.payoffs_A = np.array(payoffs_A, dtype=float)
        self.payoffs_B = np.array(payoffs_B, dtype=float)
        self.name = game_name


def read_game(file_path=None, example=None):
    if example and file_path:
        raise ValueError(
            "Please specify either a file path or an example name, not both."
        )

    if example is None and file_path is None:
        # Dynamically list available example games
        available_examples = pkg_resources.contents(games)
        game_files = [file for file in available_examples if file.endswith(".txt")]
        print("Available example games:")
        for game in game_files:
            print(game[:-4])  # Print without '.txt' extension
        return None

    if file_path:
        game_name = file_path.split("/")[-1].split(".txt")[0]
        with open(file_path, "r") as file:
            blocks = file.read().split("\n\n")
    elif example:
        try:
            # Using importlib.resources to safely access package resources
            with pkg_resources.open_text(games, f"{example}.txt") as file:
                blocks = file.read().split("\n\n")
            game_name = example
        except FileNotFoundError:
            raise ValueError(
                f"The specified game '{example}' does not exist in the games directory."
            )

    if len(blocks) < 2:
        raise ValueError("Input file must contain two blocks of payoff matrices.")

    def process_block(block):
        return np.array(
            [list(map(Fraction, line.split())) for line in block.strip().split("\n")],
            dtype=float,
        )

    payoffs_A = process_block(blocks[0])
    payoffs_B = process_block(blocks[1])

    return NashGame(payoffs_A, payoffs_B, game_name)


def best_response(payoffs, strategy):
    """Compute the best response to a given strategy using the payoff matrix."""
    return np.eye(len(payoffs))[np.argmax(payoffs @ strategy)]


def is_nash_equilibrium(game, strategy1, strategy2):
    """Determine if the given strategies form a Nash equilibrium for the game."""
    payoff1 = np.dot(game.payoffs_A, strategy2) @ strategy1
    payoff2 = np.dot(game.payoffs_B.T, strategy1) @ strategy2

    # Compute the best response payoffs
    best_response_payoffs1 = np.max(np.dot(game.payoffs_A, strategy2))
    best_response_payoffs2 = np.max(np.dot(game.payoffs_B.T, strategy1))

    # Check if the actual payoffs are close to the best response payoffs
    is_nash1 = np.isclose(payoff1, best_response_payoffs1)
    is_nash2 = np.isclose(payoff2, best_response_payoffs2)

    return is_nash1 and is_nash2, payoff1, payoff2


def generate_and_sort_strategies(game, current_strategies, visited):
    """Generate and sort possible strategies for players."""
    m, n = game.payoffs_A.shape[0], game.payoffs_B.shape[1]
    possible_strategies = []

    for i in range(m):
        for j in range(n):
            new_tuple = (i, j)
            if new_tuple not in visited:
                new_p1_strategy = np.eye(m)[i]
                new_p2_strategy = np.eye(n)[j]
                possible_strategies.append((new_p1_strategy, new_p2_strategy))
                visited.add(new_tuple)

    # Sort the strategies lexicographically based on indices
    possible_strategies.sort(
        key=lambda x: (np.where(x[0] == 1)[0][0], np.where(x[1] == 1)[0][0])
    )
    return possible_strategies


def reverse_search(game, initial_strategy, method="direct"):
    visited = set()
    stack = [initial_strategy]
    results = []

    while stack:
        current_strategies = stack.pop()
        str_tuple = (tuple(current_strategies[0]), tuple(current_strategies[1]))

        if str_tuple not in visited:
            visited.add(str_tuple)
            if method == "direct":
                is_nash, payoff1, payoff2 = is_nash_equilibrium(
                    game, current_strategies[0], current_strategies[1]
                )
                if is_nash:
                    results.append(
                        (current_strategies[0], current_strategies[1], payoff1, payoff2)
                    )
                new_strategies = generate_and_sort_strategies(
                    game, current_strategies, visited
                )
                stack.extend(new_strategies)

    return results, visited


def find_nash_equilibria(game):
    """Find all Nash equilibria in the game using the reverse search method."""
    m, n = game.payoffs_A.shape[0], game.payoffs_B.shape[1]
    vertices = [(np.eye(m)[i], np.eye(n)[j]) for i in range(m) for j in range(n)]
    all_equilibria = set()
    total_visited = set()

    for vertex in vertices:
        str_vertex = (tuple(vertex[0]), tuple(vertex[1]))
        if str_vertex not in total_visited:
            equilibria, visited = reverse_search(game, vertex)
            total_visited.update(visited)
            for equilibrium in equilibria:
                p1_strategy, p2_strategy = map(np.array, equilibrium[:2])
                normalized_eq1 = (
                    tuple(p1_strategy / p1_strategy.sum())
                    if p1_strategy.sum() > 0
                    else tuple(p1_strategy)
                )
                normalized_eq2 = (
                    tuple(p2_strategy / p2_strategy.sum())
                    if p2_strategy.sum() > 0
                    else tuple(p2_strategy)
                )
                payoff1, payoff2 = equilibrium[2], equilibrium[3]
                all_equilibria.add((normalized_eq1, normalized_eq2, payoff1, payoff2))

    equilibria = list(all_equilibria)
    num_vertices = len(total_visited)
    print(f"Total vertices visited: {num_vertices}")
    for eq in equilibria:
        print(
            "Strategy 1: {}, Strategy 2: {}, Payoffs: ({}, {})".format(
                eq[0], eq[1], eq[2], eq[3]
            )
        )
    return equilibria, num_vertices


def solve_mixed_nash(game):
    A = game.payoffs_A
    B = game.payoffs_B

    delta_A = A[:, 0] - A[:, 1]
    delta_B = B[0, :] - B[1, :]

    # Player 1's mixed strategy (solve for q)
    if delta_A[0] - delta_A[1] != 0:
        q = (A[1, 1] - A[0, 1]) / (delta_A[0] - delta_A[1])
    else:
        q = None

    # Player 2's mixed strategy (solve for p)
    if delta_B[0] - delta_B[1] != 0:
        p = (B[1, 1] - B[1, 0]) / (delta_B[0] - delta_B[1])
    else:
        p = None

    # Check for pure strategy Nash equilibria
    pure_nash = find_pure_nash_equilibria(A, B)

    # Include mixed strategy if valid
    mixed_nash = {}
    if p is not None and q is not None:
        mixed_nash["p"] = p
        mixed_nash["q"] = q

    return pure_nash, mixed_nash


def find_pure_nash_equilibria(A, B):
    n_strategies_p1, n_strategies_p2 = A.shape
    pure_nash = []
    for i in range(n_strategies_p1):
        for j in range(n_strategies_p2):
            best_strategy_p1 = [
                k for k in range(n_strategies_p1) if A[k, j] == max(A[:, j])
            ]
            best_strategy_p2 = [
                k for k in range(n_strategies_p2) if B[i, k] == max(B[i, :])
            ]
            if i in best_strategy_p1 and j in best_strategy_p2:
                pure_nash.append(((i, j), (A[i, j], B[i, j])))
    return pure_nash


def plot_best_response_polyhedra(game, savefig=True):
    for player in ["1", "2"]:
        if player == "1":
            payoffs = game.payoffs_A
            opponent = "2"
        else:
            payoffs = game.payoffs_B.T
            opponent = "1"
            if payoffs.shape != game.payoffs_A.shape:
                warnings.warn(
                    "The shapes of payoffs A and B do not match. Cannot compute Player 1's polyhedra"
                )
                break

        y_values = np.linspace(0, 1, 100)
        payoff_computations = [np.dot(payoffs, [y, 1 - y]) for y in y_values]
        payoff_computations = np.array(payoff_computations)

        fig = go.Figure()
        for strategy_index, strategy_payoff in enumerate(payoff_computations.T):
            fig.add_trace(
                go.Scatter(
                    x=y_values,
                    y=strategy_payoff,
                    mode="lines",
                    line=dict(dash="dash"),
                    name=f"Strategy {strategy_index + 1} Payoff",
                )
            )

        best_responses = np.max(payoff_computations, axis=1)
        fig.add_trace(
            go.Scatter(
                x=y_values,
                y=best_responses,
                mode="lines",
                line=dict(width=3),
                name="Best Response Polytope",
            )
        )

        fig.update_layout(
            title=f"Best Response Polyhedra for Player {player} [{game.name}]",
            xaxis_title=f"Strategy Mix of Player {opponent}",
            yaxis_title=f"Payoff for Player {player}",
            legend_title="Legend",
            width=700,
            height=500,
        )
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)

        if savefig:
            os.makedirs("outputs", exist_ok=True)
            fig.write_image(f"outputs/polyhedra_{game.name}_player_{player}.png")
        fig.show()


def plot_best_response_polytope(game, savefig=True):
    for player in ["1", "2"]:
        if player == "1":
            payoffs = game.payoffs_A
            opponent = "2"
        else:
            payoffs = game.payoffs_B.T
            opponent = "1"
            if payoffs.shape != game.payoffs_A.shape:
                warnings.warn(
                    "The shapes of payoffs A and B do not match. Cannot compute Player 1's polytope"
                )
                break

        inequalities = [np.append(a, 1) for a in payoffs]
        inequalities = np.vstack((inequalities, [[-1, 0, 0], [0, -1, 0]]))

        y1 = np.linspace(0, 1, 100)
        y2_region = np.full_like(y1, np.inf)

        fig = go.Figure()
        num_payoff_constraints = len(payoffs)

        for index, ineq in enumerate(inequalities):
            if ineq[1] != 0:
                y2 = (ineq[2] - ineq[0] * y1) / ineq[1]
            else:
                y2 = np.full_like(y1, ineq[2] / ineq[0]) if ineq[0] != 0 else np.inf

            valid = y2 >= 0
            if index < num_payoff_constraints:
                y2_region = np.minimum(y2_region, y2, where=valid)

            fig.add_trace(
                go.Scatter(
                    x=y1[valid],
                    y=y2[valid],
                    mode="lines",
                    name=f"{ineq[0]:.2f}y1 + {ineq[1]:.2f}y2 <= {ineq[2]:.2f}",
                )
            )

        fig.add_trace(
            go.Scatter(
                x=np.concatenate([y1, y1[::-1]]),
                y=np.concatenate([np.zeros_like(y1), y2_region[::-1]]),
                fill="toself",
                fillcolor="lightblue",
                line_color="lightblue",
                name="Feasible Region",
                opacity=0.6,
            )
        )

        fig.update_layout(
            title=f"Player {opponent}'s Best Response Polytope [{game.name}]",
            xaxis_title="$y_1$",
            yaxis_title="$y_2$",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            legend_title="Legend",
            width=800,
            height=600,
        )

        os.makedirs("outputs", exist_ok=True)
        if savefig:
            fig.write_image(f"outputs/polytope_{game.name}_player_{player}.png")
        fig.show()


def main():

    game = read_game(example="2x2_avis")

    plot_best_response_polyhedra(game, savefig=True)
    plot_best_response_polytope(game, savefig=True)
    equilibria, num_vertices = find_nash_equilibria(game)


if __name__ == "__main__":
    main()
