import os
import warnings
from fractions import Fraction
import importlib.resources as pkg_resources
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from . import games


class NashGame:
    """
    Initialize a NashGame object with specified payoff matrices and a game name.

    Parameters:
        payoffs_A (list of lists): Payoff matrix for Player A.
        payoffs_B (list of lists): Payoff matrix for Player B.
        game_name (str): The name of the game.
    """

    def __init__(self, payoffs_A, payoffs_B, game_name):
        self.payoffs_A = np.array(payoffs_A, dtype=float)
        self.payoffs_B = np.array(payoffs_B, dtype=float)
        self.name = game_name


def read_game(file_path=None, example=None):
    """
    Read game data from a file or a packaged example name and create a NashGame
    instance.

    Parameters:
        file_path (str, optional): Path to the game file.
        example (str, optional): Name of the example game to use from package
        resources.
        Use nashypy.read_game() to get a list of example games.

    Returns:
        NashGame: An instance of NashGame constructed from the read data.
    """
    if example and file_path:
        raise ValueError(
            "Please specify either a file path or an example name, not both."
        )

    game_files = get_game_files(example, file_path)
    if not game_files:
        return None

    payoffs_A, payoffs_B = process_game_blocks(game_files)
    game_name = file_path.split("/")[-1].split(".txt")[0] if file_path else example
    return NashGame(payoffs_A, payoffs_B, game_name)


def get_game_files(example, file_path):
    """
    Helper function to fetch game files from resources or file system based on the provided
    file path or example name.

    Parameters:
        example (str, optional): Name of the example game.
        file_path (str, optional): File path to the game file.

    Returns:
        list: List of strings where each string represents a block of game data.
    """
    if file_path:
        return open(file_path, "r").read().split("\n\n")
    elif example:
        return pkg_resources.open_text(games, f"{example}.txt").read().split("\n\n")
    else:
        list_available_games()
        return []


def list_available_games():
    """
    List available example games from the package resources.
    """
    available_examples = pkg_resources.contents(games)
    game_files = [file for file in available_examples if file.endswith(".txt")]
    print("Available example games:")
    for game in game_files:
        print(game[:-4])


def process_game_blocks(blocks):
    """
    Process blocks of text into game matrices.

    Parameters:
        blocks (list of str): Text blocks containing the raw game data.

    Returns:
        tuple: A tuple containing two numpy arrays, one for each player's payoff matrix.
    """
    if len(blocks) < 2:
        raise ValueError("Input file must contain two blocks of payoff matrices.")
    return tuple(map(process_block, blocks))


def process_block(block):
    """
    Convert a block of text into a matrix of numbers.

    Parameters:
        block (str): A string containing the game data for one player.

    Returns:
        numpy.ndarray: A matrix where each entry is a float representing a game payoff.
    """
    return np.array(
        [list(map(Fraction, line.split())) for line in block.strip().split("\n")],
        dtype=float,
    )


def find_nash_equilibria(game):
    """
    Find all Nash equilibria in the game using the reverse search method. This method
    identifies equilibria by exploring all potential vertex pairs (pure strategies)
    and expands from these points using a reverse search to identify mixed strategy
    equilibria.

    Parameters:
        game (NashGame): The game for which Nash equilibria are to be determined.

    Returns:
        tuple: A tuple containing two elements:
               - A list of tuples, each representing a Nash equilibrium found.
               Each tuple contains:
                 - Two tuples representing the normalized strategy mix for player
                 1 and player 2, respectively.
                 - Payoff values for player 1 and player 2 corresponding to the
                 strategies.
               - An integer representing the number of vertices (strategy combinations)
               visited during the search.

    Notes:
        The output includes detailed print statements that describe the total
        number of vertices visited and the details of each equilibrium found,
        including the strategy mixes and their corresponding payoffs.

    Examples:
        # Assuming 'game' is an instance of NashGame:
        equilibria, num_vertices = find_nash_equilibria(game)
        print("Found equilibria:", equilibria)
        print("Vertices visited:", num_vertices)
    """
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


def reverse_search(game, initial_strategy, method="direct"):
    """
    Conduct a reverse search to identify Nash equilibria starting from an initial
    pure strategy. The method explores adjacent strategies, checking for Nash
    equilibria and expanding search from these points.

    Parameters:
        game (NashGame): The game object containing the payoff matrices.
        initial_strategy (tuple): A tuple containing two numpy arrays, representing
        the initial strategies for player 1 and player 2, respectively.
        method (str, optional): The method to use for checking if a strategy is a Nash
        equilibrium. Default is "direct".

    Returns:
        tuple: A tuple containing:
               - A list of tuples, each tuple representing a Nash equilibrium
                 (strategy pair and their respective payoffs for both players).
               - A set containing tuples of strategies that have been visited during the
                 search.

    Notes:
        - The function uses a stack to manage the search process, exploring strategies
          depth-first until all reachable strategies are exhausted.
        - The `method` parameter currently supports only the "direct" method, which
          checks strategies directly for being Nash equilibria.

    Examples:
        # Assuming 'game' is an instance of NashGame and 'initial_strategy' is defined:
        equilibria, visited = reverse_search(game, initial_strategy)
        print("Equilibria found:", equilibria)
        print("Strategies visited:", visited)
    """
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


def is_nash_equilibrium(game, strategy1, strategy2):
    """
    Determine if the specified strategies for two players constitute a Nash equilibrium
    in a two-player game.

    Parameters:
        game (NashGame): The game object containing the payoff matrices for both players.
        strategy1 (numpy.ndarray): The strategy vector for player 1.
        strategy2 (numpy.ndarray): The strategy vector for player 2.

    Returns:
        tuple: A tuple containing:
               - A boolean indicating if the strategies form a Nash equilibrium.
               - The payoff for player 1 when both players play these strategies.
               - The payoff for player 2 when both players play these strategies.

    Notes:
        A strategy pair is a Nash equilibrium if neither player can unilaterally switch
        to another strategy and achieve a higher payoff. This function checks if the
        payoffs from the given strategies are close to the best possible responses
        against each other's strategies.
    """
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
    """
    Generate new strategies not yet visited from the current strategies, and sort them
    for further exploration.

    Parameters:
        game (NashGame): The game object containing the payoff matrices.
        current_strategies (tuple): The current strategies of player 1 and player 2.
        visited (set): A set of tuples representing strategies that have already been
        explored.

    Returns:
        list: A list of new strategy pairs, sorted by player indices.

    Notes:
        This function generates potential new strategies by iterating over possible pure
        strategy combinations not yet visited. It is used during the reverse search to
        ensure comprehensive exploration of strategy space.
    """
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


def solve_mixed_nash(game):
    """
    Solve for both pure and mixed Nash equilibria in a two-player game, given specific
    payoff matrices. Handles both 2x2 and 3x3 matrices differently.

    Parameters:
        game (NashGame): The game object containing the payoff matrices for both players.

    Returns:
        tuple: A tuple containing:
               - A dictionary of any pure strategy Nash equilibria found.
               - A dictionary containing mixed strategy Nash equilibria if they exist,
                 otherwise empty.

    Notes:
        The function first checks if the game matrix size is 3x3 and handles it with a
        separate function. For 2x2 games, it calculates potential mixed strategies by
        solving linear equations derived from the payoff differences between strategies.
        It also checks for the existence of pure strategy Nash equilibria using a
        separate function.
    """
    A = game.payoffs_A
    B = game.payoffs_B

    if A.shape == (3, 3):
        A = sp.Matrix(A)
        B = sp.Matrix(B)
        solutions = solve_mixed_nash_3x3(A, B)
        analyze_solutions(solutions)
        return None, solutions

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


def solve_mixed_nash_3x3(A, B):
    """
    Solve for mixed Nash equilibria in a 3x3 game using symbolic computation. This
    function computes the equilibria by setting up and solving a system of equations
    based on the expected payoffs and probability constraints for both players.

    Parameters:
        A (sympy.Matrix): The 3x3 payoff matrix for Player 1.
        B (sympy.Matrix): The 3x3 payoff matrix for Player 2.

    Returns:
        list: A list of dictionaries, each representing a solution (Nash equilibrium)
        to the system of equations.
        Each solution is a mapping from the probability symbols
        (p1, p2, p3, q1, q2, q3) to their values that satisfy the Nash equilibrium
        conditions.

    Notes:
        The function defines probability variables for each strategy of the players
        (p1, p2, p3 for Player 1 and q1, q2, q3 for Player 2). It ensures these
        probabilities sum to 1 and sets up indifference conditionsm for each player's
        strategies. The indifference conditions ensure that if a player is mixing
        between strategies, then the expected payoffs from those strategies must be
        equal.

        The system of equations including these conditions and the probability
        constraints is then solved using SymPy's solver, which attempts to find values
        for the probability variables that satisfy all conditions.
    """
    A = sp.Matrix(A)
    B = sp.Matrix(B)
    # Define the probability variables for Player 1 and Player 2
    p1, p2, p3 = sp.symbols("p1 p2 p3", real=True, nonnegative=True)
    q1, q2, q3 = sp.symbols("q1 q2 q3", real=True, nonnegative=True)

    # Add constraints for probabilities summing to 1
    prob_constraint_1 = sp.Eq(p1 + p2 + p3, 1)
    prob_constraint_2 = sp.Eq(q1 + q2 + q3, 1)

    # Define expected payoffs for Player 1 and Player 2 using their strategies
    U1_1 = p1 * A[0, :] @ sp.Matrix([q1, q2, q3])
    U1_2 = p2 * A[1, :] @ sp.Matrix([q1, q2, q3])
    U1_3 = p3 * A[2, :] @ sp.Matrix([q1, q2, q3])

    U2_1 = q1 * sp.Matrix([p1, p2, p3]).T @ B[:, 0]
    U2_2 = q2 * sp.Matrix([p1, p2, p3]).T @ B[:, 1]
    U2_3 = q3 * sp.Matrix([p1, p2, p3]).T @ B[:, 2]

    # Set up indifference conditions (each strategy's payoff should be equal if it's
    # played with positive probability)
    indifference_1 = [sp.Eq(U1_1, U1_2), sp.Eq(U1_1, U1_3)]
    indifference_2 = [sp.Eq(U2_1, U2_2), sp.Eq(U2_1, U2_3)]

    # Solve the system of equations
    equations = indifference_1 + indifference_2 + [prob_constraint_1, prob_constraint_2]
    solutions = sp.solve(equations, (p1, p2, p3, q1, q2, q3), dict=True)
    return solutions


def analyze_solutions(solutions):
    """
    Analyze and filter out valid Nash equilibria from a list of potential solutions
    based on the probability constraints (0 <= probability <= 1 for each strategy).

    Parameters:
        solutions (list): A list of dictionaries where each dictionary represents a
        solution containing strategy probabilities as keys and their respective values.

    Notes:
        The function prints each solution and categorizes it as valid or invalid based
        on the probability constraints. It also prints a summary of valid Nash
        equilibria found or notes the absence of such equilibria.
    """
    valid_solutions = []
    for sol in solutions:
        # Check if all probabilities are between 0 and 1
        if all(0 <= v <= 1 for v in sol.values()):
            valid_solutions.append(sol)
            print("Valid Solution:", sol)
        else:
            print("Invalid Solution:", sol)

    if valid_solutions:
        print("\nValid equilibria found:")
        for sol in valid_solutions:
            print(sol)
    else:
        print("\nNo valid Nash equilibria found.")


def find_pure_nash_equilibria(A, B):
    """
    Find all pure strategy Nash equilibria for a given game defined by payoff matrices
    A and B.

    Parameters:
        A (numpy.ndarray): The payoff matrix for Player 1.
        B (numpy.ndarray): The payoff matrix for Player 2.

    Returns:
        list: A list of tuples, each tuple representing a pure strategy Nash equilibrium.
              Each tuple includes the strategy index pair and the associated payoffs for
              both players.

    Notes:
        The function iterates through all possible strategy pairs and determines if each
        pair is a Nash equilibrium by checking if each player's strategy is a best
        response to the strategy of the opponent. The output includes the index of each
        strategy and the payoffs for both players at the equilibrium.
    """
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
    """
    Plot the best response polyhedra for each player in a two-player game based on the
    provided payoff matrices. This function visualizes how the payoff for a player
    changes with different strategy mixes of the opponent.

    Parameters:
        game (NashGame): The game object containing the payoff matrices for both players.
        savefig (bool, optional): If True, saves the generated plot images to the
        'outputs' directory. Default is True.

    Notes:
        The function iterates over the two players, computing and plotting each player's
        payoff as a function of the opponent's strategy mix, ranging from 0 to 1. The
        charts show both individual strategy payoffs and the best response polytope,
        which is the envelope of the highest payoffs achievable against all strategies
        of the opponent.
    """
    for player in ["1", "2"]:
        if player == "1":
            payoffs = game.payoffs_A
            opponent = "2"
        else:
            payoffs = game.payoffs_B.T
            opponent = "1"
            if payoffs.shape != game.payoffs_A.shape:
                warnings.warn(
                    "The shapes of payoffs A and B do not match. Cannot compute Player 2's polyhedra"
                )
                break

        y_values = np.linspace(0, 1, 100)
        payoff_computations = [np.dot(payoffs, [y, 1 - y]) for y in y_values]
        payoff_computations = np.array(payoff_computations)

        max_payoff = np.max(payoff_computations) + 1

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

        fig.add_trace(
            go.Scatter(
                x=y_values,
                y=[max_payoff] * len(y_values),
                mode="lines",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
            )
        )
        fig.data[-1].update(fill="tonexty", fillcolor="rgba(144, 238, 144, 0.5)")

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
    """
    Plot the best response polytope for each player in a two-player game. This function
    visualizes the feasible regions for player strategies, showing where their payoffs
    are maximized against the opponent's strategies.

    Parameters:
        game (NashGame): The game object containing the payoff matrices for both players.
        savefig (bool, optional): If True, saves the generated plot images to the
        'outputs' directory. Default is True.

    Notes:
        This function calculates the polytope for each player by creating a set of
        inequalities derived from the payoff matrices. These inequalities define the
        conditions under which one player's strategy is a best response to the strategy
        of the opponent.

        Each inequality is transformed into a line on a plot representing strategy
        combinations. The intersection of these lines forms the polytope—the area within
        which the opponent player's strategy is optimal. The function highlights this
        region on the plot.
    """
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


def plot_probabilities(game):
    """
    Plot the probabilities of best response strategies for each player in a two-player,
    2x2 game.

    Parameters:
        game (NashGame): The game object containing the payoff matrices for both players.

    Notes:
        This function computes and plots the best response strategies for each player as
        a function of the probability that the opponent plays their first strategy. The
        best response is calculated by determining which strategy maximizes the player's
        expected payoff against a mixed strategy of the opponent. The plot shows how the
        probability of choosing each strategy by the player changes as the opponent
        varies their strategy mix between their two strategies.

        If the payoff matrices for the two players do not match in size, the function
        issues a warning and stops execution, as it is specifically designed for 2x2
        games.
    """
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

        strategy_range = np.linspace(0, 1, 100)
        best_responses = []

        for p in strategy_range:
            opponent_strategy = np.array([p, 1 - p])
            response = best_response(payoffs, opponent_strategy)
            best_responses.append(response)

        best_responses = np.array(best_responses)

        plt.figure(figsize=(8, 6))
        plt.fill(strategy_range, best_responses[:, 0], label="Strategy 1", alpha=0.5)
        plt.fill(strategy_range, best_responses[:, 1], label="Strategy 2", alpha=0.5)
        plt.title(f"Best Response Polytope for Player {player}")
        plt.xlabel("Probability of Opponent Playing Strategy 1")
        plt.ylabel("Probability of Best Response")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"outputs/probabilities_{game.name}_player_{player}.png")
        plt.show()


def best_response(payoffs, strategy):
    """
    Compute the best response to a given opponent's strategy based on the payoff matrix.
    The best response is the strategy that maximizes a player's expected payoff, given
    the opponent's mixed strategy.

    Parameters:
        payoffs (numpy.ndarray): The payoff matrix of the player, where each row
                                 represents a different strategy of the player, and each
                                 column represents a different strategy of the opponent.
        strategy (numpy.ndarray): The mixed strategy vector of the opponent, where each
                                  element is the probability of the opponent playing a
                                  corresponding strategy.

    Returns:
        numpy.ndarray: A one-hot encoded vector representing the best response strategy.
                       The strategy that gives the highest expected payoff is marked
                       with a 1, and all other strategies are marked with 0s.

    Notes:
        This function uses matrix multiplication to calculate the expected payoff for
        each of the player's strategies against the given opponent's strategy, and then
        selects the strategy with the highest expected payoff.
    """
    return np.eye(len(payoffs))[np.argmax(payoffs @ strategy)]


def main():

    game = read_game(example="2x2_avis")

    plot_best_response_polyhedra(game)
    plot_best_response_polytope(game)
    equilibria, num_vertices = find_nash_equilibria(game)


if __name__ == "__main__":
    main()
