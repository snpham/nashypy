# NashyPy

`nashypy` is a Python package designed for the analysis of Nash equilibria in game theory. It offers tools to read game data, visualize best response polyhedra and polytopes, and find Nash equilibria.

Note: this package is in development and currently can only run 2x2 games, 3x2 games are WIP.

## Development Status

**Note:** `nashypy` is currently under development. At this time, it supports 2x2 bimatrix games only. Support for larger games is in progress and will be available in a future update.

## Installation

To install `nashypy`, you can use pip:

```bash
pip install nashypy
```


## Features

- **Read Game**: Load game data from a file or use a predefined example.
- **Plot Best Response Polyhedra**: Visualize the best response strategies in a two-dimensional plot.
- **Plot Best Response Polytope**: Plot feasible regions based on game strategies.
- **Find Nash Equilibria**: Compute and list all Nash equilibria of a given game.

Reading Game Data
You can load games either by specifying the path to a game file or using one of the included examples:

```bash
import nashypy

# Load a game using an included example
game = nashypy.read_game(example="2x2_staghunt")

# Alternatively, load a game from a custom file path
game = nashypy.read_game(file_path="path/to/game.txt")
```

## Visualizing Game Strategies
Once you have a game object, you can visualize the best responses:

```bash
# Plot best response polyhedra
nashypy.plot_best_response_polyhedra(game, savefig=True)

# Plot best response polytope
nashypy.plot_best_response_polytope(game, savefig=True)
```

## Finding Nash Equilibria
To find and list all Nash equilibria of the loaded game:

```bash
equilibria, num_vertices = find_nash_equilibria(game)
```
