import unittest
import numpy as np
from nashypy import read_game, solve_mixed_nash


class test_Nash_Equilibria(unittest.TestCase):
    def test_2x2_game(self):
        # no mixed, 1 pure nash
        game = read_game(example="2x2_onemixed_nopure")
        pure_nash, mixed_nash = solve_mixed_nash(game)
        self.assertEqual(mixed_nash, {})
        self.assertEqual(pure_nash, [((1, 1), (2, 2))])

    def test_2purenash(self):
        # 2 pure nash
        game = read_game(example="2x2_twopurenash")
        pure_nash, mixed_nash = solve_mixed_nash(game)
        expected_values = {"p": 0.5, "q": 0.33}
        mixed_nash_values = np.array([mixed_nash[key] for key in expected_values])
        expected_values = np.array([expected_values[key] for key in expected_values])
        self.assertTrue(
            np.allclose(pure_nash, [((0, 1), (3, 1)), ((1, 0), (2, 3))], atol=1e-2)
        )
        self.assertTrue(np.allclose(mixed_nash_values, expected_values, atol=1e-2))

    def test_staghunt_mixedstrategy(self):
        # stag hunt
        game = read_game(example="2x2_staghunt")
        pure_nash, mixed_nash = solve_mixed_nash(game)
        expected_values = {"p": 0.5, "q": 0.5}
        mixed_nash_values = np.array([mixed_nash[key] for key in expected_values])
        expected_values = np.array([expected_values[key] for key in expected_values])
        self.assertTrue(
            np.allclose(pure_nash, [((0, 0), (4, 4)), ((1, 1), (2, 2))], atol=1e-2)
        )
        self.assertTrue(np.allclose(mixed_nash_values, expected_values, atol=1e-2))

    def test_game_of_chicken_stablenash(self):
        # game of chicken
        game = read_game(example="2x2_chicken")
        pure_nash, mixed_nash = solve_mixed_nash(game)
        expected_values = {"p": 0.75, "q": 0.75}
        mixed_nash_values = np.array([mixed_nash[key] for key in expected_values])
        expected_values = np.array([expected_values[key] for key in expected_values])
        self.assertTrue(
            np.allclose(pure_nash, [((0, 1), (4, 0)), ((1, 0), (0, 4))], atol=1e-2)
        )
        self.assertTrue(np.allclose(mixed_nash_values, expected_values, atol=1e-2))


if __name__ == "__main__":  #
    unittest.main()
