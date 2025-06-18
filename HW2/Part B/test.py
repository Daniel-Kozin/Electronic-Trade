from itertools import combinations

import numpy as np

test_1 = {
    'P': np.array([
        [0.2, 0.6, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0],
        [0.1, 0.8, 0.5, 0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.1],
        [0.0, 0.7, 0.2, 0.3, 0.0, 0.2, 0.1, 0.0, 0.0, 0.1],
        [0.1, 0.8, 0.2, 0.2, 0.2, 0.0, 0.0, 0.1, 0.0, 0.0],
        [0.0, 0.7, 0.1, 0.3, 0.0, 0.1, 0.0, 0.0, 0.4, 0.0],
        [0.0, 0.9, 0.2, 0.2, 0.2, 0.0, 0.1, 0.2, 0.0, 0.3],
        [0.1, 0.5, 0.2, 0.3, 0.0, 0.0, 0.1, 0.0, 0.1, 0.2]]),
    'item_prices': np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]),
    'budget': 20,
    'n_weeks': 1500
}

test_2 = {
    'P': np.array([
        [0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.0, 0.3, 0.1, 0.1, 0.0, 0.1, 0.1],
        [0.1, 0.8, 0.2, 0.0, 0.1, 0.1, 0.0, 0.1, 0.0, 0.1],
        [0.1, 0.1, 0.1, 0.0, 0.1, 0.3, 0.1, 0.0, 0.1, 0.0],
        [0.1, 0.0, 0.1, 0.4, 0.2, 0.1, 0.1, 0.1, 0.0, 0.1],
        [0.2, 0.1, 0.0, 0.1, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1],
        [0.1, 0.0, 0.2, 0.0, 0.1, 0.3, 0.2, 0.2, 0.1, 0.1]]),
    
    'item_prices': np.array([25, 25, 5, 10, 10, 10, 10, 10, 10, 10]),
    'budget': 30,
    'n_weeks': 1500
}

test_3 = {
    'P': np.array([
        [0.2, 0.4, 0.0, 0.0, 0.0, 0.3, 0.0, 0.3, 0.0, 0.0], 
        [0.2, 0.0, 0.5, 0.0, 0.3, 0.0, 0.3, 0.1, 0.0, 0.1],
        [0.2, 0.4, 0.1, 0.3, 0.0, 0.0, 0.1, 0.0, 0.0, 0.4],
        [0.2, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0, 0.9, 0.1, 0.1],
        [0.2, 0.0, 0.3, 0.1, 0.0, 0.2, 0.0, 0.1, 0.1, 0.0],
        [0.2, 0.6, 0.1, 0.4, 0.1, 0.2, 0.5, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.1, 0.1, 0.5, 0.2, 0.0, 0.0, 0.0, 0.1]
    ]),
    
    'item_prices': np.array([30, 15, 15, 10, 10, 15, 10, 10, 5, 10]),
    'budget': 40,
    'n_weeks': 1500
}

test_4 = {
    "P": np.array(
        [
            [0.6, 0.3, 0.7, 0.4, 0.6, 0.9, 0.2, 0.6, 0.7, 0.4],
            [0.3, 0.7, 0.7, 0.2, 0.5, 0.4, 0.1, 0.7, 0.5, 0.1],
            [0.4, 0.0, 0.9, 0.5, 0.8, 0.0, 0.9, 0.2, 0.6, 0.3],
            [0.8, 0.2, 0.4, 0.2, 0.6, 0.4, 0.8, 0.6, 0.1, 0.3],
            [0.8, 0.1, 0.9, 0.8, 0.9, 0.4, 0.1, 0.3, 0.6, 0.7],
            [0.2, 0.0, 0.3, 0.1, 0.7, 0.3, 0.1, 0.5, 0.5, 0.9],
            [0.3, 0.5, 0.1, 0.9, 0.1, 0.9, 0.3, 0.7, 0.6, 0.8],
        ]
    ),
    "item_prices": np.array([30, 25, 25, 10, 25, 10, 5, 20, 20, 20]),
    "budget": 60,
    "n_weeks": 1500,
}

test_5 = {
    "P": np.array(
        [
            [0.0, 0.8, 0.6, 0.8, 0.7, 0.0, 0.7, 0.7, 0.2, 0.0],
            [0.7, 0.2, 0.2, 0.0, 0.4, 0.9, 0.6, 0.9, 0.8, 0.6],
            [0.8, 0.7, 0.1, 0.0, 0.6, 0.6, 0.7, 0.4, 0.2, 0.7],
            [0.5, 0.2, 0.0, 0.2, 0.4, 0.2, 0.0, 0.4, 0.9, 0.6],
            [0.6, 0.8, 0.9, 0.9, 0.2, 0.6, 0.0, 0.3, 0.3, 0.4],
            [0.6, 0.6, 0.3, 0.6, 0.2, 0.5, 0.1, 0.9, 0.8, 0.4],
            [0.5, 0.3, 0.9, 0.6, 0.8, 0.6, 0.0, 0.0, 0.8, 0.8],
        ]
    ),
    "item_prices": np.array([20, 25, 5, 15, 30, 15, 5, 30, 25, 5]),
    "budget": 30,
    "n_weeks": 1500,
}
test_6 = {
    "P": np.array(
        [
            [0.9, 0.7, 0.5, 0.7, 0.8, 0.3, 0.0, 0.0, 0.9, 0.3],
            [0.6, 0.1, 0.2, 0.0, 0.4, 0.0, 0.7, 0.0, 0.0, 0.1],
            [0.1, 0.5, 0.6, 0.4, 0.0, 0.0, 0.2, 0.1, 0.4, 0.9],
            [0.5, 0.6, 0.3, 0.6, 0.7, 0.0, 0.5, 0.7, 0.4, 0.3],
            [0.1, 0.5, 0.5, 0.0, 0.8, 0.5, 0.2, 0.3, 0.3, 0.2],
            [0.9, 0.2, 0.2, 0.3, 0.6, 0.3, 0.8, 0.0, 0.7, 0.6],
            [0.1, 0.7, 0.0, 0.8, 0.8, 0.1, 0.6, 0.9, 0.2, 0.6],
        ]
    ),
    "item_prices": np.array([20, 10, 5, 20, 20, 5, 10, 5, 30, 20]),
    "budget": 40,
    "n_weeks": 1500,
}

test_7 = {
    "P": np.array(
        [
            [0.4, 0.6, 0.8, 0.8, 0.2, 0.2, 0.2, 0.3, 0.7, 0.5],
            [0.7, 0.0, 0.7, 0.3, 0.0, 0.7, 0.3, 0.5, 0.7, 0.3],
            [0.2, 0.8, 0.2, 0.8, 0.1, 0.1, 0.1, 0.5, 0.2, 0.8],
            [0.3, 0.0, 0.3, 0.0, 0.4, 0.3, 0.7, 0.7, 0.6, 0.2],
            [0.0, 0.0, 0.2, 0.5, 0.6, 0.5, 0.5, 0.5, 0.2, 0.5],
            [0.7, 0.1, 0.4, 0.0, 0.0, 0.4, 0.2, 0.3, 0.2, 0.0],
            [0.0, 0.4, 0.5, 0.2, 0.8, 0.4, 0.7, 0.0, 0.4, 0.2],
        ]
    ),
    "item_prices": np.array([20, 5, 20, 25, 25, 5, 15, 10, 5, 10]),
    "budget": 45,
    "n_weeks": 1500,
}
test_8 = {
    "P": np.array(
        [
            [0.9, 0.2, 0.7, 0.7, 0.1, 0.5, 0.6, 0.1, 0.9, 0.1],
            [0.9, 0.0, 0.7, 0.0, 0.8, 0.5, 0.6, 0.9, 0.6, 0.9],
            [0.2, 0.1, 0.8, 0.7, 0.9, 0.6, 0.8, 0.3, 0.3, 0.0],
            [0.7, 0.2, 0.6, 0.1, 0.1, 0.6, 0.5, 0.2, 0.8, 0.9],
            [0.5, 0.9, 0.9, 0.5, 0.0, 0.3, 0.9, 0.5, 0.5, 0.4],
            [0.0, 0.7, 0.4, 0.4, 0.6, 0.3, 0.5, 0.3, 0.2, 0.6],
            [0.7, 0.3, 0.1, 0.9, 0.2, 0.0, 0.7, 0.2, 0.9, 0.6],
        ]
    ),
    "item_prices": np.array([10, 20, 10, 20, 20, 5, 20, 20, 5, 10]),
    "budget": 20,
    "n_weeks": 1500,
}

tests = [test_1, test_2, test_3, test_4, test_5, test_6, test_7, test_8]
required_results = [7250, 2250, 4100, 0, 0, 0, 0, 0]


def all_legal_combinations(item_prices: np.ndarray, budget: int):
    """
    Returns all legal combinations of items where sum(prices) <= budget
    Each combination is represented as a tuple of item indices
    """
    n_items = len(item_prices)
    legal_combinations = []

    # Generate combinations of all possible sizes
    for r in range(1, n_items + 1):
        for combo in combinations(range(n_items), r):
            total_price = sum(item_prices[i] for i in combo)
            if total_price <= budget:
                legal_combinations.append(combo)

    return legal_combinations


def calc_best_option(
    P: np.ndarray, item_prices: np.ndarray, budget: int, n_weeks: int
) -> int:
    """
    Calculate the best option based on the given probability matrix P,
    item prices, budget, and number of weeks.
    """
    legal_combinations = all_legal_combinations(item_prices, budget)
    best_option_value = 0

    for combo in legal_combinations:
        # Get the columns of P corresponding to the current combination
        combo_probs = P[:, list(combo)]

        # print(f"Combo: {combo}, Probs: {combo_probs}")

        # For each player, choose the item with the highest prob for him in the combo
        best_choices = []
        for player in range(combo_probs.shape[0]):
            # best choice is the item in combo with the highest probability for this player
            best_choice = combo[int(np.argmax(combo_probs[player]))]
            best_choices.append(best_choice)
        # print(f"Best choices: {best_choices}")

        # Get the corresponding items and their prices
        best_probs = []
        for i, choice in enumerate(best_choices):
            best_probs.append(P[i, choice])

        # Calculate total expected value over all weeks
        total_value = sum(best_probs) * n_weeks

        # Update best option if this one is better
        if total_value > best_option_value:
            best_option_value = total_value

    return int(best_option_value)

if __name__ == "__main__":
    for i, test in enumerate(tests):
        result = calc_best_option(
            P=test["P"],
            item_prices=test["item_prices"],
            budget=test["budget"],
            n_weeks=test["n_weeks"],
        )
        print(f"Test {i + 1}: Got {result}, Expected {required_results[i]}")