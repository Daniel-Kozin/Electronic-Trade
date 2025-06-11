import numpy as np


class Recommender:
    def __init__(self, n_weeks: int, n_users: int, prices: np.array, budget: int):
        self.n_rounds = n_weeks
        self.n_users = n_users
        self.item_prices = prices
        self.budget = budget

        # Needed for calculating the mean in the UCB part
        eps = 1e-5
        self.num_chosen = np.full(shape=len(self.item_prices), fill_value=eps)
        self.num_wins = np.zeros(len(self.item_prices))

        # Possible podcasts to choose from
        self.podcasts = self.get_initial_podcasts()

        # The podcasts we recommended
        self.recommendations = None

        # Defining basic beta distribution per person per podcast
        self.distribution = {i: [] for i in range(n_users)}
        for person in range(n_users):
            for podcast in range(len(self.item_prices)):
                self.distribution[person].append((1, 1))

    # Thompson Sampling
    def recommend(self) -> np.array:
        res = np.zeros(self.n_users, dtype=int)
        for user in range(self.n_users):
            samples = []

            for i, podcast in enumerate(self.podcasts):
                a, b = self.distribution[user][podcast]
                samples.append(np.random.beta(a, b))

            chosen_idx = np.argmax(samples)
            chosen = self.podcasts[chosen_idx]
            res[user] = chosen

        self.recommendations = res
        return self.recommendations

    def update(self, results: np.array):
        # Updating the distributions
        for i, res in enumerate(results):
            chosen_podcast = self.recommendations[i]
            a, b = self.distribution[i][chosen_podcast]

            # For UCB part
            self.num_chosen[chosen_podcast] += 1

            if res == 1:
                a += 1
                self.num_wins[chosen_podcast] += 1  # For UCB part
            else:
                b += 1

            self.distribution[i][chosen_podcast] = (a, b)

        """for podcast in self.podcasts:
            if podcast not in self.recommendations:
                self.num_chosen[podcast] += 0.5"""

        estimated_mean = self.num_wins / self.num_chosen
        # calculating radius of UCB
        rad = np.sqrt(2 * np.log(self.n_rounds) / (7.5 * self.num_chosen))
        ucb = estimated_mean + rad
        # print(ucb)
        self.podcasts = self.get_podcasts(ucb)

    # TODO: optimize this
    def get_initial_podcasts(self):
        indexed_prices = []
        for original_index, price in enumerate(self.item_prices):
            indexed_prices.append((price, original_index))

        # This ensures we pick the cheapest items first
        indexed_prices.sort(key=lambda x: x[0])

        selected_indices = []
        current_cost = 0

        # Iterate through the sorted items, adding them if within budget
        for price, original_index in indexed_prices:
            if current_cost + price <= self.budget:
                selected_indices.append(original_index)
                current_cost += price
            else:
                break
        # print(f'selected_indices is {selected_indices} and the price is {current_cost}')
        return selected_indices

    def get_podcasts(self, scores):
        best_subset = self.find_best_subset(scores)
        return best_subset

    def find_best_subset(self, scores):
        """
        Find subsets of items where the total price is between self.budget - 5 and self.budget.

        Args:
            scores: List of scores for each item.
            self: An object containing:
                - item_prices: List of prices for each item.
                - budget: The maximum budget.

        Returns:
            List of tuples where each tuple contains:
                - a tuple of indexes representing the subset
                - the sum of scores for that subset
        """
        item_prices = self.item_prices
        budget = self.budget
        n = len(item_prices)
        valid_subsets = []

        # We'll use bitmasking to generate all possible subsets
        for mask in range(1, 1 << n):
            # Get the indexes where the bit is set
            indexes = [i for i in range(n) if (mask & (1 << i))]

            total_price = np.sum(item_prices[indexes])
            total_score = np.sum(np.array(scores)[indexes])

            if (budget - 5) <= total_price <= budget:
                valid_subsets.append((indexes, total_score))
        best_subset = max(valid_subsets, key=lambda x: x[1])

        return best_subset


