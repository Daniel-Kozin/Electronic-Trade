import numpy as np

class Recommender:
    def __init__(self, n_weeks: int, n_users: int, prices: np.array, budget: int):
        self.n_rounds = n_weeks
        self.n_users = n_users
        self.item_prices = prices
        self.budget = budget

        eps = 1e-5  # small number to avoid division by zero
        self.num_chosen = np.full(shape=len(self.item_prices), fill_value=eps)
        self.num_wins = np.zeros(len(self.item_prices))

        # Initially select cheapest podcasts that fit budget
        self.podcasts = self.get_initial_podcasts()

        # Store last recommendations to update stats later
        self.recommendations = None

        # Thompson Sampling distributions per user per podcast (optional, can keep or remove)
        self.distribution = {i: [] for i in range(n_users)}
        for person in range(n_users):
            for podcast in range(len(self.item_prices)):
                self.distribution[person].append((1, 1))  # Beta(1,1)

    def recommend(self) -> np.array:
        """
        Recommend one podcast per user from the current selected set self.podcasts.
        Uses Thompson Sampling per user.
        """
        res = np.zeros(self.n_users, dtype=int)
        for user in range(self.n_users):
            samples = []
            for podcast in self.podcasts:
                a, b = self.distribution[user][podcast]
                samples.append(np.random.beta(a, b))
            chosen_idx = np.argmax(samples)
            chosen = self.podcasts[chosen_idx]
            res[user] = chosen

        self.recommendations = res
        return self.recommendations

    def update(self, results: np.array):
        """
        Update the statistics based on user feedback.
        results: array of 0/1 rewards for each user.
        """
        current_round = self.n_rounds

        for i, reward in enumerate(results):
            chosen_podcast = self.recommendations[i]
            reward_val = 1 if reward == 1 else 0

            # Update global stats
            self.num_chosen[chosen_podcast] += 1
            self.num_wins[chosen_podcast] += reward_val

            # Update Beta distribution per user (optional)
            a, b = self.distribution[i][chosen_podcast]
            if reward_val == 1:
                a += 1
            else:
                b += 1
            self.distribution[i][chosen_podcast] = (a, b)

        # Calculate cost-aware UCB scores
        estimated_mean = self.num_wins / self.num_chosen
        log_term = np.log(current_round + 1)  # +1 to avoid log(0)

        confidence_radius = np.sqrt((2 * log_term) / self.num_chosen)

        # Avoid division by zero price
        prices_safe = np.maximum(self.item_prices, 1e-6)

        ucb_scores = (estimated_mean + confidence_radius) / prices_safe

        # Select podcasts greedily by UCB score subject to budget
        self.podcasts = self.get_podcasts(ucb_scores)

    def get_initial_podcasts(self):
        # Select cheapest podcasts that fit within the budget initially
        indexed_prices = [(price, idx) for idx, price in enumerate(self.item_prices)]
        indexed_prices.sort(key=lambda x: x[0])  # ascending by price

        selected_indices = []
        current_cost = 0

        for price, idx in indexed_prices:
            if current_cost + price <= self.budget:
                selected_indices.append(idx)
                current_cost += price
            else:
                break

        return selected_indices

    def get_podcasts(self, scores):
        # Select podcasts greedily by scores within budget
        indexed_scores = [(score, idx) for idx, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[0], reverse=True)  # descending

        selected_indices = []
        current_cost = 0

        for score, idx in indexed_scores:
            price = self.item_prices[idx]
            if current_cost + price <= self.budget:
                selected_indices.append(idx)
                current_cost += price

        return selected_indices
