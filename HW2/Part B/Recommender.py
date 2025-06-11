import numpy as np

class Recommender:
    def __init__(self, n_weeks: int, n_users: int, prices: np.array, budget: int, al, be):
        self.n_rounds = n_weeks
        self.n_users = n_users
        self.item_prices = prices
        self.budget = budget

        self.al = al
        self.be = be
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

        # Used in UCB
        self.C = 7.3

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
            # 7, 7 - 7453, 2374, 4598
            # 7, 10 - 7452 2409 4578
            # 5, 7 - 7446 2402 4600
            # 3, 5 - 7432 2395 4660
            # 5, 3 - 7446 2402 4611
            # 4, 6 - 7442 2401 4660
            # 5, 8 - 7445 2391 4663
            if res == 1:
                a += self.al
                self.num_wins[chosen_podcast] += 1  # For UCB part
            else:
                b += self.be

            self.distribution[i][chosen_podcast] = (a, b)

        """for podcast in self.podcasts:
            if podcast not in self.recommendations:
                self.num_chosen[podcast] += 0.5"""

        estimated_mean = self.num_wins / self.num_chosen
        # calculating radius of UCB
        rad = np.sqrt(2 * np.log(self.n_rounds) / (self.C * self.num_chosen))
        ucb = estimated_mean + rad
        #print(ucb)
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
        #print(f'selected_indices is {selected_indices} and the price is {current_cost}')
        return selected_indices

    def get_podcasts(self, scores):
        indexed_scores = []
        for original_index, score in enumerate(scores):
            indexed_scores.append((score, original_index))

        indexed_scores.sort(key=lambda x: x[0], reverse=True)

        selected_indices = []
        current_cost = 0

        for score, original_index in indexed_scores:
            if current_cost + self.item_prices[original_index] <= self.budget:
                selected_indices.append(original_index)
                current_cost += self.item_prices[original_index]

        #print(f'selected_indices is {selected_indices} and the price is {current_cost}')
        return selected_indices

