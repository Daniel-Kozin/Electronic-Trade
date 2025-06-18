import time
from test import test_1, test_2, test_3, test_4, test_5, test_6, test_7, test_8
from tqdm import tqdm
from Recommender import Recommender
import numpy as np

TOTAL_TIME_LIMIT = 120  # seconds


class Simulation():
    def __init__(self, P: np.array, prices: np.array, budget, n_weeks: int):
        self.P = P.copy()
        self.item_prices = prices
        self.budget = budget
        self.n_weeks = n_weeks

    def _validate_recommendation(self, recommendation):
        if not isinstance(recommendation, np.ndarray):
            print(f'ERROR: {recommendation} is not an np.array')
            return False

        if not np.issubdtype(recommendation.dtype, np.integer):
            print(f'ERROR: type of {recommendation} is not int')
            return False

        if recommendation.shape != (self.P.shape[0],):
            print(f'ERROR: {recommendation} is not 1D array or has wrong length')
            return False

        if ((recommendation < 0) | (recommendation >= self.P.shape[1])).any():
            print(f'ERROR: {recommendation} contains invalid podcasts')
            return False

        podcasts = np.unique(recommendation)
        total_price = np.sum(self.item_prices[podcasts])

        if total_price > self.budget:
            print(f'ERROR: {total_price} is above budget of {self.budget}')
            return False

        return True

    def simulate(self) -> int:
        total_time_taken = 0

        init_start = time.perf_counter()

        try:
            recommender = Recommender(n_weeks=self.n_weeks, n_users=self.P.shape[0],
                                      prices=self.item_prices.copy(),
                                      budget=self.budget)
        except Exception as e:
            print('Recommender __init__ caused error')
            raise e

        init_end = time.perf_counter()

        total_time_taken += init_end - init_start

        reward = 0

        for round_idx in range(self.n_weeks):
            try:
                recommendation_start = time.perf_counter()
                recommendation = recommender.recommend()
                recommendation_end = time.perf_counter()
            except Exception as e:
                print(f'Recommmender.recommend() raised error at round {round_idx}')
                raise e

            if recommendation is None:
                print('No recommendation supplied.')
                return 0

            recommendation_time = recommendation_end - recommendation_start

            if not self._validate_recommendation(recommendation):
                print(f'Error: Invalid recommendation at round {round_idx}')
                return 0

            results = np.random.binomial(n=1, p=self.P[np.arange(self.P.shape[0]), recommendation])
            current_reward = np.sum(results)
            next_reward = reward + current_reward

            try:
                update_start_time = time.perf_counter()
                recommender.update(results)
                update_end_time = time.perf_counter()
            except Exception as e:
                print(f'Recommmender.update() raised error at round {round_idx}')
                raise e

            update_time = update_end_time - update_start_time

            time_for_current_round = recommendation_time + update_time

            if total_time_taken + time_for_current_round > TOTAL_TIME_LIMIT:
                print(f'TOTAL TIME LIMIT EXCEEDED. Returning reward at after {round_idx} rounds')
                return reward
            else:
                total_time_taken += time_for_current_round
                reward = next_reward

        #print(f'Total time taken: {total_time_taken} seconds')
        return reward

def alpha_beta_sim():
    results = []
    alphas = [i for i in range(1, 11)]
    betas = [i for i in range(1, 11)]
    iterations = 100

    test_cases = [test_1, test_2, test_3, test_4, test_5, test_6, test_7, test_8]

    for alpha, beta in tqdm([(a, b) for a in alphas for b in betas], desc="Testing alpha/beta pairs"):
        totals = [0 for _ in range(len(test_cases))]

        for _ in range(iterations):
            for idx, test in enumerate(test_cases):
                sim = Simulation(test['P'], test['item_prices'], test['budget'], test['n_weeks'])
                totals[idx] += sim.simulate()

        averages = [total / iterations for total in totals]
        overall_avg = sum(averages) / len(averages)
        results.append(((alpha, beta), *averages, overall_avg))

    top_10 = sorted(results, key=lambda x: x[-1], reverse=True)[:10]

    print('\nTop 10 (alpha, beta) configurations by average reward:')
    for rank, result in enumerate(top_10, 1):
        alpha, beta = result[0]
        rewards = result[1:-1]
        print(f'{rank}. Alpha = {alpha}, Beta = {beta} → ' +
              ', '.join([f'Reward {i + 1} = {r:.2f}' for i, r in enumerate(rewards)]))


if __name__ == '__main__':
    iterations = 300
    #test_cases = [test_1, test_2, test_3, test_4, test_5, test_6, test_7, test_8]
    test_cases = [test_1, test_2, test_3]

    totals = [0 for _ in test_cases]

    for _ in range(iterations):
        for idx, test in enumerate(test_cases):
            sim = Simulation(test['P'], test['item_prices'], test['budget'], test['n_weeks'])
            totals[idx] += sim.simulate()

    for i, total in enumerate(totals, 1):
        print(f'Reward {i} = {total / iterations}')

"""
Test 1: Got 7500, Expected 7250
Test 2: Got 2700, Expected 2250
Test 3: Got 4800, Expected 4100
Test 4: Got 9000, Expected 0
Test 5: Got 7800, Expected 0
Test 6: Got 7950, Expected 0
Test 7: Got 7200, Expected 0
Test 8: Got 8100, Expected 0


Reward 1 = 7445.8
Reward 2 = 2397.49
Reward 3 = 4642.68

Reward 1 = 7431.36
Reward 2 = 2421.7
Reward 3 = 4674.84

"""
