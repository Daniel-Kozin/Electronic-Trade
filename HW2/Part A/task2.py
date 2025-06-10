import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from task1 import check


def get_X_and_approx_X():
	X = np.zeros((n_users, n_items))

	for i, row in df.iterrows():
		user_id = row['user id']
		item_id = item_id_to_index[row['item id']]
		rating = row['rating']

		X[user_id - 1, item_id - 1] = rating

	u, s, v_t = svds(X, k=10)

	X_approx = u @ np.diag(s) @ v_t

	return X, X_approx


def solve(X_approx):
	total = 0
	for i, row in df.iterrows():
		user_id = row['user id']
		item_id = item_id_to_index[row['item id']]
		rating = row['rating']

		# No need to check if r_pred is inside the bound
		r_pred = X_approx[user_id - 1, item_id - 1]

		total += (rating - r_pred) ** 2

	pred2 = []
	for i, row in df_test.iterrows():
		user_id = row['user id']
		item_id = item_id_to_index[row['item id']]

		# We make it inside 1 throughout 5
		r_pred = check(X_approx[user_id - 1, item_id - 1])

		row = [user_id, index_to_item_id[item_id], r_pred]
		pred2.append(row)

	pred2 = pd.DataFrame(pred2, columns=['user id', 'item id', 'rating'])
	pred2.to_csv('pred2.csv', index=False)

	MSE = (1 / n_pairs) * total
	file_path = 'mse.txt'
	with open(file_path, 'a') as file:
		file.write('\n')
		file.write(str(MSE))

	print(MSE)


if __name__ == "__main__":
	df = pd.read_csv('train.csv')
	df_test = pd.read_csv('test.csv')

	r_avg = np.mean(df['rating'])
	C = ((df['rating'] - r_avg).to_numpy()).reshape(-1, 1)

	user_ids = df['user id'].unique()
	item_ids = df['item id'].unique()
	item_ids_sorted = sorted(item_ids)

	item_id_to_index = {iid: j for j, iid in enumerate(item_ids)}
	index_to_item_id = {j: iid for j, iid in enumerate(item_ids)}

	n_users = len(user_ids)
	n_items = len(item_id_to_index)
	n_pairs = len(df)

	X, approx_X = get_X_and_approx_X()
	solve(approx_X)
