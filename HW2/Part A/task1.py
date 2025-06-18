import numpy as np
import pandas as pd


def check(rating):
	if rating > 5:
		return 5
	elif rating < 1:
		return 1
	return rating


def find_sol():
	X1 = np.zeros((n_pairs, n_users))
	X2 = np.zeros((n_pairs, n_items))

	for i, row in df.iterrows():
		user_id = row['user id']
		item_id = item_id_to_index[row['item id']]

		X1[i][user_id - 1] = 1
		X2[i][item_id] = 1

	X = np.hstack((X1, X2))

	lam = 1

	I = np.eye(n_users + n_items) * np.sqrt(lam)
	A = X.T @ X + I
	B = X.T @ C

	sol = np.linalg.solve(A, B)

	np.savetxt('sol.csv', sol, delimiter=',')
	print("Vector sol saved to 'sol.csv'")



def solve():
	sol = np.loadtxt('sol.csv', delimiter=',', dtype=float)
	print("Matrix sol read from file:")


	total = 0
	for i, row in df.iterrows():
		user_id = row['user id']
		item_id = item_id_to_index[row['item id']]
		rating = row['rating']

		x = sol[user_id - 1]
		y = sol[n_users + item_id]

		# No need to check if r_pred is inside the bound
		r_pred = r_avg + x + y

		total += (rating - r_pred) ** 2

	pred1 = []

	for i, row in df_test.iterrows():
		user_id = row['user id']
		item_id = item_id_to_index[row['item id']]

		# We make it inside 1 throughout 5
		r_pred = check(r_avg + sol[user_id - 1] + sol[n_users + item_id])

		row = [user_id, index_to_item_id[item_id], r_pred]
		pred1.append(row)

	pred1 = pd.DataFrame(pred1, columns=['user id', 'item id', 'rating'])
	pred1.to_csv('pred1.csv', index=False)

	file_path = 'mse.txt'

	MSE = (1 / n_pairs) * total
	with open(file_path, 'w') as file:
		file.write(str(MSE))

	print(MSE)


if __name__ == "__main__":
	df = pd.read_csv('train.csv')
	df_test = pd.read_csv('test.csv')

	r_avg = np.mean(df['rating'])
	C = ((df['rating'] - r_avg).to_numpy()).reshape(-1, 1)

	user_ids = df['user id'].unique()
	item_ids = df['item id'].unique()

	item_id_to_index = {iid: j for j, iid in enumerate(item_ids)}
	index_to_item_id = {j: iid for j, iid in enumerate(item_ids)}


	n_users = len(user_ids)
	n_items = len(item_id_to_index)
	n_pairs = len(df)

	# switch this flag only if u run find_sol() before
	# Because it takes like a minute to find the solution to the LS problem
	flag = True
	if flag:
		find_sol()

	solve()
