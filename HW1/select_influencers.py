import csv


def hill(graph, haters, costs, max_price=1500, FILE="selected_users.csv"):

    # Ignore cyclic import
    from Praducci_simulation import simulate_influence

    best_nodes = set()
    total_price = 0

    nodes = set(graph.nodes) - set(haters.keys())

    while total_price < max_price:
        best_score_per_gain = 0
        last_cost = 0
        best_node = None
        score = 0
        iterations = 10
        for node in (nodes - best_nodes):
            total = 0
            if node in haters or node in best_nodes:
                continue

            if total_price + costs[node] <= max_price:
                current_score = score

                for x in range(iterations):
                    total += simulate_influence(graph, list(best_nodes | {node}), haters, rounds=6)
                new_score = total / iterations

                delta_score = new_score - current_score
                score_per_cost = delta_score / costs[node]

                if (score_per_cost > best_score_per_gain or (score_per_cost == best_score_per_gain
                        and costs[node] < last_cost)):
                    best_score_per_gain = score_per_cost
                    last_cost = costs[node]
                    best_node = node

        if best_node is None:
            break

        total_price += costs[best_node]
        best_nodes.add(best_node)
        print(f"\nThe group right now is {best_nodes}")
        print(f"The price right now is {total_price} / {max_price}")
        print(f"The score right now is {simulate_influence(graph,list(best_nodes), haters, rounds=6)}\n")

    print(best_nodes)
    print(total_price)

    # Write the nodes into a file
    with open(FILE, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['user_id'])  # Write header
        for user in best_nodes:
            writer.writerow([user])

    return best_nodes


