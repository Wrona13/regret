import random
import tsplib95
import networkx
import numpy as np
from matplotlib import pyplot as plt


def draw_points(instance):
    x_cords = [i[0] for i in instance.node_coords.values()]
    y_cords = [i[1] for i in instance.node_coords.values()]
    labels = list(instance.node_coords.keys())

    # draw original points
    for x, y, label in zip(x_cords, y_cords, labels):
        plt.scatter(x, y, marker='o', color='green')
        plt.text(x + 0.3, y + 0.3, label, fontsize=7)


def draw_path(a, b, instance):
    # draw result
    res_cords_x_a, res_cords_y_a = [instance.node_coords[i][0] for i in a], [instance.node_coords[i][1] for i in a]
    res_cords_x_b, res_cords_y_b = [instance.node_coords[i][0] for i in b], [instance.node_coords[i][1] for i in b]

    # print(res_cords)

    plt.plot(res_cords_x_a, res_cords_y_a, '-o')
    plt.plot(res_cords_x_b, res_cords_y_b, '-o')

    plt.pause(0.05)


def vertex_to_dist_matrix(problem):
    return networkx.to_numpy_matrix(problem.get_graph())


def calc_cost(path, matrix):
    return sum([matrix[path[i] - 1, path[i + 1] - 1] for i in range(len(path) - 1)])


def find_nearest_neighbor(vertex, matrix, visited_vertexes):
    distances = np.copy(matrix[vertex - 1, :])
    distances[:, [a - 1 for a in visited_vertexes]] = float('inf')
    result = np.argmin(distances) + 1

    return result


def find_new_path_by_neighbor(path, matrix, visited_vertexes):
    min_cost = float('inf')
    res_path = []

    for i in range(len(path)):
        neighbor = find_nearest_neighbor(path[i], matrix, visited_vertexes)

        # insert on left side
        new_path_left = path[:i] + [neighbor] + path[i:]

        cost_left = calc_cost(new_path_left, matrix)

        if cost_left < min_cost:
            res_path = new_path_left
            min_cost = cost_left

        # insert on right side
        new_path_right = path[:i + 1] + [neighbor] + path[i + 1:]

        cost_right = calc_cost(new_path_right, matrix)

        if cost_right < min_cost:
            res_path = new_path_right
            min_cost = cost_right

    return res_path


def find_new_path_by_cycle_cost(path, matrix, visited_vertexes):
    min_cost = float('inf')
    res_path = []

    for i in range(0, len(path), 2):
        for neighbor in set(range(1, len(matrix) + 1)) - set(visited_vertexes):
            # print(neighbor)
            # insert on left side
            if i == 0:
                new_path_left = path[:i] + [neighbor] + path[i:]
                new_path_left[-1] = neighbor
            else:
                new_path_left = path[:i] + [neighbor] + path[i:]

            cost_left = calc_cost(new_path_left, matrix)

            if cost_left < min_cost:
                res_path = new_path_left
                min_cost = cost_left

            # insert on right side
            if i == len(path) - 1:
                new_path_right = path[:i + 1] + [neighbor] + path[i + 1:]
                new_path_right[0] = neighbor
            else:
                new_path_right = path[:i + 1] + [neighbor] + path[i + 1:]

            cost_right = calc_cost(new_path_right, matrix)

            if cost_right < min_cost:
                res_path = new_path_right
                min_cost = cost_right

    return res_path


def find_new_path_by_regret_cost(path, matrix, visited_vertexes):
    min_cost = float('inf')
    res_path = []
    old_regret, regret, regret_min = 0, 0, 0
    cost = []
    for neighbor in set(range(1, len(matrix) + 1)) - set(visited_vertexes):
        dict = {}
        for i in range(0, len(path)):

            if i == 0:
                new_path_left = path[:i] + [neighbor] + path[i:]
                new_path_left[-1] = neighbor
            else:
                new_path_left = path[:i] + [neighbor] + path[i:]
            cost_left = calc_cost(new_path_left, matrix)
            if cost_left < min_cost:
                new_path = new_path_left
                min_cost = cost_left
            dict[cost_left] = new_path_left
        cost.append(dict)
    for i in range(len(path)):
        neighbor = find_nearest_neighbor(path[i], matrix, visited_vertexes)

        new_path_left = path[:i] + [neighbor] + path[i:]
        cost_left = calc_cost(new_path_left, matrix)

        if cost_left < min_cost:
            res_path = new_path_left
            min_cost = cost_left

        new_path_right = path[:i + 1] + [neighbor] + path[i + 1:]
        cost_right = calc_cost(new_path_right, matrix)

        if cost_right < min_cost:
            res_path = new_path_right
            min_cost = cost_right
    for c in cost:
        if len(c) > 0: regret_min = sorted(c.keys())[0]
        if len(c) > 1:
            regret = sorted(c.keys())[1] - sorted(c.keys())[0]
            if regret > old_regret:
                old_regret = regret
                res_path = c[sorted(c.keys())[0]]
    if regret_min * 0.7 > min_cost or (regret_min - 2 * regret) > min_cost:
        res_path = new_path
    return res_path


def greedy_2_regret_algorithm(matrix, instance):
    # select starting vertexes
    result_a, result_b = [[a] for a in random.sample(range(1, len(matrix) + 1), 2)]
    # add second and third vertex to create a cycle
    result_a += [find_nearest_neighbor(result_a[0], matrix, result_a + result_b)]
    result_a.append(result_a[0])

    result_b += [find_nearest_neighbor(result_b[0], matrix, result_a + result_b)]
    result_b.append(result_b[0])

    if not testing: print(f"Start_a = {result_a}, start_b = {result_b}")

    curr_set = 'a'
    while len(set(result_a)) + len(set(result_b)) < len(matrix):
        if curr_set == 'a':
            result_a = find_new_path_by_regret_cost(result_a, matrix, result_a + result_b)
            curr_set = 'b'

        elif curr_set == 'b':
            result_b = find_new_path_by_regret_cost(result_b, matrix, result_a + result_b)
            curr_set = 'a'

        if not testing and len(result_a) % 10 == 0:
           draw_path(result_a, result_b,instance)

    return result_a, result_b


def nearest_neighbor_algorithm(matrix, instance):
    result_a, result_b = [[a] for a in random.sample(range(1, len(matrix) + 1), 2)]

    # print(f"Start_a = {result_a[0]}, start_b = {result_b[0]}")

    curr_set = 'a'
    while len(result_a) + len(result_b) < len(matrix):
        if curr_set == 'a':
            result_a = find_new_path_by_neighbor(result_a, matrix, result_a + result_b)
            curr_set = 'b'

        elif curr_set == 'b':
            result_b = find_new_path_by_neighbor(result_b, matrix, result_a + result_b)
            curr_set = 'a'

        if not testing and len(result_a) % 10 == 0:
            draw_path(result_a, result_b, instance)

    result_a.append(result_a[0])
    result_b.append(result_b[0])

    return result_a, result_b


def greedy_cycle_algorithm(matrix, instance):
    # select starting vertexes
    result_a, result_b = [[a] for a in random.sample(range(1, len(matrix) + 1), 2)]

    # add second and third vertex to create a cycle
    result_a += [find_nearest_neighbor(result_a[0], matrix, result_a + result_b)]
    result_a.append(result_a[0])

    result_b += [find_nearest_neighbor(result_b[0], matrix, result_a + result_b)]
    result_b.append(result_b[0])

    if not testing: print(f"Start_a = {result_a}, start_b = {result_b}")

    curr_set = 'a'
    while len(set(result_a)) + len(set(result_b)) < len(matrix):
        if curr_set == 'a':
            result_a = find_new_path_by_cycle_cost(result_a, matrix, result_a + result_b)
            curr_set = 'b'

        elif curr_set == 'b':
            result_b = find_new_path_by_cycle_cost(result_b, matrix, result_a + result_b)
            curr_set = 'a'

        if not testing and len(result_a) % 10 == 0:
            draw_path(result_a, result_b, instance)

    return result_a, result_b


NUM_OF_ITERATIONS = 3


def perform_test(functions, instance_filenames):
    for function in functions:
        for instance_filename in instance_filenames:

            inst = tsplib95.load(instance_filename)
            dist = vertex_to_dist_matrix(inst)
            results = []

            for i in range(NUM_OF_ITERATIONS):
                if i * 100 / NUM_OF_ITERATIONS % 5 == 0: print(f"{int(i * 100 / NUM_OF_ITERATIONS)}, ", end="")

                res_a, res_b = function(dist, inst)
                results.append((calc_cost(res_a, dist) + calc_cost(res_b, dist), res_a,
                                res_b))  # sum of costs of a and b, start_a, start_b

            maxi = max(results, key=lambda item: item[0])
            mini = min(results, key=lambda item: item[0])
            print(
                f"{function.__name__}, {instance_filename}, min={mini[0]}, maxi={maxi[0]}, avg={sum([a[0] for a in results]) / len(results)}")

            # save best result to file
            plt.clf()
            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
            draw_points(inst)
            draw_path(mini[1], mini[2], inst)
            plt.title(f"Aggregate cost = {mini[0]}")
            plt.savefig(f"{function.__name__}_{instance_filename.replace('/', '-')}.png")


testing = False

if testing:
    functions = [nearest_neighbor_algorithm, greedy_2_regret_algorithm, greedy_cycle_algorithm]
    instance_filenames = ['data/kroA100.tsp','data/kroB100.tsp']
    perform_test(functions, instance_filenames)
else:
    # load instance
    instance = tsplib95.load('data/kroA100.tsp')
    dist_matrix = vertex_to_dist_matrix(instance)

    # prepare plot
    draw_points(instance)

    # NN algorithm
    a, b = nearest_neighbor_algorithm(dist_matrix, instance)

    # greedy cycle algorithm
    # a, b = greedy_cycle_algorithm(dist_matrix, instance)

    # 2-regret algorithm
    # a, b = greedy_2_regret_algorithm(dist_matrix, instance)

    # draw final plot
    plt.clf()
    draw_points(instance)
    draw_path(a, b, instance)
    plt.title(
        f"Start with {a[0]}=>{calc_cost(a, dist_matrix)}, with {b[0]}=>{calc_cost(b, dist_matrix)}")
    plt.show()
