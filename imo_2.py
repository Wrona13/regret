import random
import tsplib95
import networkx
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer


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

    # if not testing: print(f"Start_a = {result_a}, start_b = {result_b}")

    curr_set = 'a'
    while len(set(result_a)) + len(set(result_b)) < len(matrix):
        if curr_set == 'a':
            result_a = find_new_path_by_regret_cost(result_a, matrix, result_a + result_b)
            curr_set = 'b'

        elif curr_set == 'b':
            result_b = find_new_path_by_regret_cost(result_b, matrix, result_a + result_b)
            curr_set = 'a'

        #if not testing and len(result_a) % 10 == 0:
        #    draw_path(result_a, result_b, instance)

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

        # if not testing and len(result_a) % 10 == 0:
        #     draw_path(result_a, result_b, instance)

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


NUM_OF_ITERATIONS = 1000


def perform_test(instance_filenames, start_with, methods, versions):
    for instance_filename in instance_filenames:
        for start in start_with:
            random_walk_t_threshold = []
            for method in methods:
                for version in versions:

                    inst = tsplib95.load(instance_filename)
                    dist = vertex_to_dist_matrix(inst)
                    results = []

                    time_results=[]

                    for i in range(NUM_OF_ITERATIONS):
                        if i * 100 / NUM_OF_ITERATIONS % 5 == 0: print(f"{int(i * 100 / NUM_OF_ITERATIONS)}, ", end="")

                        ts = timer()

                        res_a, res_b = None, None

                        if version == "random_walk":
                            # print(f"going with {max(random_walk_t_threshold)} threshold")
                            res_a, res_b = operate(dist, instance_filename, method, start, version, random_walk_t_threshold = max(random_walk_t_threshold))
                        else: res_a, res_b = operate(dist, instance_filename, method, start, version)

                        results.append((calc_cost(res_a, dist) + calc_cost(res_b, dist), res_a,
                                        res_b))  # sum of costs of a and b, start_a, start_b

                        time_results.append(timer()-ts)

                    maxi = max(results, key=lambda item: item[0])
                    mini = min(results, key=lambda item: item[0])
                    # print(f"{start}, {method}, {instance_filename}, {version} min={mini[0]}, maxi={maxi[0]}, avg={sum([a[0] for a in results]) / len(results)}, avg_time={(te - ts)/NUM_OF_ITERATIONS}")

                    print(f"{start}, {method}, {instance_filename}, {version} | {int(sum([a[0] for a in results]) / len(results))} ({int(mini[0])} - {int(maxi[0])}) , {round(sum(time_results) / len(time_results),3)} ({round(min(time_results),3)} - {round(max(time_results),3)})")

                    random_walk_t_threshold.append(max(time_results))

                    # save best result to file
                    plt.clf()
                    plt.gca().get_xaxis().set_visible(False)
                    plt.gca().get_yaxis().set_visible(False)
                    draw_points(inst)
                    draw_path(mini[1], mini[2], inst)
                    plt.title(f"Aggregate cost = {mini[0]}")
                    plt.savefig(f"{start}_{method}_{version}_{instance_filename.replace('/', '-')}.png")

                    # print(mini[1], mini[2])


def swap_vertexes(a, b, i, j, single_path=False):
    a_, b_ = np.copy(a), np.copy(b)

    if single_path:
        a_[i], a_[j] = a_[j], a_[i]
        b_[i], b_[j] = b_[j], b_[i]

        if i == 0 or j == 0: a_[-1] = a_[0]
        if i == len(a) - 1 or j == len(a) - 1: a_[0] = a_[-1]
    else:
        a_ = list(map(lambda x: x if x != a[i] else b[j], a_))
        b_ = list(map(lambda x: x if x != b[j] else a[i], b_))

    return a_, b_


def swap_edges(a, i, j):
    a_ = a.copy()
    a_[i:j] = a_[j - 1:i - 1:-1]
    return a_


def calc_delta(a, b, i, j, matrix, mode):
    a_next = i + 1 if i != len(a) - 1 else 2
    b_next = j + 1 if j != len(b) - 1 else 2

    a_before = i - 1 if i != 0 else len(a) - 2
    b_before = j - 1 if j != 0 else len(b) - 2

    if mode == "vertexes":

        if np.array_equiv(a,b) and (a_next == j or b_next == i):
            delta_a_gain = matrix[a[i] - 1, a[a_before] - 1] + matrix[a[i] - 1, a[a_next] - 1]
            delta_b_gain = matrix[b[j] - 1, b[b_next] - 1]

            delta_a_loss = matrix[a[i] - 1, b[b_before] - 1] + matrix[a[i] - 1, b[b_next] - 1]
            delta_b_loss = matrix[b[j] - 1, a[a_next] - 1]
        else:
            delta_a_gain = matrix[a[i] - 1, a[a_before] - 1] + matrix[a[i] - 1, a[a_next] - 1]
            delta_b_gain = matrix[b[j] - 1, b[b_before] - 1] + matrix[b[j] - 1, b[b_next] - 1]

            delta_a_loss = matrix[a[i] - 1, b[b_before] - 1] + matrix[a[i] - 1, b[b_next] - 1]
            delta_b_loss = matrix[b[j] - 1, a[a_before] - 1] + matrix[b[j] - 1, a[a_next] - 1]

    elif mode == "edges":
        delta_a_loss = matrix[a[a_before] - 1, b[b_before] - 1]
        delta_b_loss = matrix[a[i] - 1, b[j] - 1]

        delta_a_gain = matrix[a[a_before] - 1, a[i] - 1]
        delta_b_gain = matrix[b[b_before] - 1, b[j] - 1]

    else:
        raise NotImplementedError()

    return (delta_a_gain + delta_b_gain) - (delta_a_loss + delta_b_loss)


def random_paths(set_size=100):
    l = [i for i in range(1, set_size + 1)]
    random.shuffle(l)

    p1, p2 = l[:set_size // 2], l[set_size // 2:]

    return p1 + [p1[0]], p2 + [p2[0]]


def operate(matrix, instance_filename, method="vertexes", start_with="nn",
            version="greedy", random_walk_t_threshold = None):  # mode: vertexes/edges, start_with: random, nn

    a, b = [None] * 2

    if start_with == "nn":
        # a, b = nearest_neighbor_algorithm(matrix, instance)

        # take best result from nn algo - kroA100, kroB100
        if instance_filename == 'data/kroA100.tsp': a, b = [77, 60, 62, 35, 86, 27, 55, 83, 34, 12, 20, 57, 7, 9, 87, 51, 61, 25, 81, 69, 64, 40, 54, 2, 44, 50, 73, 68, 85, 39, 30, 29, 46, 43, 3, 14, 71, 41, 100, 48, 52, 78, 96, 5, 37, 33, 76, 13, 95, 82, 77], [90, 49, 6, 63, 1, 92, 8, 42, 89, 31, 80, 56, 97, 75, 19, 4, 65, 26, 66, 70, 94, 22, 16, 88, 53, 79, 18, 24, 38, 99, 36, 84, 10, 72, 21, 74, 59, 17, 15, 11, 32, 45, 91, 23, 98, 47, 93, 28, 67, 58, 90]
        elif instance_filename == 'data/kroB100.tsp':a, b = [10, 68, 49, 86, 38, 20, 80, 30, 75, 72, 37, 65, 47, 79, 81, 56, 100, 26, 69, 62, 5, 67, 40, 39, 70, 53, 73, 85, 93, 11, 3, 29, 8, 99, 28, 91, 97, 76, 59, 32, 98, 95, 1, 21, 90, 12, 71, 9, 25, 46, 10], [35, 61, 27, 94, 57, 34, 7, 84, 58, 52, 54, 88, 23, 22, 55, 24, 18, 77, 16, 2, 78, 45, 36, 96, 19, 92, 44, 41, 17, 13, 63, 31, 48, 51, 42, 82, 15, 6, 4, 83, 33, 64, 14, 50, 43, 89, 87, 60, 74, 66, 35]
        else: raise NotImplementedError()

    elif start_with == "random":
        a, b = random_paths()
    else:
        raise NotImplementedError()

    # save_plot(a, b, "in.png")

    # print(f"Starting with {a} and {b}")

    if version == "steepest":

        if method == "vertexes":
            deltas = []  # to stop algorithm when it loops between two solutions

            while True:

                # find best swap
                best_swap, best_delta = None, None
                swap_by = "mix"

                for i in range(len(a)):
                    for j in range(len(b)):

                        delta = calc_delta(a, b, i, j, matrix, mode="vertexes")

                        if best_delta is None or best_delta < delta:
                            best_delta = delta
                            best_swap = [i, j]

                for path in (a, b):
                    for i, j in zip(range(len(path)), range(len(path))):
                        if i == j: continue

                        delta = calc_delta(path, path, i, j, matrix, mode="vertexes")

                        if best_delta is None or best_delta < delta:
                            best_delta = delta
                            best_swap = [i, j]
                            swap_by = path

                deltas.append(best_delta)

                if best_delta <= 0 or (len(set(deltas[-10:])) <= 2 and len(deltas) > 12):
                    # print("No gain, imma head out. Will return a and b.")
                    return a, b

                if swap_by == "mix":
                    a, b = swap_vertexes(a, b, *best_swap)
                elif np.array_equiv(swap_by, a):
                    a = swap_vertexes(a, a, *best_swap, single_path=True)[0]
                elif np.array_equiv(swap_by, b):
                    b = swap_vertexes(b, b, *best_swap, single_path=True)[0]
                else:
                    raise NotImplementedError

        elif method == "edges":

            res = []

            for source_path in (a, b):
                deltas = []  # to stop algorithm when it loops between two solutions

                while True:
                    # find best swap
                    best_swap, best_delta = None, None

                    for i in range(1, len(source_path) - 2):  # TODO
                        for j in range(i + 1, len(source_path)):  # TODO
                            if i == j: continue

                            delta = calc_delta(source_path, source_path, i, j, matrix, mode="edges")

                            if best_delta is None or best_delta < delta:
                                best_delta = delta
                                best_swap = [i, j]

                    # print(f"Found that it would be best to swap EDGES {[_ + 1 for _ in best_swap]} (delta={best_delta})")

                    deltas.append(best_delta)

                    if best_delta <= 0 or (len(set(deltas[-10:])) <= 2 and len(deltas) > 12):
                        res.append(source_path)
                        break

                    source_path = swap_edges(source_path, *best_swap)

            return res[0], res[1]

        else:
            raise NotImplementedError()

    elif version == "greedy":

        if method == "vertexes":

            while True:

                a, b = a[::random.choice([-1, 1])], b[::random.choice([-1, 1])]

                a_, b_ = list(range(len(a))), list(range(len(b)))

                random.shuffle(a_)
                random.shuffle(b_)

                possibs = []

                if random.random() < 0.5:
                    for x, y in zip(a_, b_):
                        possibs.append([x, y, "mix"])
                else:
                    for x in range(len(a)):
                        for y in range(len(a)):
                            if x == y: continue
                            possibs.append([x,y,"a"])

                    for x in range(len(b)):
                        for y in range(len(b)):
                            if x == y: continue
                            possibs.append([x,y,"b"])

                random.shuffle(possibs)

                improved = False

                for x, y, mode in possibs:

                    if mode == "mix": delta = calc_delta(a, b, x, y, matrix, mode="vertexes")
                    elif mode == "a": delta = calc_delta(a, a, x, y, matrix, mode="vertexes")
                    elif mode == "b": delta = calc_delta(b, b, x, y, matrix, mode="vertexes")
                    else: raise NotImplementedError

                    if delta > 0:
                        if mode == "mix": a, b = swap_vertexes(a, b, x, y)
                        elif mode == "a": a = swap_vertexes(a, a, x, y, single_path=True)[0]
                        elif mode == "b": b = swap_vertexes(b, b, x, y, single_path=True)[0]
                        else: raise NotImplementedError

                        improved = True
                        break

                if not improved:
                    return a, b

        elif method == "edges":

            res = []

            for source_path in (a, b):
                deltas = []  # to stop algorithm when it loops between two solutions

                while True:
                    last_delta, curr_delta = None, None

                    source_path = source_path[::random.choice([-1,1])]

                    possibs = []

                    for i in range(1, len(source_path) - 2):  # TODO
                        for j in range(i + 1, len(source_path)):  # TODO
                            if i == j: continue
                            possibs.append([i, j])

                    random.shuffle(possibs)

                    improved = False

                    for x, y in possibs:

                        delta = calc_delta(source_path, source_path, x, y, matrix, mode="edges")

                        if delta > 0:
                            source_path = swap_edges(source_path, x, y)
                            improved = True
                            break

                    if not improved:
                        res.append(source_path)
                        break

            return res

    elif version == "random_walk":

        if random_walk_t_threshold is None: raise NotImplementedError

        ts = timer()

        while True:
            action = random.randint(0,2) # 0-swap edges, 1-swap vertexes

            a, b = a[::random.choice([-1,1])], b[::random.choice([-1,1])]

            if action == 0:

                best_delta, best_swap = None, None

                for source_path in (a, b):

                    for i in range(1, len(source_path) - 2):
                        for j in range(i + 1, len(source_path)):
                            if i == j: continue

                            delta = calc_delta(source_path, source_path, i, j, matrix, mode="edges")

                            if best_delta is None or best_delta > delta:
                                best_delta = delta
                                best_swap = [i, j]

                    if best_swap is not None:
                        if source_path == a: a = swap_edges(a, *best_swap)
                        elif source_path == b: b = swap_edges(b, *best_swap)
                        else: raise AssertionError

            else:  # swap vertexes
                best_swap, best_delta = None, None
                swap_by = "mix"

                for i in range(len(a)):
                    for j in range(len(b)):

                        delta = calc_delta(a, b, i, j, matrix, mode="vertexes")

                        if best_delta is None or best_delta < delta:
                            best_delta = delta
                            best_swap = [i, j]

                for path in (a, b):
                    for i, j in zip(range(len(path)), range(len(path))):
                        if i == j: continue

                        delta = calc_delta(path, path, i, j, matrix, mode="vertexes")

                        if best_delta is None or best_delta < delta:
                            best_delta = delta
                            best_swap = [i, j]
                            swap_by = path

                if swap_by == "mix":
                    a, b = swap_vertexes(a, b, *best_swap)
                elif np.array_equiv(swap_by, a):
                    a = swap_vertexes(a, a, *best_swap, single_path=True)[0]
                elif np.array_equiv(swap_by, b):
                    b = swap_vertexes(b, b, *best_swap, single_path=True)[0]
                else:
                    raise NotImplementedError

            if timer() - ts > random_walk_t_threshold:
                break
        return a, b
    else:
        raise NotImplementedError()


def save_plot(a, b, filename):
    # save result to file
    plt.clf()
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    draw_points(instance)
    draw_path(a, b, instance)
    plt.title(f"Aggregate cost = {calc_cost(a, dist_matrix) + calc_cost(b, dist_matrix)}")
    plt.savefig(filename)


testing = True

if testing:
    instance_filenames = ['data/kroA100.tsp', 'data/kroB100.tsp']
    # perform_test(instance_filenames, methods=['vertexes', 'edges'], start_with=['nn','random'])
    perform_test(instance_filenames=['data/kroA100.tsp', 'data/kroB100.tsp'], methods=['vertexes', 'edges'], start_with=['nn', 'random'], versions=["greedy","steepest","random_walk"])
    # perform_test(instance_filenames=['data/kroA100.tsp', 'data/kroB100.tsp'], methods=['vertexes', 'edges'], start_with=['nn', 'random'],
    #                 versions=["random_walk"])
else:
    # load instance
    instance = tsplib95.load('data/kroA100.tsp')
    dist_matrix = vertex_to_dist_matrix(instance)

    # prepare plot
    # draw_points(instance)

    # NN algorithm
    # a, b = nearest_neighbor_algorithm(dist_matrix, instance)

    # greedy cycle algorithm
    # a, b = greedy_cycle_algorithm(dist_matrix, instance)

    # 2-regret algorithm
    # a, b = greedy_2_regret_algorithm(dist_matrix, instance)

    # steepest algorithm
    # a,b = operate(dist_matrix)
    # a, b = operate(dist_matrix, start_with="random", method="vertexes", mix_paths=False)

    mini = 1000000
    while True:
        a, b = nearest_neighbor_algorithm(dist_matrix,instance)
        cost = calc_cost(a,dist_matrix)+calc_cost(b,dist_matrix)
        if cost<mini:
            print(cost,a,b)
            mini=cost
            save_plot(a, b, "out.png")

    # save_plot(a, b, "out.png")
