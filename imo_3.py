import random
import tsplib95
import networkx
import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer

"""Plotting graphs"""


def draw_points(instance):
    x_cords = [i[0] for i in instance.node_coords.values()]
    y_cords = [i[1] for i in instance.node_coords.values()]
    labels = list(instance.node_coords.keys())

    # draw original points
    for x, y, label in zip(x_cords, y_cords, labels):
        plt.scatter(x, y, marker='o', color='green')
        # plt.text(x + 0.3, y + 0.3, label, fontsize=7)


def draw_path(a, b, instance):
    # draw result
    res_cords_x_a, res_cords_y_a = [instance.node_coords[i][0] for i in a], [instance.node_coords[i][1] for i in a]
    res_cords_x_b, res_cords_y_b = [instance.node_coords[i][0] for i in b], [instance.node_coords[i][1] for i in b]

    # print(res_cords)

    plt.plot(res_cords_x_a, res_cords_y_a, '-o')
    plt.plot(res_cords_x_b, res_cords_y_b, '-o')


"""Calculate cost"""


def vertex_to_dist_matrix(problem):
    return networkx.to_numpy_matrix(problem.get_graph())


def calc_cost(path, matrix):
    return np.sum([matrix[path[i] - 1, path[i + 1] - 1] for i in range(len(path) - 1)])


"""Nearest neighbour"""


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


def nearest_neighbor_algorithm(matrix):
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


"""Additionl functions for steepest algotithm"""


def swap_edges(a, i, j):
    a_ = a.copy()
    a_[i:j] = a_[j - 1:i - 1:-1]

    if a_[0] != a[-1]:
        print(a)
        print(i, j)
        print(a_)
        raise AssertionError

    return a_


def swap_vertexes(a, b, i, j, single_path=False):


    if single_path:
        a_, b_ = np.copy(a), np.copy(b)

        a_[i], a_[j] = a_[j], a_[i]
        b_[i], b_[j] = b_[j], b_[i]

        if i == 0 or j == 0: a_[-1] = a_[0]
        if i == len(a) - 1 or j == len(a) - 1: a_[0] = a_[-1]
    else:
        old_a, old_b = a[i], b[j]

        a_ = [k if k!=old_a else old_b for k in a ]
        b_ = [k if k!=old_b else old_a for k in b ]

        # a_ = list(map(lambda x: x if x != a[i] else b[j], a_))
        # b_ = list(map(lambda x: x if x != b[j] else a[i], b_))

    return a_, b_


def calc_delta(a, b, i, j, matrix, mode):
    neigh_num = 1

    a_, b_ = a + a[1:], b + b[1:]
    i_, j_ = i + len(a) - 1, j + len(b) - 1

    if a[i] != a_[i_]:
        print(a[i])
        print(a_[i_])
        print(a)
        print(a_)
        print(i)
        print(i_)
        raise AssertionError

    if mode == "vertexes":
        frag_a, frag_b = a_[i_ - neigh_num:i_ + neigh_num + 1], b_[j_ - neigh_num:j_ + neigh_num + 1]

        cost_before = calc_cost(frag_a, matrix) + calc_cost(frag_b, matrix)

        frag_a, frag_b = swap_vertexes(frag_a, frag_b, len(frag_a)//2, len(frag_b)//2)

        cost_after = calc_cost(frag_a, matrix) + calc_cost( frag_b, matrix)

        return cost_before - cost_after

    elif mode == "edges":

        removed = matrix[a_[i_] - 1, a_[i_-1] - 1] + matrix[a_[j_] - 1, a_[j_-1] - 1]
        added = matrix[a_[i_-1] - 1, a_[j_-1] - 1] + matrix[a_[i_] - 1, b_[j_] - 1]

        return 2*(removed - added)

    else:
        raise AttributeError

    """

    a_next = i + 1 if i != len(a) - 1 else 2
    b_next = j + 1 if j != len(b) - 1 else 2

    a_before = i - 1 if i != 0 else len(a) - 2
    b_before = j - 1 if j != 0 else len(b) - 2

    if mode == "vertexes":

        if np.array_equiv(a, b) and (a_next == j or b_next == i):
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
    """


def random_paths(set_size=200):
    l = [i for i in range(1, set_size + 1)]
    random.shuffle(l)

    p1, p2 = l[:set_size // 2], l[set_size // 2:]

    return p1 + [p1[0]], p2 + [p2[0]]


"""Steepest algorithm"""


def steepest_edges(matrix, instance_filename):  # mode: vertexes/edges, start_with: random, nn

    # random starting paths
    a, b = random_paths()
    res = []

    for source_path in (a, b):
        deltas = []  # to stop algorithm when it loops between two solutions
        while True:
            # find best swap
            best_swap, best_delta = None, None
            for i in range(1, len(source_path) - 2):
                # if random.randint(0,4)==0: continue
                for j in range(i + 1, len(source_path)):
                    # if random.randint(0,4)==0: continue
                    if i == j: continue
                    delta = calc_delta(source_path, source_path, i, j, matrix, mode="edges")
                    if best_delta is None or best_delta < delta:
                        best_delta = delta
                        best_swap = [i, j]
            deltas.append(best_delta)

            # print(calc_cost(source_path,matrix), (len(set(deltas[-30:]))))
            if best_delta <= 0 or (len(set(deltas[-30:])) <= 3 and len(deltas) > 120):
                res.append(source_path)
                break
            # draw_path(source_path,[],tsplib95.load(instance_filename))
            source_path = swap_edges(source_path, *best_swap)

    return res[0], res[1]


def get_adjacent_vertexes(path,vert):

    path_ = path + path[1:]

    idx = path_.index(vert, len(path)-1)

    return [path_[idx - 1], path_[idx + 1]]



def steepest_edges_with_list(matrix, candidates_mode=False, n_neighbors=10):
    LM = []  # list of moves
    a, b = random_paths()  # starting paths

    performed_moves = []

    last_changes = None

    candidates_dist = candidates_dist_dict(matrix)

    init = True


    while True:

        # scanning NEW moves
        new_moves = get_new_available_moves(a, b, LM, matrix, last_changes, init, candidates_mode, n_neighbors, candidates_dist)

        init = False

        moves_after_changes = new_moves.copy()

        found_move = False

        for i, mov in enumerate(new_moves):
            # print(i)

            applicable = is_applicable(mov, a, b)

            if applicable is False:  # Usuwane krawędzie nie występują już w bieżącym rozwiązaniu
                moves_after_changes[i] = None

            elif applicable == "reverse":  # Usuwane krawędzie występują w bieżącym rozwiązaniu w różnym od zapamiętanego kierunku – może być aplikowalny w przyszłości
                continue

            elif applicable is True:  # Usuwane krawędzie występują w bieżącym rozwiązaniu w tym samym kierunku (także obie odrócone)

                affected_vertexes = get_adjacent_vertexes(a, mov.ver_a) if mov.ver_a in a else get_adjacent_vertexes(b, mov.ver_a)
                affected_vertexes += get_adjacent_vertexes(b, mov.ver_b) if mov.ver_b in b else get_adjacent_vertexes(a, mov.ver_b)

                last_changes = [mov, affected_vertexes]

                # if [a, b] == perform_move(mov, a, b):
                #    raise AssertionError

                a, b = perform_move(mov, a, b)


                performed_moves.append((mov.ver_a, mov.ver_b,mov.mode))

                # print(f"curr_cost={calc_cost(a, matrix) + calc_cost(b, matrix)}, avail_options size = {len(new_moves)}, did move {vars(mov)}")

                # draw_path(a, b, instance)

                found_move = True

                """
                if a[0] != a[-1] or b[0] != b[-1]:
                    print(a)
                    print(b)
                    raise AssertionError
                """



                moves_after_changes[i] = None

                if i < len(moves_after_changes) - 1:
                    if {moves_after_changes[i + 1].ver_a, moves_after_changes[i + 1].ver_b} == {mov.ver_a,
                                                                                                mov.ver_b} and \
                            moves_after_changes[i + 1].mode == mov.mode == 'vertexes':
                        moves_after_changes[i + 1] = None

                LM = [a for a in moves_after_changes if a is not None]

                break

            else:
                return NotImplementedError

        if not found_move or (len(set(performed_moves[-30:])) <= 3 and len(performed_moves) > 120):
            return a, b


"""Find new moves"""


class Move:
    def __init__(self, ver_a, ver_b, mode, delta, path_name, edges_to_swap):
        self.ver_a = ver_a
        self.ver_b = ver_b
        self.edges_to_swap = edges_to_swap
        self.mode = mode
        self.delta = delta
        self.path_name = path_name


def perform_move(move, a, b):
    if move.mode == "vertexes":

        return swap_vertexes(a, b, a.index(move.ver_a), b.index(move.ver_b)) if (move.ver_a in a) else swap_vertexes(a, b, a.index(move.ver_b), b.index(move.ver_a))

    elif move.mode == "edges":

        if move.path_name == 'a':
            return swap_edges(a, a.index(move.ver_a), a.index(move.ver_b)), b
        else:
            return a, swap_edges(b, b.index(move.ver_a), b.index(move.ver_b))

    else:
        raise NotImplementedError


def move_delta_update_or_append(move, lst):
    """
    for i, a in enumerate(lst):
        if move.mode == a.mode == 'vertexes' and {move.ver_a, move.ver_b} == {a.ver_a, a.ver_b}:
            lst[i].delta = move.delta
            return

        elif move.mode == a.mode == 'edges' and set(move.edges_to_swap) == set(a.edges_to_swap):
            lst[i].delta = move.delta
            return
    """
    lst.append(move)


def candidates_dist_dict(matrix):

    candidates = {}

    for i in range(1, len(matrix)+1):

        costs = []

        for j in range(1, len(matrix)+1):

            if i != j: costs.append([j, matrix[i-1, j-1]])

        costs.sort(key=lambda x: x[1])

        candidates[i] = [c[0] for c in costs]

    return candidates

def find_candidates(path, candidates_distances, n=10):

    candidates = {}

    for v in path:
        candidates[v] = candidates_distances[v][0:n]

    return candidates.items()



DELTA_ADD_THRESHOLD = 1


def get_new_available_moves(a, b, moves, matrix, last_changes, init=False, candidates_mode=False, n_neighbors=10,candidates_dist = None):

    if not candidates_mode:

        avail_moves = moves.copy()

        a_idxs, b_idxs = [], []

        if not init:
            for v in [last_changes[0].ver_a, last_changes[0].ver_b, *last_changes[1]]:
                if v in a:
                    a_idxs.append(a.index(v))
                elif v in b:
                    b_idxs.append(b.index(v))
                else:
                    raise AssertionError

        # swapping-edges moves
        if init:
            for path, path_name in zip([a, b], ['a', 'b']):
                for i in range(1, len(path) - 2):
                    for j in range(i + 1, len(path) - 1):
                        move = Move(path[i], path[j], 'edges', calc_delta(path, path, i, j, matrix, mode="edges"),
                                    path_name, path[i:j + 1])
                        if move.delta > DELTA_ADD_THRESHOLD: avail_moves.append(move)
        elif last_changes[0].mode == "edges":
            for path, path_name, idxs in zip([a, b], ['a', 'b'], [a_idxs, b_idxs]):
                for idx in idxs:
                    if idx == 0 or idx >= len(path) - 1: continue

                    possibs = []

                    for j in range(idx + 1, len(path) - 1): possibs.append([idx, j])
                    for j in range(1, idx - 1): possibs.append([j, idx])

                    for i, j in possibs:
                        if i == j: continue
                        move = Move(path[i], path[j], 'edges', calc_delta(path, path, i, j, matrix, mode="edges"),
                                    path_name, edges_to_swap=path[i:j + 1])

                        if move.delta > DELTA_ADD_THRESHOLD:   # if delta>0
                            move_delta_update_or_append(move, avail_moves)
        """
        if init:
            # swapping-vertexes moves
    
            for i in range(len(a)):
                for j in range(len(b)):
                    if i==j: continue
                    move = Move(a[i], b[j], 'vertexes', calc_delta(a, b, i, j, matrix, mode="vertexes"), None, None)
                    if move.delta > DELTA_ADD_THRESHOLD: avail_moves.append(move)
        
        elif last_changes[0].mode == "vertexes":
            for idxs, own_set, opposite_set in zip([a_idxs, b_idxs], [a,b], [b, a]):
                for i in idxs:
                    for j in range(len(opposite_set)):
                        if opposite_set[j] in [last_changes[0].ver_a, last_changes[0].ver_b]: continue
                        move = Move(own_set[i], opposite_set[j], 'vertexes', calc_delta(own_set, opposite_set, i, j, matrix, mode="vertexes"), None, None)
                        if move.delta > DELTA_ADD_THRESHOLD:
                            move_delta_update_or_append(move, avail_moves)
        """
    else:

        if not init and random.randint(0,3)!=0: return moves # TODO

        avail_moves = []

        for path, opposite_path, path_name in zip([a,b], [b,a], ['a','b']):  # dla każdej ścieżki
            for v, candidates in find_candidates(path, candidates_dist, n_neighbors):  # iteruj po wierzchołkach ścieżki
                for candidate in candidates:                        # i najbliższym temu wierzchołkowi kandydatach

                    if candidate in path:       # jesli wierzchołek i kandydat są w tej samej ścieżce
                        i, j = path.index(v), path.index(candidate)
                        # wymień krawędzie

                        if v == candidate:
                            print(v)
                            print(candidates)
                            raise AssertionError

                        if i == 0 or j == 0 or i >= len(path) - 1 or j >= len(path): continue

                        move = Move(path[i], path[j], 'edges', calc_delta(path, path, i, j, matrix, mode="edges"),
                                    path_name, edges_to_swap=path[i:j + 1])

                        if move.delta > DELTA_ADD_THRESHOLD:  # if delta>0
                            avail_moves.append(move)

                    else:
                        # wymiana wierzchołków

                        i, j = path.index(v), opposite_path.index(candidate)

                        move = Move(path[i], opposite_path[j], 'vertexes', calc_delta(path, opposite_path, i, j, matrix, mode="vertexes"), None, None)
                        if move.delta > DELTA_ADD_THRESHOLD:
                            move_delta_update_or_append(move, avail_moves)

    avail_moves.sort(key=lambda x: x.delta, reverse=True)

    return avail_moves


def is_subpath(path, subpath):  # sprawdzanie czy podsciezka subpath znajduje sie w ścieżce path
    path_ = path + path[1:]  # for path is a cycle

    for i in range(len(path_)):
        if path_[i:i + len(subpath)] == subpath: return True

    return False


def is_applicable(move, a, b):
    if move.mode == 'vertexes':

        if move.ver_a in a and move.ver_b in b or move.ver_a in b and move.ver_b in a:
            return True
        else:
            return False

    elif move.mode == 'edges':
        path = a if move.path_name == 'a' else b

        if not is_subpath(path, move.edges_to_swap):
            return False
        elif is_subpath(path, move.edges_to_swap[::-1]):
            return "reverse"
        elif is_subpath(path, move.edges_to_swap):
            return True
        else:
            return AttributeError

    else:
        raise AttributeError


"""Perform test"""

NUM_OF_ITERATIONS = 100


def perform_test(instance_filenames):
    for instance_filename in instance_filenames:

        inst = tsplib95.load(instance_filename)
        dist = vertex_to_dist_matrix(inst)

        for algorithm in ('candidates','memory','steepest_edges','nn'):

            results, time_results = [], []

            for i in range(NUM_OF_ITERATIONS):
                ts = timer()

                # if int(i * 100 / NUM_OF_ITERATIONS)%10==0: print(f"{int(i * 100 / NUM_OF_ITERATIONS)}, ", end="")
                # print(f"{int(i * 100 / NUM_OF_ITERATIONS)}, ", end="")

                if algorithm == 'nn': res_a, res_b = nearest_neighbor_algorithm(dist)
                elif algorithm == 'steepest_edges': res_a, res_b = steepest_edges(dist, instance_filename)
                elif algorithm == 'memory': res_a, res_b = steepest_edges_with_list(dist, candidates_mode=False)
                elif algorithm == 'candidates': res_a, res_b = steepest_edges_with_list(dist, candidates_mode=True, n_neighbors=10)
                else: raise AssertionError

                time_results.append(timer() - ts)

                print(f"{int(i * 100 / NUM_OF_ITERATIONS)} - {calc_cost(res_a, dist) + calc_cost(res_b, dist)}", end=", ")

                results.append((calc_cost(res_a, dist) + calc_cost(res_b, dist), res_a, res_b))  # sum of costs of a and b, start_a, start_b

                # save best result to file
                plt.clf()
                plt.gca().get_xaxis().set_visible(False)
                plt.gca().get_yaxis().set_visible(False)
                draw_points(inst)
                draw_path(res_a, res_b, inst)
                plt.title(f"Aggregate cost = {calc_cost(res_a, dist) + calc_cost(res_b, dist)}")
                plt.savefig(f"{instance_filename.replace('/', '-')}_{algorithm}.png")

            maxi = max(results, key=lambda item: item[0])
            mini = min(results, key=lambda item: item[0])

            # print(f"{start}, {method}, {instance_filename}, {version} min={mini[0]}, maxi={maxi[0]}, avg={sum([a[0] for a in results]) / len(results)}, avg_time={(te - ts)/NUM_OF_ITERATIONS}")

            print(
                f" {instance_filename}, {algorithm} | {int(sum([a[0] for a in results]) / len(results))} ({int(mini[0])} - {int(maxi[0])}), {round(sum(time_results) / len(time_results), 3)} ({round(min(time_results), 3)} - {round(max(time_results), 3)})")


            # save best result to file
            plt.clf()
            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
            draw_points(inst)
            draw_path(mini[1], mini[2], inst)
            plt.title(f"Aggregate cost = {mini[0]}")
            plt.savefig(f"{instance_filename.replace('/', '-')}_{algorithm}.png")


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
    perform_test(instance_filenames=['kroA200.tsp', 'kroB200.tsp'])

else:
    # load instance
    instance = tsplib95.load('kroA100.tsp')
    dist_matrix = vertex_to_dist_matrix(instance)
    mini = 1000000
    while True:
        a, b = nearest_neighbor_algorithm(dist_matrix, instance)
        cost = calc_cost(a, dist_matrix) + calc_cost(b, dist_matrix)
        if cost < mini:
            print(cost, a, b)
            mini = cost
            save_plot(a, b, "out.png")

    # save_plot(a, b, "out.png")

"""noooooooowe zmiany"""
