import concurrent.futures
import random

import numpy.ma as ma
import scipy.sparse.csgraph
import scipy.spatial.distance
from collections import defaultdict, namedtuple

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

from utilities import euclidean


def cluster_waypoints_with_k_means(waypoints, n_depots, random_seed=None):
    if n_depots == 1:
        return {0: waypoints}
    if not isinstance(waypoints, np.ndarray):
        waypoints = np.array(list(waypoints))
    kmeans = KMeans(n_clusters=n_depots, init='k-means++',
                    max_iter=300, random_state=random_seed,
                    algorithm='auto')
    kmeans.fit(waypoints)
    clusters = defaultdict(list)
    for point, cluster in zip(waypoints, kmeans.labels_):
        clusters[cluster].append(tuple(point))
    return clusters


def route_clusters(clusters, method):
    if method.lower() in {'nearest insertion', 'nearest_insertion'}:
        tours, dists, t_bar = route_clusters_with_nearest_insertion(clusters)
    elif method.lower() in {'exact'}:
        tours, dists, t_bar = None, None, None
    else:
        raise NotImplementedError(f"Method:'{method}' not implemented")
    return tours, dists, t_bar


def route_clusters_with_nearest_insertion(clusters, MP=True):
    if len(clusters) == 1:
        MP = False
    if MP:
        tours, dists, t_bar = route_clusters_with_nearest_insertion_using_MP(clusters)
    else:
        tours, dists, t_bar = route_clusters_with_nearest_insertion_serially(clusters)
    return tours, dists, t_bar


def route_clusters_with_nearest_insertion_serially(clusters):
    tours, dists, t_bar = dict(), dict(), -1
    for cluster, waypts in tqdm(clusters.items(), desc='Routing...'):
        tour_as_wp, dist, _ = route_waypoints_with_nearest_insertion(waypts, c_num=cluster)
        dists[cluster] = dist
        tours[cluster] = tour_as_wp
    t_bar = max(dists.values())
    return tours, dists, t_bar


def route_clusters_with_nearest_insertion_using_MP(clusters):
    tours, dists, t_bar = dict(), dict(), -1
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list()
        for cluster, waypts in tqdm(clusters.items(), desc='Submitting'):
            results.append(
                executor.submit(route_waypoints_with_nearest_insertion_WRAPPER, [waypts, cluster]))
        for result in tqdm(concurrent.futures.as_completed(results), desc='Getting Results'):
            tour_as_wp, dist, cluster = result.result()
            dists[cluster] = dist
            tours[cluster] = tour_as_wp
    t_bar = max(dists.values())
    return tours, dists, t_bar


def route_waypoints_with_nearest_insertion_WRAPPER(args):
    return route_waypoints_with_nearest_insertion(*args)


def route_waypoints_with_nearest_insertion(waypoints, c_num=None):
    waypoints = sorted(sorted(list(waypoints)), key=lambda x: x[1])
    dist_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(waypoints))

    tour, dist = construct_path_nearest_insertion_heuristic_v2(dist_matrix, start_min_arc=True, c_num=c_num)
    # tour2, dist2 = construct_path_nearest_insertion_heuristic(dist_matrix, start_min_arc=False, c_num=c_num)
    # tour = tour if dist < dist2 else tour2
    # dist = min(dist, dist2)
    tour_as_wp = [waypoints[idx] for idx in tour]
    dist = sum(euclidean(p1, p2) for p1, p2 in zip(tour_as_wp, tour_as_wp[1:]))
    return tour_as_wp, dist, c_num


def construct_path_nearest_insertion_heuristic_v2(dist_matrix, start_min_arc=True, c_num=None):
    def _get_val_from_dist_matrix(_i, _j, _dist_matrix):
        if _i in {-888, -999} or _j in {-888, -999}:
            return 0
        if _i == _j:
            return 0
        return _dist_matrix[_i, _j]

    def _d_ijk(_i, _j, _k, _dist_matrix):
        return _get_val_from_dist_matrix(_i, _k, _dist_matrix) + \
               _get_val_from_dist_matrix(_k, _j, _dist_matrix) - \
               _get_val_from_dist_matrix(_i, _j, _dist_matrix)

    n_cities = len(dist_matrix)
    D_ijk = namedtuple("D_ijk", ["i", "j", "k", "val"])
    masked_dist_matrix = ma.masked_array(dist_matrix, mask=dist_matrix == 0)
    if start_min_arc:
        desired_val = masked_dist_matrix.min()
    else:
        desired_val = masked_dist_matrix.max()
    arc = np.where(desired_val == dist_matrix)
    tour = list(random.choice(arc))
    tour = [-999] + tour + [-888]

    city_ids = set(range(n_cities))
    [city_ids.discard(i) for i in tour]
    idx_i = 1
    change_arc_list = [D_ijk(i, j, k, _d_ijk(i, j, k, dist_matrix))
                       for k in city_ids for i, j in zip(tour, tour[1:])
                       ]

    p_bar = tqdm(total=len(dist_matrix), position=0, leave=False, desc=f"Routing Cluster {c_num}")
    while len(tour) < n_cities + 2:
        # change_arc_list += [D_ijk(i, j, k, _d_ijk(i, j, k, dist_matrix))
        #                     for k in city_ids
        #                     for i, j in
        #                     zip(tour[min(idx_i - 2, 0):min(idx_i + 2, len(tour) - 1)], tour[min(idx_i - 2, 0) + 1:])
        #                     ]
        change_arc_list = [
            D_ijk(i, j, k, _d_ijk(i, j, k, dist_matrix))
            for k in city_ids for i, j in zip(tour, tour[1:])
        ]
        change_arc_list.sort(key=lambda x: x.val)
        if change_arc_list:
            near_insert = change_arc_list.pop(0)
            # _i, _j = tour.index(near_insert.i), tour.index(near_insert.j)
            while not (near_insert.k in city_ids):  # and #not in tour and
                # near_insert.i in tour and
                # near_insert.j in tour and
                # abs(_i - _j) == 1):
                near_insert = change_arc_list.pop(0)
                # _i, _j = tour.index(near_insert.i), tour.index(near_insert.j)
            idx_i = tour.index(near_insert.i)
            tour = tour[:idx_i + 1] + [near_insert.k] + tour[idx_i + 1:]
            city_ids.discard(near_insert.k)
        else:
            assert False, "Something Has Gone Wrong Here!"
        p_bar.set_postfix_str(f"{len(tour) - 2}")
        p_bar.update()
    return tour[1:-1], 0


def construct_path_nearest_insertion_heuristic(dist_matrix, start_min_arc=True, c_num=None):
    masked_dist_matrix = ma.masked_array(dist_matrix, mask=dist_matrix == 0)
    if start_min_arc:
        desired_val = masked_dist_matrix.min()
    else:
        desired_val = masked_dist_matrix.max()
    arc = np.where(desired_val == dist_matrix)
    tour = list(random.choice(arc))
    dist_matrix = pd.DataFrame(dist_matrix).replace(0, np.nan)
    p_bar = tqdm(total=len(dist_matrix), position=0, leave=False, desc=f"Routing Cluster {c_num}")
    while len(tour) < len(dist_matrix):
        dm_explore = dist_matrix.filter(items=tour, axis=0).filter(
            items=set(dist_matrix.index).difference(set(tour)),
            axis=1)
        i = dm_explore.min(axis=0).idxmin()
        j = dm_explore[i].idxmin()
        loc_j = tour.index(j)
        test_tour_1 = tour[:loc_j] + [i] + tour[loc_j:]
        test_tour_2 = tour[:loc_j + 1] + [i] + tour[loc_j + 1:]
        delta_d_1 = sum(dist_matrix.at[p1, p2] for p1, p2 in zip(test_tour_1, test_tour_1[1:]))
        delta_d_2 = sum(dist_matrix.at[p1, p2] for p1, p2 in zip(test_tour_2, test_tour_2[1:]))
        if delta_d_1 < delta_d_2 or (delta_d_1 == delta_d_2 and random.choice([0, 1]) == 1):
            # If perchance, the distances are equal, choose randomly
            tour = tour[:loc_j] + [i] + tour[loc_j:]
            p_bar.set_postfix_str(f"d = {delta_d_1}")
        else:  # f delta_d_2 < delta_d_1:
            tour = tour[:loc_j + 1] + [i] + tour[loc_j + 1:]
            p_bar.set_postfix_str(f"d = {delta_d_2}")
        p_bar.update()
    dist = sum(dist_matrix.at[p1, p2] for p1, p2 in zip(tour, tour[1:]))
    return tour, dist
