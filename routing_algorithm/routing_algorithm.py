import random
from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse.csgraph
import scipy.spatial.distance
from shapely.geometry import LineString
from sklearn.cluster import KMeans
import numpy.ma as ma
import concurrent.futures

import concurrent.futures
from collections import defaultdict

from sklearn.cluster import KMeans
import pulp
from tqdm import tqdm

from utilities import euclidean


def routing_algorithm(pars, waypoints, stations, n_depots, use_mp=True):
    clusters = cluster_waypoints(pars, waypoints, n_depots)
    endurance = pars['endurance_seconds'] * pars['drone_speed_mps']
    max_t_bar = pars['maximum_service_time_hours'] * 60 * 60 * pars['drone_speed_mps']
    if pars.get('mp', False) or use_mp:
        routes = route_as_mp(clusters, endurance, max_t_bar)
    else:
        routes = route_serially(clusters, endurance, max_t_bar)
    ...


def route_as_mp(clusters, endurance, max_t_bar):
    results = list()
    routes = dict()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for cluster, waypts in tqdm(clusters.items(), desc='Submitting'):
            results.append(
                executor.submit(tour_maker_wrapper_for_mp, [clusters, cluster, waypts, endurance, max_t_bar]))
        for result in tqdm(concurrent.futures.as_completed(results), desc='Getting Results'):
            cluster, tour, tour_as_wp, dist = result.result()
            routes[cluster] = dict(cluster=cluster, tour=tour, tour_as_wp=tour_as_wp, dist=float(dist))
    return routes


# def cluster_waypoints(pars, waypoints, n_depots):
#     if n_depots == 1:
#         return {0: waypoints}
#     if not isinstance(waypoints, np.ndarray):
#         waypoints = np.array(list(waypoints))
#     kmeans = KMeans(n_clusters=n_depots, init='k-means++',
#                     max_iter=300, random_state=pars.get('random_seed'),
#                     algorithm='auto')
#     kmeans.fit(waypoints)
#     clusters = defaultdict(list)
#     for point, cluster in zip(waypoints, kmeans.labels_):
#         clusters[cluster].append(tuple(point))
#     return clusters


def route_waypoints(clusters, method, MP=True):
    routes, distances = dict(), dict()
    if method.lower() in {'nearest insertion', 'nearest_insertion'}:
        if MP:
            route_as_mp(clusters, None, None)
        else:
            ...
    elif method.lower() in {'exact'}:
        ...
    else:
        raise NotImplementedError(f"Method:'{method}' not implemented")
    return routes, distances


def find_closest_fire_stations(routes, stations):
    data = {
        cluster: {
            (station_id, wp): euclidean(wp, station_data['point'])
            for station_id, station_data in stations.items() for wp in
            (route_data['tour_as_wp'][0], route_data['tour_as_wp'][-1])
        }
        for cluster, route_data in routes.items()
    }
    problem = pulp.LpProblem('find closes stations', pulp.LpMinimize)
    x = {(i, j, k): pulp.LpVariable(f"x_{i}_{j[:14]}_{k}", 0, 1, pulp.LpInteger) for i in routes.keys() for j, k in
         data[i].keys()}
    problem += pulp.lpSum(data[i][(j, k)] * x[i, j, k] for i in routes.keys() for j, k in data[i].keys()), "ObjFn"
    for i in routes.keys():
        problem += pulp.lpSum(x[(i, j, k)] for j, k in data[i].keys()) == 1, f"C1@{i}"
    problem.solve(pulp.GUROBI_CMD(msg=0))
    solution = {key[0]: dict(station_name=key[1], next_wpt=key[2]) for key, value in x.items() if value.varValue > 0.9}
    for cluster, route_data in routes.items():
        routes[cluster]['station'] = solution[cluster]['station_name']
        next_wpt = solution[cluster]['next_wpt']
        if next_wpt == routes[cluster]["tour_as_wp"][0]:
            routes[cluster]["tour_as_wp"].insert(0, stations[solution[cluster]['station_name']]['point'])
        else:
            routes[cluster]["tour_as_wp"].append(stations[solution[cluster]['station_name']]['point'])
            routes[cluster]["tour_as_wp"].reverse()
            routes[cluster]["tour"].reverse()
    return routes


def construct_path_heuristic(dist_matrix, start_min_arc=True, c_num=None, max_tour_len=None):
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
        if max_tour_len and min(delta_d_2, delta_d_1) > max_tour_len:
            return tour, float('inf')
        p_bar.update()
    dist = sum(dist_matrix.at[p1, p2] for p1, p2 in zip(tour, tour[1:]))
    return tour, dist


def tour_maker_wrapper_for_mp(args):
    return tour_maker_wrapper(*args)


def tour_maker_wrapper(clusters, cluster, waypoints, endurance, max_t_bar):
    clusters[cluster] = sorted(sorted(waypoints), key=lambda x: x[1])
    dist_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(clusters[cluster]))

    tour1, dist1 = construct_path_heuristic(dist_matrix, c_num=cluster, start_min_arc=True, max_tour_len=max_t_bar)
    tour2, dist2 = construct_path_heuristic(dist_matrix, c_num=cluster, start_min_arc=False, max_tour_len=max_t_bar)

    tour = tour1 if dist1 < dist2 else tour2
    dist = min(dist1, dist2)
    tour_as_wp = [clusters[cluster][idx] for idx in tour]
    if max_t_bar and dist > max_t_bar:
        return cluster, tour, tour_as_wp, dist
    # tour_as_wp = clean_up_intersections(tour_as_wp, 0)
    dist = sum(euclidean(p1, p2) for p1, p2 in zip(tour_as_wp, tour_as_wp[1:]))
    return cluster, tour, tour_as_wp, dist
