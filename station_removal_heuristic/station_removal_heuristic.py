import copy
import itertools
import random
from collections import namedtuple, Counter

import pulp

from station_removal_heuristic.StoppingCriteria import StoppingCriteria
from utilities import euclidean


def station_removal_heuristic(fire_stations, waypoints, tornado_cases, pars, heuristic_pars):
    initial_solution = _convert_df_to_alternative_data_structure(fire_stations)
    best_solution = copy.deepcopy(initial_solution)
    current_solution = copy.deepcopy(best_solution)

    stopping_criteria = StoppingCriteria(heuristic_pars)

    depot_failure = pars.get('depot_failure', 0.05)

    stopping_criteria.start()

    fire_station_counter = Counter()
    ResultCounter = namedtuple('ResultCounter', ['n_fire_stations', 't_bar', "fire_station_keys"])
    result_counter = []

    while stopping_criteria.is_not_complete():
        stopping_criteria.reset_local_counter()
        while stopping_criteria.is_not_complete():
            stopping_criteria.update_counter()
            up_stations = {
                key: value for key, value in current_solution.items()
                if random.random() > depot_failure
            }
            routes, dists, t_bar = make_solution(up_stations, tornado_cases, current_solution, pars)


def _convert_df_to_alternative_data_structure(original_df):
    new_struct = {f'{index}_{row["id"]}_{row["name"]}'.replace(' ', '_'):
                      dict(name=row["name"], index=index, point=(row.geometry.x, row.geometry.y), row_id=row["id"])
                  for index, row in original_df.iterrows()}
    return new_struct


def assign_stations(routes, stations, exact=False):
    routes = routes[:]
    if exact:
        routes, used_stations = assign_stations_exact(routes, stations)
    else:
        routes, used_stations = assign_stations_heuristic(routes, stations)
    return routes, used_stations


def assign_stations_heuristic(routes, stations):
    used_stations, used_clusters = set(), set()
    StationRelation = namedtuple('StationRelation', ['station_key', 'cluster_num', 'reverse', 'dist'])
    station_relations = []
    for cluster_num, route in routes.items():
        for station_key, station_data in stations.items():
            station_relations.append(
                StationRelation(station_key, cluster_num, False, euclidean(route[0], station_data['point'])))
            station_relations.append(
                StationRelation(station_key, cluster_num, True, euclidean(route[-1], station_data['point'])))
    station_relations.sort(key=lambda x: x.dist)
    new_routes = dict()
    while len(new_routes) < len(routes):
        station_relation = station_relations.pop(0)
        if station_relation.station_key in new_routes:
            continue
        if station_relation.cluster_num in used_clusters:
            continue
        new_routes[station_relation.station_key] = routes[station_relation.cluster_num][:]
        if station_relation.reverse:
            new_routes[station_relation.station_key].reverse()
        new_routes[station_relation.station_key].insert(0, stations[station_relation.station_key]['point'])

        used_clusters.add(station_relation.cluster_num)
        used_stations.add(station_relation.station_key)

    return new_routes, used_stations


def assign_stations_exact(routes, stations):
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
    solution = {key[0]: dict(station_name=key[1], next_wpt=key[2]) for key, value in x.items() if
                value.varValue > 0.9}
    used_stations = []
    for cluster, route_data in routes.items():
        routes[cluster]['station'] = solution[cluster]['station_name']
        next_wpt = solution[cluster]['next_wpt']
        used_stations.append(solution[cluster]['station_name'])
        if next_wpt == routes[cluster]["tour_as_wp"][0]:
            routes[cluster]["tour_as_wp"].insert(0, stations[solution[cluster]['station_name']]['point'])
        else:
            routes[cluster]["tour_as_wp"].append(stations[solution[cluster]['station_name']]['point'])
            routes[cluster]["tour_as_wp"].reverse()
            routes[cluster]["tour"].reverse()
    return routes, used_stations


def update_dists(routes):
    dists = {key: sum(euclidean(i, j) for i, j in zip(route, route[1:])) for key, route in routes.items() }
    t_bar = max(dists.values())
    endurance = max(sum(euclidean(i, j) for i, j in zip(route+[route[0]], route[1:]+[route[0]]))
                    for route in routes.values())
    return dists, t_bar, endurance


def make_solution(up_stations, tornado_cases, all_stations, pars):
    tornado_date, tornado_event = tornado_cases.get_random_event()
    while not (50 < len(tornado_event.waypoints) < 2000):
        tornado_date, tornado_event = tornado_cases.get_random_event()
    print(f"{tornado_date}, {len(tornado_event.waypoints)}")

    endurance = pars['endurance_seconds'] * pars['drone_speed_mps']
    max_t_bar = pars['maximum_service_time_hours'] * 60 * 60 * pars['drone_speed_mps']
    depot_up_bound = min(len(up_stations), len(tornado_event.waypoints)//2)+1
    n_depots_to_check = list(range(1, depot_up_bound))
    start = int(1 + (pars.get('scanning_r', 300) * len(tornado_event.waypoints))//endurance)
    n_depots_to_check = n_depots_to_check[start-1:] + n_depots_to_check[start-1::-1] if start < len(n_depots_to_check) else n_depots_to_check
    for n_depots in n_depots_to_check:
        routes, dists, t_bar = tornado_event.route_data.get_route(n_depots)
        routes, used_stations = assign_stations_heuristic(routes, up_stations)
        dists, t_bar, endurance_check = update_dists(routes)
        if t_bar > max_t_bar:
            # Drones have exceeded the service time
            print(f"\nExceeds Service Dist ({max_t_bar}) by {(t_bar - max_t_bar):.2f}m")
            continue
        if endurance_check > endurance:
            # drones cannot complete the journey
            print(f"\nExceeds Endurance ({endurance}) by {(endurance_check - endurance):.2f}m")
            continue
        if endurance_check <= endurance and t_bar <= max_t_bar:
            print(f"\nFound Solution with {n_depots}")
            return routes, t_bar, endurance_check
    routes, dists, t_bar = tornado_event.route_data.get_route(max(n_depots_to_check))
    routes, used_stations = assign_stations_heuristic(routes, up_stations)
    dists, t_bar, endurance_check = update_dists(routes)
    if t_bar > max_t_bar:
        # Drones have exceeded the service time
        print(f"\nExceeds Service Dist ({max_t_bar}) by {(t_bar - max_t_bar):.2f}m")
        print(f"Using solution with {max(n_depots_to_check)}")
        return routes, t_bar, endurance_check
    if endurance_check > endurance:
        # drones cannot complete the journey
        print(f"\nExceeds Endurance ({endurance}) by {(endurance_check - endurance):.2f}m")
        print(f"Using solution with {max(n_depots_to_check)}")
        return routes, t_bar, endurance_check
    # value = calc_dist_of_solution(solution)
    # problem = pulp.LpProblem('TEST', pulp.LpMinimize)
    # x = {i: {j: pulp.LpVariable(f"x_{i}_{j}", 0, 1, pulp.LpInteger) for j in waypoints} for i in waypoints}
    # c = {i: {j: euclidean(i, j) for j in waypoints} for i in waypoints}
    # problem += pulp.lpSum(c[i][j] * x[i][j] for i in waypoints for j in waypoints), "ObjFn"
    # problem += pulp.lpSum(x[i][j]) - pulp.lpSum(x[i][j])
    # u = {i: pulp.LpVariable(f"u_{i}", 0, None, pulp.LpInteger) for i in waypoints}
    # for idx, i in enumerate(cities[1:], 1):
    #     for j in cities[1:]:
    #         if i != j:
    #             problem += u[i] - u[j] + len(cities) * x[i][j] <= len(cities) - 1, f"C3@{i}-{j}".replace(' ', '')

    return None, None, None