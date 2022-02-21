import copy
import math
import random
from collections import Counter, namedtuple

from station_removal_heuristic.StoppingCriteria import StoppingCriteria
from station_removal_heuristic.station_removal_heuristic import _convert_df_to_alternative_data_structure, \
    log_data_to_csv, perturb_solution
from utilities import euclidean


def simple_make_solution(up_stations, tornado_event, pars):
    waypoints = tornado_event.waypoints
    endurance = pars['endurance_seconds'] * pars['drone_speed_mps']
    max_t_bar = pars['maximum_service_time_hours'] * 60 * 60 * pars['drone_speed_mps']
    num_wpts = len(waypoints)
    scanning_d = pars.get('scanning_r', 300) * 2
    approx_len = scanning_d * num_wpts
    n_stations_needed = math.ceil(approx_len/max_t_bar)
    station_min_matrix = sorted([(min(euclidean(i, up_stations[j]['point']) for i in waypoints), j) for j in up_stations.keys()])
    stations_to_use = station_min_matrix[:n_stations_needed]
    first_leg_dists, used_station_keys = zip(*stations_to_use)
    return max(first_leg_dists) + (approx_len / n_stations_needed), used_station_keys


def simplified_station_removal_heuristic(fire_stations, waypoints,
                                         tornado_cases, pars, heuristic_pars,
                                         init_start_smaller=0.5):
    initial_solution = _convert_df_to_alternative_data_structure(fire_stations)
    if init_start_smaller is not None and 0.1 <= init_start_smaller <= 1.0:
        initial_solution = list(initial_solution.items())
        random.shuffle(initial_solution)
        initial_solution = initial_solution[:math.ceil(len(initial_solution))*init_start_smaller]
        initial_solution = dict(initial_solution)
    best_solution = copy.deepcopy(initial_solution)
    current_solution = copy.deepcopy(best_solution)
    heuristic_pars["local_iter_count"] = 9999999
    stopping_criteria = StoppingCriteria(heuristic_pars)

    depot_failure = pars.get('depot_failure', 0.05)

    stopping_criteria.start()

    fire_station_counter = Counter()
    ResultCounter = namedtuple('ResultCounter', ['n_fire_stations', 't_bar', "fire_station_keys"])
    result_counter = []
    k = 0
    row = ['kounter', 'n_routes', "t_bar", "n_stations", "n_waypoints", "date"]
    log_data_to_csv(row, filename='simple_prj_william_log')
    while stopping_criteria.is_not_complete():
        stopping_criteria.reset_local_counter()
        for _tornado_date in tornado_cases.dates:
            print(f"\n{_tornado_date}")
            tornado_date, tornado_event = tornado_cases.get_specific_event(_tornado_date)
            for _ in range(3):  # while stopping_criteria.is_not_complete():
                up_stations = {
                    key: value for key, value in current_solution.items()
                    if random.random() > depot_failure
                }
                t_bar, stations_used = simple_make_solution(up_stations, tornado_event, pars)
                result_counter.append(ResultCounter(len(stations_used), t_bar, tuple(stations_used)))
                fire_station_counter.update(tuple(stations_used))
                stopping_criteria.update_counter()
                k += 1
                row = [k, len(stations_used), t_bar, len(current_solution), len(tornado_event.waypoints), str(_tornado_date)]
                print(row)
                log_data_to_csv(row, filename='simple_prj_william_log')
        current_solution = perturb_solution(current_solution, initial_solution,
                                            fire_station_counter, result_counter, perturbation=None)
        stopping_criteria.reset_local_counter()
