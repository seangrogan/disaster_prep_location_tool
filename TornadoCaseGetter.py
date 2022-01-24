import itertools
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from shapely.geometry import Point
from tqdm import tqdm

from routing_algorithm.routing_algo_refresh import cluster_waypoints_with_k_means, route_clusters


# from routing_algorithm.routing_algorithm import cluster_waypoints, route_waypoints

class TornadoCaseGetterIterator:
    def __init__(self, dates):
        self.dates = dates


class TornadoCaseGetter:
    def __init__(self, tornadoes, sbws, waypoints):
        _sbws = sbws[sbws['PHENOM'] == 'TO']
        self.sbw_cases, self.tor_cases, self.dates = self.__COLLATE_EVENTS(_sbws, tornadoes)
        # self.sbw_cases = self.sbw_cases[self.sbw_cases['PHENOM'] == 'TO']
        __temp_waypoints = pd.DataFrame(data=waypoints, index=waypoints, columns=['x', 'y'])
        with tqdm(total=len(self.sbw_cases), desc="Attaching waypoints to SBWs", leave=False) as pbar:
            self.sbw_cases['waypoints'] = \
                self.sbw_cases.apply(
                    lambda x: self.__FILTER_POINTS(__temp_waypoints, x, pbar),
                    axis=1,
                    result_type='reduce')

        self.sbw_cases.dropna(axis=0, how='any', thresh=None, subset=['waypoints'], inplace=True)
        self.sbw_cases, self.tor_cases, self.dates = self.__COLLATE_EVENTS(self.sbw_cases, self.tor_cases)
        temp_tornado_events = dict()
        for date in self.dates:
            _, _tornadoes, _sbws = self.get_specific_case(date)
            tornado_geometries = _tornadoes.geometry.to_list()
            sbw_geometries = _sbws.geometry.to_list()
            _waypoints = set(itertools.chain.from_iterable(_sbws.waypoints.to_list()))
            temp_tornado_events[date] = dict(
                tornado_geometries=tornado_geometries,
                sbw_geometries=sbw_geometries,
                waypoints=_waypoints,
                route_data=RouteClass(_waypoints)
            )
        self.tornado_events = pd.DataFrame.from_dict(temp_tornado_events, orient='index')
        self.dates = list(self.dates)
        self.dates.sort()
        self.current_date = self.dates[0]

    # def atta`ch_a_route_entry(self):
    #     with tqdm(total=len(self.sbw_cases), desc="Attaching RouteClass to SBWs", leave=False) as pbar:
    #         self.sbw_cases['route_data'] = \
    #             self.sbw_cases.apply(
    #                 lambda x: self.__FILTER_POINTS(..., x, pbar),
    #                 axis=1,
    #                 result_type='reduce')
    def __iter__(self):
        self.current_date_index = 0

        return self

    def __next__(self):
        if self.current_date_index >= len(self.dates):
            raise StopIteration
        curr_date = self.dates[self.current_date_index]
        self.current_date_index += 1
        return self.get_specific_event(curr_date)

    def iterator(self):
        pass

    def get_random_event(self):
        date = random.choice(self.dates)
        event = self.tornado_events.loc[date]
        return date, event

    def get_specific_event(self, date):
        # date = random.choice(self.dates)
        event = self.tornado_events.loc[date]
        return date, event

    def get_random_case(self):
        date = random.choice(self.dates)
        tornadoes = self.tor_cases[self.tor_cases.date == date]
        sbws = self.sbw_cases[self.sbw_cases.issued_date == date]
        return date, tornadoes, sbws

    def get_specific_case(self, date):
        tornadoes = self.tor_cases[self.tor_cases.date == date]
        sbws = self.sbw_cases[self.sbw_cases.issued_date == date]
        return date, tornadoes, sbws

    @staticmethod
    def __COLLATE_EVENTS(sbws, tornadoes):
        sbw_cases = []
        tor_cases = []
        sbws['issued_date'] = sbws['ISSUED'].dt.date
        tornado_dates = set(tornadoes['date'].to_list())
        sbw_dates = set(sbws['issued_date'].to_list())
        _dates = sorted(list(tornado_dates.intersection(sbw_dates)))
        for date in tqdm(sorted(list(_dates), reverse=True), desc="Checking Events", leave=False):
            temp_tornado_db, temp_sbws = tornadoes[tornadoes['date'] == date], sbws[sbws['issued_date'] == date]
            _temp_torn = temp_tornado_db[
                pd.concat([temp_tornado_db.intersects(row.geometry) for idx, row in temp_sbws.iterrows()], axis=1).max(
                    axis=1)]
            _temp_sbw = []
            for idx, row in temp_tornado_db.iterrows():
                __temp = temp_sbws[
                    pd.concat([temp_sbws.intersects(row.geometry)], axis=1).max(
                        axis=1)]
                if len(__temp) > 0:
                    _temp_sbw.append(__temp)
            # _temp_sbw = temp_sbws[
            #     pd.concat([temp_sbws.intersects(row.geometry) for idx, row in temp_tornado_db.iterrows()], axis=1).max(
            #         axis=1)]
            if len(_temp_sbw)>0:
                _temp_sbw = pd.concat(_temp_sbw)
                sbw_cases.append(_temp_sbw)
            tor_cases.append(_temp_torn)
        _tor_cases = pd.concat(tor_cases)
        _sbw_cases = pd.concat(sbw_cases)
        return _sbw_cases, _tor_cases, _dates

    @staticmethod
    def __FILTER_POINTS(waypoints, geom, pbar=None):
        if pbar:
            pbar.update()
        minx, miny, maxx, maxy = geom.geometry.bounds
        waypoints = waypoints[
            (waypoints.x >= minx - 1) & (waypoints.x <= maxx + 1) &
            (waypoints.y >= miny - 1) & (waypoints.y <= maxy + 1)
            ]
        waypoints = list(pt for pt in waypoints.index.tolist() if geom.geometry.contains(Point(pt[0], pt[1])))
        if waypoints:
            return waypoints
        return np.NaN

    def plot_helper(self, date):
        event = self.tornado_events.loc[date]
        plt.title(f"Date : {date} | Wpts : {len(event.waypoints)}")
        x, y = zip(*list(event.waypoints))
        plt.scatter(x, y)
        plt.savefig(f"./plots/{date}.png")
        plt.close()


class RouteClass:
    def __init__(self, waypoints):
        self._waypoints = list(waypoints)
        self._routes = dict()
        self._dists, self._t_bars = dict(), dict()

    def get_route(self, n_depots):
        if self._routes.get(n_depots, None):
            return self._routes.get(n_depots, None), self._dists[n_depots], self._t_bars[n_depots]
        else:
            routes, dists, t_bar = self.cluster_first_nearest_insertion_route_second(n_depots, self._waypoints)
            self._routes[n_depots] = routes
            self._dists[n_depots] = dists
            self._t_bars[n_depots] = t_bar
            return self._routes.get(n_depots, None), self._dists[n_depots], self._t_bars[n_depots]

    def get_number_of_waypoints(self):
        return len(self._waypoints)

    def plot_helper(self, n_depots):
        if n_depots not in self._routes:
            self.get_route(n_depots)
        plt.title(f"Num Depots {n_depots}, t_bar {self._t_bars[n_depots]}")
        for rte in self._routes[n_depots].values():
            x, y = zip(*rte)
            plt.plot(x, y)
            plt.show()
            plt.close()

    def __repr__(self):
        return f"RouteClass with {len(self._routes)} routes and {len(self._waypoints)} waypoints"

    @staticmethod
    def cluster_first_nearest_insertion_route_second(n_depots, _waypoints):
        clusters = cluster_waypoints_with_k_means(_waypoints, n_depots)
        routes, dists, t_bar = route_clusters(clusters, 'nearest insertion')
        return routes, dists, t_bar
