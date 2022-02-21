#%% md

# Disaster Prep Tool

#%%

from collections import Counter, namedtuple

from tqdm import tqdm

from TornadoCaseGetter import TornadoCaseGetter
from disaster_prep_location_tool import read_data
from parameters.parfile_reader import parfile_reader

#%%

# Read Parameters and Data
from station_removal_heuristic.simplified_station_removal_heuristic import simplified_station_removal_heuristic
from station_removal_heuristic.station_removal_heuristic import station_removal_heuristic

def main():
    pars = parfile_reader("./parameters/par1.json")

    # Reading and ensuring data is correct
    fire_stations, sbws, waypoints, tornado_db = read_data(pars)

    #%%

    # Creating a class to organize historic tornado events
    # and allows for 'calling' of a random event
    tornado_cases = TornadoCaseGetter(tornado_db, sbws, waypoints)
    # for
    # e

    #%%
    # print()
    # with open('WaypointBins.csv', 'w') as f:
    #     f.write(f'date, wpts\n')
    #     for _tornado_date in tornado_cases.dates:
    #         tornado_date, tornado_event = tornado_cases.get_specific_event(_tornado_date)
    #         print(f"{_tornado_date} | {len(tornado_event.waypoints)}")
    #         f.write(f"{_tornado_date},{len(tornado_event.waypoints)}\n")
    #
    #
    # assert False
    fire_station_counter = Counter()
    ResultCounter = namedtuple('ResultCounter',['n_fire_stations','t_bar', "fire_station_keys"])


    #%%
    # date, event= tornado_cases.get_random_event()
    # print(f"{date}, {len(event.waypoints)}")
    #
    # while not (50 < len(event.waypoints) < 1500):
    #     date, event = tornado_cases.get_random_event()
    #     print(f"{date}, {len(event.waypoints)}")

    #%%

    # event.route_data.get_route(4)
    # event.route_data.plot_helper(4)
    # station_removal_heuristic(fire_stations, waypoints, tornado_cases, pars, dict())
    simplified_station_removal_heuristic(fire_stations, waypoints, tornado_cases, pars, dict())

    #%%

if __name__ == "__main__":
    main()
