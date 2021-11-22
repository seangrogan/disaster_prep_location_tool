import json

from utilities import date_time_for_filename


def parfile_reader(par_file_name=None):
    with open(par_file_name) as pf_reader:
        pars = json.load(pf_reader)
        pars = {key.lower() if type(key) == str else key: value for key, value in pars.items()}
    if "name" not in pars:
        pars["name"] = f"NoParName{date_time_for_filename()}"
    return pars