import datetime


def date_time_for_filename(date_to_stringify=datetime.datetime.now()):
    return date_to_stringify.strftime("%Y%m%d_%H%M%S")


def euclidean(p1, p2):
    return pow(sum(pow((a - b), 2) for a, b in zip(p1, p2)), 0.5)


def midpoint(p1, p2):
    return tuple((j + i) / 2 for i, j in zip(p1, p2))
