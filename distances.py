"""
Created on:     2016-04-12
Author:         Sophie Murray
Developed in:   Python 2.7.11 |Anaconda 2.3.0 (x86_64)| (default, Dec  6 2015, 18:57:58)
                [GCC 4.2.1 (Apple Inc. build 5577)] on darwin
Description:    This code is supplementary to zalando_solution.py, showing alternative
                ways to calculate distances from points to points/lines/line segments in
                cartesian or spherical geometry.
                It was something I had initially researched before discovering the
                shapely package for geometrical analysis, which I ultimately went with
                instead of these mathematical versions.
"""

import numpy as np

def dist_points(p1, p2):
    """Compute distance between
        two points with Pythagorus
        """
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]

    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


def dist_point_line(line_start, line_end, point):
    """Compute distance between
        point (x0,y0)
        and line between (x1, y1) and (x2, y2)
        """
    x0 = float(point[0])
    y0 = float(point[1])
    x1 = float(line_start[0])
    y1 = float(line_end[1])
    x2 = float(line_end[0])
    y2 = float(line_end[1])

    distance = ((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2. + (x2-x1)**2.)
    return distance


def dist_point_seg(line_start, line_end, point):
    """Code to calculate distance from point to line segment,
        modified from original by 'quano' on Stack Overflow:
        http://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
        Note,
        - (x3, y3) is the 'point'
        - (x1, y1) is starting point of line segment
        - (x2, y2) is ending point of line segement
        """
    x1 = line_start[0]
    y1 = line_start[1]
    x2 = line_end[0]
    y2 = line_end[1]
    x3 = point[0]
    y3 = point[1]

    px = x2 - x1
    py = y2 - y1

    pxy = px*px + py*py

    u = ((x3 - x1) * px + (y3 - y1) * py) / float(pxy)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance

    dist = np.sqrt(dx*dx + dy*dy)

    return dist


def haversine(lon1, lat1, lon2, lat2, earth_radius):
    """
        Great circle distance between two points using Haversine formula
        modified from Michael Dunn's answer on Stack Overflow:
        http://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
        Answer specified in decimal degrees.
        """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    delt_lon = lon2 - lon1
    delt_lat = lat2 - lat1
    a = np.sin(delt_lat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(delt_lon/2)**2
    c = 2 * earth_radius * np.asin(np.sqrt(a))

    return c


