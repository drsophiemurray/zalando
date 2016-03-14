# .........

import numpy as np
from math import radians, sin, cos, sqrt, asin


def dist_points(p1, p2):
	"""Compute distance between two points with Pythagorus"""
	x1 = p1[0]; y1 = p1[1]
	x2 = p2[0]; y2 = p2[1]
	distance =  np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
	return distance
	
	
def dist_point_line(line_start, line_end, point):
 	"""Compute distance between point (x0,y0) and line
 	between (x1, y1) and (x2, y2)"""
 	x0 = float(point[0]); y0 = float(point[1])
 	x1 = float(line_start[0]); y1 = float(line_end[1])
 	x2 = float(line_end[0]); y2 = float(line_end[1])
 	distance = ( (y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1 ) / np.sqrt( (y2-y1)**2. + (x2-x1)**2. ) 
 	return distance
 	

def dist_point_seg(line_start, line_end, point): # x3,y3 is the point
	"""Code to calculate distance from point to line segment,
	modified from original by 'quano' on Stackoverflow: 
	http://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
	"""
	
	x1 = line_start[0] ; y1 = line_start[1]
	x2 = line_end[0] ; y2 = line_end[1]
	x3 = point[0] ; y3 = point[1]
	
	px = x2 - x1
	py = y2 - y1
	
	something = px*px + py*py
	
	u =  ((x3 - x1) * px + (y3 - y1) * py) / float(something)
	
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
    (Source: Stackoverflow)
    """
    # have to use radians 
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2]) 

    delt_lon = lon2 - lon1 
    delt_lat = lat2 - lat1 
    a = math.sin(delt_lat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(delt_lon/2)**2
    c = 2 * earth_radius * math.asin(math.sqrt(a)) 

    return c
    

