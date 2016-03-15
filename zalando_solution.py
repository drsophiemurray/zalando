#........
import gmplot
import matplotlib as mpl
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import stats
from shapely.geometry import LineString, Point

# Coordinates given in teaser question
R_E = 6371 #km
BRAN_GPS = (52.516288, 13.377689)
SAT_GPS_START = (52.590117, 13.39915)
SAT_GPS_END = (52.437385, 13.553989)
SPREE_GPS = [(52.529198, 13.274099), (52.531835, 13.292340),
             (52.522116, 13.298541), (52.520569, 13.317349),
             (52.524877, 13.322434), (52.522788, 13.329000),
             (52.517056, 13.332075), (52.522514, 13.340743),
             (52.517239, 13.356665), (52.523063, 13.372158),
             (52.519198, 13.379453), (52.522462, 13.392328),
             (52.520921, 13.399703), (52.515333, 13.406054),
             (52.514863, 13.416354), (52.506034, 13.435923),
             (52.496473, 13.461587), (52.487641, 13.483216),
             (52.488739, 13.491456), (52.464011, 13.503386)]
SW_CORNER = (52.464011, 13.274099)

# Lets keep to km for nicer plots
M_TO_KM = 1000.

# Other info given in teaser
BRAN_MEAN = 4700. / M_TO_KM
BRAN_MODE = 3877. / M_TO_KM
SAT_DIST = 2400. / M_TO_KM
SPREE_DIST = 2730. / M_TO_KM

# Some plot settings
mpl.rc('font', family='serif', weight='normal', size=10)

# Some settings I have defined
# Search box size
LAT_RANGE = (52.41583, 52.61167)
LON_RANGE = (13.23389, 13.58944)
# Search resolution (placed up here in case want to change it):
RES = 0.1


def main():
    """......
        """
    ##First I mapped all the information as a sanity check,
    ##Note, my first version was using Basemap but commented out as it totally sucked...
    plot_teaser_coords()

	##lets work with the info we have been given re: the probability distributions.
    bran_lognorm = log_normal(BRAN_MEAN, BRAN_MODE)
    sat_norm = normal(SAT_DIST, mu=0.)
    spree_norm = normal(SPREE_DIST, mu=0.)

	##I plotted the below as a sanity check, 
	##commented out as really not needed for the solution
#    plot_distribs(bran_lognorm, sat_norm, spree_norm)

	##I decided to use the conversion equations provided to me to work in cartesian
	##rather than spherical coordinates, so next step is to convert everything needed.
    bran_coords = sphere_to_cart(BRAN_GPS[0], BRAN_GPS[1])
    sat_coords_start = sphere_to_cart(SAT_GPS_START[0], SAT_GPS_START[1])
    sat_coords_end = sphere_to_cart(SAT_GPS_END[0], SAT_GPS_END[1])
    spree_coords = [sphere_to_cart(coord[0], coord[1]) for coord in SPREE_GPS]

	##I also need to define a search box,
	##which I based on the min/max of the coords provided
    x_shape, box = search_box(LAT_RANGE, LON_RANGE, RES)

	##Now calculate pdfs
    bran_prob = pdf_point(box, Point(bran_coords), bran_lognorm)
    sat_prob = pdf_point(box, LineString([(sat_coords_start), (sat_coords_end)]), sat_norm)
    spree_prob = pdf_point(box, LineString(spree_coords), spree_norm)

    bran_z = np.array(bran_prob).reshape(x_shape)
    sat_z = np.array(sat_prob).reshape(x_shape)
    spree_z = np.array(spree_prob).reshape(x_shape)

	##Convert back to spherical coords
    lat_grid, lon_grid = search_grid(box, x_shape)

	##Combine them and print out max
    total_z = bran_z + sat_z + spree_z
    max_loc = np.where(total_z == total_z.max())
    analyst_loc = (float(lat_grid[max_loc]), float(lon_grid[max_loc]))
    print analyst_loc

	##Plot the distributions
    plot_pdf(bran_z, sat_z, spree_z, total_z, lat_grid, lon_grid, max_loc)

    plot_solution(analyst_loc)


def cart_to_sphere(p_x, p_y):
    """......
        """
    sw_lon = SW_CORNER[1]
    sw_lat = SW_CORNER[0]
    p_lat = (p_y / 111.323) + sw_lat
    p_lon = p_x / (111.323 * np.cos(sw_lat * np.pi / 180.)) + sw_lon
    return p_lat, p_lon


def log_normal(mean, mode):
    """
        #mode = np.exp(mu - sig**2)
        #mean = np.exp(mu + (sig**2/2))
        """
    sigma = np.sqrt((2./3.) * (np.log(mean) - np.log(mode)))
    mu = ((2.*np.log(mean)) + np.log(mode)) / 3.
    lognorm = stats.lognorm(sigma, loc=0., scale=np.exp(mu))
    return lognorm


def normal(distance, mu):
    """......
        """
    sigma = distance / stats.norm.ppf(1. - (1. - 0.95) / 2.)
    norm = stats.norm(mu, sigma)
    return norm


def pdf_point(box, point, distrib):
    """......
        """
    prob = []
    for xy in box:
        xy = Point(xy)
        distance = xy.distance(point)
        prob.append(distrib.pdf(distance))
    return prob


#def plot_distribs(bran, sat, spree):
#    """......
#        """
#    x = np.linspace(0, 10, 101)
#    fig = plt.figure()
#    ax = fig.add_subplot(1, 1, 1)
#    ax.plot(x, bran.pdf(x),
#            label = 'Brandenburg', color = 'k')
#    ax.plot(x, sat.pdf(x),
#            label = 'Satellite', color = 'm')
#    ax.plot(x, spree.pdf(x),
#            label = 'Spree', color = 'b')
#    ax.legend()
#    fig.savefig('./results/norms.png', format = "png")
#    plt.close()


def plot_pdf(bran_z, sat_z, spree_z, total_z, lat, lon, loc):
    """......
        """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    alpha = 0.5
    lw = 0.1
    ax.plot_surface(lon, lat, bran_z,
                    color='k',
                    alpha=alpha, lw=lw)
    ax.plot_surface(lon, lat, sat_z,
                    color='m',
                    alpha=alpha, lw=lw)
    ax.plot_surface(lon, lat, spree_z,
                    color='b',
                    alpha=alpha, lw=lw)
    fig.savefig('./results/distribs_all.png', format="png")
    plt.close()

    fig = plt.figure()
    ax_tot = fig.add_subplot(111, projection="3d")

    ax_tot.plot_surface(lon, lat, total_z,
                        color='y',
                        alpha=alpha, lw=lw)
    ax_tot.contourf(lon, lat, total_z,
                    alpha=alpha,
                    offset=0, cmap='hot')
    ax_tot.plot(lon[loc], lat[loc],
                'ko', markersize=2)
    fig.savefig('./results/distrib_total.png', format="png")
    plt.close()


def plot_solution(loc):
    """......
        """
    gmap = gmplot.GoogleMapPlotter.from_geocode("Berlin")

    gmap.marker(loc[0], loc[1],
                'r')

    gmap.draw("./results/analyst_location.html")


# def plot_teaser_basemap():
# 	"""......
#       """
# 	map = Basemap(projection='merc', resolution = 'l',
# 					llcrnrlat = SAT_GPS_START[0] - 1., urcrnrlat = SAT_GPS_START[0] + 1.,
# 					llcrnrlon = SAT_GPS_START[1] - 1., urcrnrlon = SAT_GPS_END[1] + 1.)
# 	map.etopo()
# 	map.drawcountries()
# 	# draw parallels
# 	map.drawparallels(np.arange(0., 90., 1.), labels = [1, 1, 0, 1])
# 	# draw meridians
# 	map.drawmeridians(np.arange(-180., 180., 1), labels = [1, 1, 0, 1])
# 	# draw gate position
# 	xt, yt = map(BRAN_GPS[1], BRAN_GPS[0])
# 	map.plot(xt, yt, 'ko', markersize = 2)
# 	# draw satellite path
# 	map.drawgreatcircle(SAT_GPS_START[1], SAT_GPS_START[0],
# 						SAT_GPS_END[1], SAT_GPS_END[0],
# 						linewidth = 2, color = 'm')
# 	# draw river
# 	spree_lats, spree_lons = zip(*SPREE_GPS)
# 	xt, yt = map(spree_lons, spree_lats)
# 	map.plot(xt, yt, linewidth = 2, color = 'b')
# 	plt.show()
# 	plt.close()


def plot_teaser_coords():
    """......
        """
    gmap = gmplot.GoogleMapPlotter.from_geocode("Berlin")

    spree_lats, spree_lons = zip(*SPREE_GPS)
    gmap.plot(spree_lats, spree_lons,
              'b', edge_width=5.)

    sat_lats = (SAT_GPS_START[0], SAT_GPS_END[0])
    sat_lons = (SAT_GPS_START[1], SAT_GPS_END[1])
    gmap.plot(sat_lats, sat_lons,
              'm', edge_width=5.)

    gmap.marker(BRAN_GPS[0], BRAN_GPS[1],
                'k')

    gmap.draw("./results/teaser_coords.html")


def search_box(lat_range, lon_range, interval):
    """......
        """
    xy_start = sphere_to_cart(lat_range[0], lon_range[0])
    xy_end = sphere_to_cart(lat_range[1], lon_range[1])

    x_range = np.arange(xy_start[0], xy_end[0], interval)
    y_range = np.arange(xy_start[1], xy_end[1], interval)

    x, y = np.meshgrid(x_range, y_range)

    box = []
    for i in range(0, len(x)):
        for j in range(0, len(y[i])):
            box.append([x[i][j], y[i][j]])

    return x.shape, box


def search_grid(box, x_shape):
    """......
        """
    lats = []
    lons = []

    for i, j in box:
        lat, lon = cart_to_sphere(i, j)
        lats.append(lat)
        lons.append(lon)

    lat_grid = np.array(lats).reshape(x_shape)
    lon_grid = np.array(lons).reshape(x_shape)

    return lat_grid, lon_grid


def sphere_to_cart(p_lat, p_lon):
    """......
        """
    sw_lon = SW_CORNER[1]
    sw_lat = SW_CORNER[0]

    p_x = (p_lon - sw_lon) * np.cos(sw_lat * np.pi / 180.) * 111.323
    p_y = (p_lat - sw_lat) * 111.323

    return p_x, p_y


if __name__ == '__main__':
    main()
