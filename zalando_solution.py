"""
Created on:     2016-04-12
Author:         Sophie Murray
Developed in:   Python 2.7.11 |Anaconda 2.3.0 (x86_64)| (default, Dec  6 2015, 18:57:58)
                [GCC 4.2.1 (Apple Inc. build 5577)] on darwin
Description:    - This code is an attempt at solving the teaser posed to
                    prospective data scientists at Zalando.
                - The teaser can be found here:
                    https://tech.zalando.com/jobs/data/65946-senior-data-scientist-m-f/?gh_jid=65946
                - More detailed information can be found in my readme file,
                    which I think will be useful to understand why I've
                    done what I've done! See:
                    https://github.com/sophiemurray/zalando
TODO:           - The analysis of the various sources of information could
                    be split up rather than one after another in the main
                    part of the code as ITS SO LONG.
                - Theres no error handling, feels wrong :O
"""

# Import needed packages
import os
import gmplot
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
#from mpl_toolkits.basemap import Basemap
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import stats
from shapely.geometry import LineString, Point

# Coordinates given in teaser question are defined
# up here for ease of use!
R_E = 6371  # km
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

# Search box size
LAT_RANGE = (52.41583, 52.61167)
LON_RANGE = (13.23389, 13.58944)
# Search resolution
RES = 0.1


def main():
    """This is the main analysis and what you run to get the resulting
        most likely analyst location. Main steps are:
        - Check out the given information.
        - Get probability distribution information for
            - Brandenburg Gate
            - Satellite path
            - River Spree
        - Convert from spherical to cartesian coordinates.
        - Create a grid to search over.
        - Calculate probability distributions for each of the
            three info sources listed above.
        - Combine probabilities linearly.
        - Convert back to latitiude/longitude.
        - Calculate most likely GPS coordinates of analyst.
        - Output this information for the recruiters.
        """
    # Just making a results folder for the output..
    if not os.path.isdir('results'):
        os.mkdir('results')

    # First I mapped all the information as a sanity check.
    # Note, my first version was using Basemap but commented it
    # out as its not so pretty if you dont have global maps...
#    plot_teaser_basemap()
    # Plotted onto a Google map instead!
    plot_teaser_coords()

	# Next work with the info that has bee provided
    # re: the probability distributions.
    bran_lognorm = log_normal(BRAN_MEAN, BRAN_MODE)
    sat_norm = normal(SAT_DIST, mu=0.)
    spree_norm = normal(SPREE_DIST, mu=0.)

	# I plotted the below as a sanity check, but have
	# commented out as really not needed for the solution.
#    plot_distribs(bran_lognorm, sat_norm, spree_norm)

    #---------------------------------

	# I decided to use the conversion equations provided to work
    # in cartesian rather than spherical coordinates,
	# so the next step is to convert everything needed.
    bran_coords = sphere_to_cart(BRAN_GPS[0], BRAN_GPS[1])
    sat_coords_start = sphere_to_cart(SAT_GPS_START[0], SAT_GPS_START[1])
    sat_coords_end = sphere_to_cart(SAT_GPS_END[0], SAT_GPS_END[1])
    spree_coords = [sphere_to_cart(coord[0], coord[1]) for coord in SPREE_GPS]

	# I also need to define a search box,
	# which I based on the min/max of the coords provided.
    xx, yy, box = search_box(LAT_RANGE, LON_RANGE, RES)

	# Now calculate pdfs.
    bran_prob = pdf_point(box, Point(bran_coords), bran_lognorm)
    sat_prob = pdf_point(box, LineString([(sat_coords_start),
                                          (sat_coords_end)]), sat_norm)
    spree_prob = pdf_point(box, LineString(spree_coords), spree_norm)

    # Reshape back into grid shape for plotting.
    bran_z = np.array(bran_prob).reshape(xx.shape)
    sat_z = np.array(sat_prob).reshape(xx.shape)
    spree_z = np.array(spree_prob).reshape(xx.shape)

    # I plotted the results here just as a sanity check,
    # but as before its commented out as its
    # not needed for the solution.
#    plot_contour_distribs(bran_z, sat_z, spree_z, xx, yy)

    #---------------------------------

	# Convert back to spherical coords.
    lat_grid, lon_grid = search_grid(box, xx.shape)

	# Combine them linearly and print out max.
    total_z = bran_z + sat_z + spree_z
    max_loc = np.where(total_z == total_z.max())
    analyst_loc = (float(lat_grid[max_loc]), float(lon_grid[max_loc]))
    print "Analyst most likely at:", analyst_loc,
    print "with a probability of", total_z.max()

    # Save the location in a text file.
    np.savetxt("./results/location.txt",
               np.c_[analyst_loc],
               fmt="%0.4f",
               header="latitude longitude")

	# Plot the distributions.
    plot_pdf(bran_z, sat_z, spree_z, total_z, lat_grid, lon_grid, max_loc)

    # Show location on Google map.
    plot_solution(analyst_loc)


# ===========================================================================
# All functions below are listed alphabetically rather than a
# logical progression due to my Met Office ways...

def axis_contour_settings(ax, xx, yy, text):
    """Setting up some plot (with matplotlib) settings
        for checking the probability distributions
        with plot_contour_distribs().
        """
    # axis limits
    ax.axis([xx.min(), xx.max(), yy.min(), yy.max()])
    # label the plots
    ax.text(xx.min(), yy.min(), text,
            path_effects=[PathEffects.withStroke(linewidth=0.8, foreground="w")])


def axis_pdf_settings(ax, xlabel=[], ylabel=[], zlabel=[]):
    """Setting up some plot (with matplotlib) settings
        for checking the probability distributions
        with plot_pdf().
        """
    # adding labels to the axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)


def cart_to_sphere(p_x, p_y):
    """Converting back from cartesian
        to spherical geometry using the
        equations originally provided by Zalando.
        """
    sw_lon = SW_CORNER[1]
    sw_lat = SW_CORNER[0]
    p_lat = (p_y / 111.323) + sw_lat
    p_lon = p_x / (111.323 * np.cos(sw_lat * np.pi / 180.)) + sw_lon
    return p_lat, p_lon


def log_normal(mean, mode):
    """Creating log normal, by first calculating
        sigma and mu by manipulating these equations:
        - mode = np.exp(mu - sig**2)
        - mean = np.exp(mu + (sig**2/2))
        then using the scipy.stats package.
        """
    sigma = np.sqrt((2. / 3.) * (np.log(mean) - np.log(mode)))
    mu = ((2. * np.log(mean)) + np.log(mode)) / 3.
    lognorm = stats.lognorm(sigma, loc=0., scale=np.exp(mu))
    return lognorm


def normal(distance, mu):
    """Creating info for normal distribution
        using the scipy.stats package.
        """
    sigma = distance / stats.norm.ppf(1. - (1. - 0.95) / 2.)
    norm = stats.norm(mu, sigma)
    return norm


def pdf_point(box, point, distrib):
    """Calculate the probability density function
        by first calculating distance from a point
        in 'box' to another 'point' with the
        shapely geometry package, then using
        scipy.stats package.
        """
    prob = []
    for xy in box:
        xy = Point(xy)
        distance = xy.distance(point)
        prob.append(distrib.pdf(distance))
    return prob


def plot_contour_distribs(bran_z, sat_z, spree_z, xx, yy):
    """Use matplotlib to plot a 2D image of the
        calculated probability distributions.
        Some plot settings are also defined with
        axis_contour_settings() since I find it
        easier to add more later if needed that way!
        TODO: could include the pcolor part within
        plot settings function as I normally do rather
        than repeating it three times in a row!!
        """
    # set data max/min for consistency
    norm = mpl.colors.Normalize(vmin=bran_z.min(),
                                vmax=bran_z.max())

    fig = plt.figure()
    fig.subplots_adjust(top=0.33)

    # plot result for brandenburg gate
    bran = fig.add_subplot(131)
    bran_fig = bran.pcolor(xx, yy, bran_z,
                           cmap='hot', norm=norm)
    axis_contour_settings(bran, xx, yy,
                          text='Brandenburg')
    # plot result for satellite path
    sat = fig.add_subplot(132)
    sat.pcolor(xx, yy, sat_z,
               cmap='hot', norm=norm)
    axis_contour_settings(sat, xx, yy,
                          text='Satellite')
    # plot result for river spree source
    spree = fig.add_subplot(133)
    spree.pcolor(xx, yy, spree_z,
                 cmap='hot', norm=norm)
    axis_contour_settings(spree, xx, yy,
                          text='Spree')

    fig.colorbar(bran_fig, label='Probability')
    fig.savefig('./results/distribs_2d.png',
                bbox_inches='tight',
                format="png")
    plt.close()


def plot_distribs(bran, sat, spree):
    """Quick and dirty plot of distribution
        information using matplotlib.
        """
    # create random space
    xy = np.linspace(0, 10, 101)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot result for brandenburg gate
    ax.plot(xy, bran.pdf(xy),
            label='Brandenburg', color='k')
    # plot result for satellite path
    ax.plot(xy, sat.pdf(xy),
            label='Satellite', color='m')
    # plot result for river spree source
    ax.plot(xy, spree.pdf(xy),
            label='Spree', color='b')
    ax.legend()
    fig.savefig('./results/norms.png',
                bbox_inches='tight',
                format="png")
    plt.close()


def plot_pdf(bran_z, sat_z, spree_z, total_z, lat, lon, loc):
    """Two plots are created here using matplotlib:
        - 3D figure of all three calculated distribtions.
        - 3D figure of total probability distribution.
        Some plot settings are also defined with
        axis_pdf_settings() since I find it
        easier to add more later if needed that way!
        TODO: could include the plot_surface part within
        plot settings function as I normally do rather
        than repeating it constantly!
        """
    # first the plot of all distributions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # some settings that are used multiple times
    # and could be changed..
    alpha = 0.5
    line_width = 0.1

    # plot result for brandenburg gate
    ax.plot_surface(lon, lat, bran_z,
                    color='k',
                    alpha=alpha, lw=line_width)
    # this creates something to make the legend work
    # that I found on stackoverflow
    bran_proxy = plt.Rectangle((0, 0), 1, 1, fc="k")
    # plot result for satellite path
    ax.plot_surface(lon, lat, sat_z,
                    color='m',
                    alpha=alpha, lw=line_width)
    sat_proxy = plt.Rectangle((0, 0), 1, 1, fc="m")
    # plot result for river spree
    ax.plot_surface(lon, lat, spree_z,
                    color='b',
                    alpha=alpha, lw=line_width)
    spree_proxy = plt.Rectangle((0, 0), 1, 1, fc="b")

    axis_pdf_settings(ax,
                      xlabel=r'Longitude [$^\circ$E]',
                      ylabel=r'Latitude [$^\circ$N]',
                      zlabel='Probability')
    ax.legend([bran_proxy, sat_proxy, spree_proxy],
              ['Brandenburg', 'Satellite', 'Spree'],
              fontsize=10)
    fig.savefig('./results/distribs_all.png',
                bbox_inches='tight',
                format="png")
    plt.close()

    # now just the combined distribution
    fig = plt.figure()
    ax_tot = fig.add_subplot(111, projection="3d")

    # 3D version
    ax_tot.plot_surface(lon, lat, total_z,
                        color='y',
                        alpha=alpha, lw=line_width)
    # 2D contours
    ax_tot.contourf(lon, lat, total_z,
                    alpha=alpha,
                    offset=0, cmap='hot')
    # max point
    ax_tot.plot(lon[loc], lat[loc],
                'ko', markersize=2)

    axis_pdf_settings(ax_tot,
                      xlabel=r'Longitude [$^\circ$E]',
                      ylabel=r'Latitude [$^\circ$N]',
                      zlabel='Probability')
    fig.savefig('./results/distrib_total.png',
                bbox_inches='tight',
                format="png")
    plt.close()


def plot_solution(loc):
    """Plot a marker showing the most likely
        location of the analyst onto a
        Google map using the gmplot package.
        """
    # this is a quick way to define the boundaries
    # of the map since I know its in Berlin
    gmap = gmplot.GoogleMapPlotter.from_geocode("Berlin")

    gmap.marker(loc[0], loc[1],
                'r')

    gmap.draw("./results/analyst_location.html")


def plot_teaser_basemap():
    """Context map of the information provided by
        Zalando using the Basemap package.
        """
    map = Basemap(projection='merc', resolution='l',
                  llcrnrlat=SAT_GPS_START[0] - 1., urcrnrlat=SAT_GPS_START[0] + 1.,
                  llcrnrlon=SAT_GPS_START[1] - 1., urcrnrlon=SAT_GPS_END[1] + 1.)

    map.etopo()
    map.drawcountries()
    # draw parallels
    map.drawparallels(np.arange(0., 90., 1.),
                      labels=[1, 1, 0, 1])
    # draw meridians
    map.drawmeridians(np.arange(-180., 180., 1),
                      labels=[1, 1, 0, 1])

    # draw gate position
    xt, yt = map(BRAN_GPS[1], BRAN_GPS[0])
    map.plot(xt, yt,
             'ko', markersize=2)
    # draw satellite path
    map.drawgreatcircle(SAT_GPS_START[1], SAT_GPS_START[0],
                        SAT_GPS_END[1], SAT_GPS_END[0],
                        linewidth=2, color='m')
    # draw river
    spree_lats, spree_lons = zip(*SPREE_GPS)
    xt, yt = map(spree_lons, spree_lats)
    map.plot(xt, yt,
             linewidth=2, color='b')

    plt.show()
    plt.close()


def plot_teaser_coords():
    """Context Google map of information provided by
        Zalando using the gmplot package.
        """
    # this is a quick way to define the boundaries
    # of the map since I know its in Berlin
    gmap = gmplot.GoogleMapPlotter.from_geocode("Berlin")

    # brandenburg gate marker
    gmap.marker(BRAN_GPS[0], BRAN_GPS[1],
                'k')

    # plot over river spree
    spree_lats, spree_lons = zip(*SPREE_GPS)
    gmap.plot(spree_lats, spree_lons,
              'b', edge_width=5.)

    # plot satellite path
    sat_lats = (SAT_GPS_START[0], SAT_GPS_END[0])
    sat_lons = (SAT_GPS_START[1], SAT_GPS_END[1])
    gmap.plot(sat_lats, sat_lons,
              'm', edge_width=5.)

    gmap.draw("./results/teaser_coords.html")


def search_box(lat_range, lon_range, interval):
    """Set up grid for searching for the
        analyst, by first converting to cartesian
        coordinates and then creating the 'box'.
        """
    xy_start = sphere_to_cart(lat_range[0], lon_range[0])
    xy_end = sphere_to_cart(lat_range[1], lon_range[1])

    x_range = np.arange(xy_start[0], xy_end[0], interval)
    y_range = np.arange(xy_start[1], xy_end[1], interval)

    xx, yy = np.meshgrid(x_range, y_range)

    # making a long list so that can call the points easier
    # (thank goodness for stackoverflow for the inspiration
    # for this part!)
    box = []
    for i in range(0, len(xx)):
        for j in range(0, len(yy[i])):
            box.append([xx[i][j], yy[i][j]])

    return xx, yy, box


def search_grid(box, xx_shape):
    """Create grid in spherical geometry from the
        'box' in cartesian coordinates.
        """
    lats = []
    lons = []

    for i, j in box:
        lat, lon = cart_to_sphere(i, j)
        lats.append(lat)
        lons.append(lon)

    lat_grid = np.array(lats).reshape(xx_shape)
    lon_grid = np.array(lons).reshape(xx_shape)

    return lat_grid, lon_grid


def sphere_to_cart(p_lat, p_lon):
    """Converting from spherical geometry
        to cartesian geometry using the
        equations originally provided by Zalando.
        """
    sw_lon = SW_CORNER[1]
    sw_lat = SW_CORNER[0]

    p_x = (p_lon - sw_lon) * np.cos(sw_lat * np.pi / 180.) * 111.323
    p_y = (p_lat - sw_lat) * 111.323

    return p_x, p_y


if __name__ == '__main__':
    main()
