zalando
=======

This is my attempt at solving the teaser posed for the Data Scientist [position](https://tech.zalando.com/jobs/data/65946-senior-data-scientist-m-f/?gh_jid=65946) at Zalando's Fashion Insight Centre.

Zalando's next top analyst
--------------------------

The Zalando Data Intelligence Team is searching for a new top analyst. We already know of an excellent candidate with top analytical and programming skills. Unfortunately, we don't know her exact whereabouts but we only have some vague information where she might be. Can you tell us where to best send our recruiters and plot an easy to read map of your solution for them?

This is what we could extract from independent sources:

* The candidate is likely to be close to the river Spree. The probability at any point is given by a Gaussian function of its shortest distance to the river. The function peaks at zero and has 95% of its total integral within +/-2730m
*  A probability distribution centered around the Brandenburg Gate also informs us of the candidate's location. The distribution’s radial profile is log-normal with a mean of 4700m and a mode of 3877m in every direction.
* A satellite offers further information: with 95% probability she is located within 2400 m distance of the satellite’s path (assuming a normal probability distribution)
* Please make use of the additional information in the [file](http://bit.ly/19fdgVa). If you send us the solution with both code and diagram along with your application, it will help us better assess your skills. However, the task is not mandatory.


Coordinates
-----------

Radius of Earth is defined as:

    earth_radius = 6371 #units in km

The GPS coordinates of the Brandenburg Gate are:

    brandenburg_gate_gps = (52.516288, 13.377689)

Satellite path is a great circle path between coordinates:

    satellite_path_start = (52.590117, 13.39915)
    satellite_path_end = (52.437385, 13.553989)

River Spree can be approximated as piecewise linear between the following coordinates:

    spree_coords = [(52.529198, 13.274099), (52.531835, 13.292340), (52.522116, 13.298541), (52.520569, 13.317349), (52.524877, 13.322434), (52.522788, 13.329000), (52.517056, 13.332075), (52.522514, 13.340743), (52.517239, 13.356665), (52.523063, 13.372158), (52.519198, 13.379453), (52.522462, 13.392328), (52.520921, 13.399703), (52.515333, 13.406054), (52.514863, 13.416354), (52.506034, 13.435923), (52.496473, 13.461587), (52.487641, 13.483216), (52.488739, 13.491456), (52.464011, 13.503386)]

You can (but don’t have to) use following simple projection for getting GPS coordinates into an orthogonal coordinate system. The projection is reasonably accurate for the Berlin area. Result is an XY coordinate system with the origin (0,0) at the South-West corner of the area we are interested in. The X axis corresponds to East-West and is given in kilometres. The Y axis corresponds to North-South and is also given in kilometres.

South-west corner of the area we are interested in:

    SW_lat = 52.464011
    SW_lon = 13.274099

The x and y coordinates of a GPS coordinate P with _(P_lat, P_lon)_ can be calculated using:

    P_x = (P_lon − SW_lon) ∗ cos(SW_lat * pi / 180) ∗ 111.323
    P_y = (P_lat − SW_lat) ∗ 111.323


Solution
========

Set up
--------
I will use Python and its many fabulous packages to attempt to solve this teaser. In particular I'll be working with an [Anaconda](https://www.continuum.io/why-anaconda) installation of Python 2.7 with Mac OSX 10.10.5. It has also been tested in Red Hat Enterprise Linux 6.7. All code is in `zalando_solution.py` unless otherwise stated below. You can run it yourself  if you have the right dependencies installed (see below):

    import zalando_solution
    zalando_solution.main()

External dependencies
--------------------------------

    gmplot 
    matplotlib 
    \-patheffects 
    \-pyplot 
    mpl_toolkits 
    \-mplot3d 
    numpy 
    scipy 
    \-stats 
    shapely 
    \-geometry 

Method
----------
Firstly I decided to take a quick look at the coordinates just to get a sense of what was going on. The only package for Earth maps I've used in the past is Basemap so that was my starting point. For thermospheric modelling I plot latitude and longitude all the time so I kept that system rather than converting. You will find this defined as `plot_teaser_basemap()`, however its commented out since it turns out that Basemap sucks for really zoomed in areas unless you have street map etc files! 

So I investigated another way to do it, and after some googling I found [``gmplot``](https://github.com/vgm64/gmplot.git) a good option since it allows plotting straight onto a Google map with Python. See `plot_teaser_coords()`, which outputs _teaser_coords.html_ in the _results_ folder.

![Google map of info provided for teaser](http://i.imgur.com/UEgKoN8.jpg)

In this image, the Brandenburg Gate location is in black, the satellite path is in magenta, and the river is in blue. This colour convention will be continued for the rest of the analysis. Note that since no information is given about the satellite at all (its altitude, whether its in e.g., LEO or GEO orbit, etc), I had to assume its a straight line path. Over such a short distance this is probably pretty accurate anyway.

----------

Next up I had a look at what information is given in more detail regarding the probability distributions.

The Brandenburg Gate source has log-normal distribution with mean mean of 4.7km and a mode of 3.877km. 
Now from maths we know that for a [log normal distribution](https://en.wikipedia.org/wiki/Log-normal_distribution),

    mean = np.exp(mu - sigma**2)
    mode = np.exp(mu - sigma**2)

So some simple rearranging (I actually did this by hand for fun!), we can define,

    sigma = np.sqrt((2./3.) * (np.log(mean) - np.log(mode)))
    mu = ((2.*np.log(mean)) + np.log(mode)) / 3.

You'll find all of this in the `log_normal()` function.

Now the satellite and Spree sources are both [normal distributions](https://en.wikipedia.org/wiki/Normal_distribution), with 95% probability of location within 2.4km of the satellite path and 2.73km within the river path. 

That means we can use that lovely number 1.96 (the joys of 2sigma) for calculating the mean, with `mu = 0.` here. You'll see in the `normal()` function that I didn't just write the number though ;)

As a sanity check I plotted the results using `plot_distribs()`, which will output _norms.png_ in the _results_ folder. Note this really isn't needed for the solution, and has been commented out of the main code. 

![Quick view of all the distributions](https://i.imgur.com/N66DbQ6.png)

----------

Now we need to actually get the proper distributions within a search grid over Berlin. I first converted all the GPS coordinates to cartesian since some equations for that purpose were provided. See the `sphere_to_cart()` function for converting from spherical to cartesian, and also `cart_to_sphere()` for converting back to spherical (will need that later to get the GPS location of the analyst!).

Then I defined a search grid. I used the boundaries of the river and satellite path for this purpose. I used equations from (the really useful) [_Movable Type_](http://www.movable-type.co.uk/scripts/latlong.html) to calculate LAT_RANGE and LON_RANGE from the following conditions:

* Minimum longitude is 2.73km west of the minimum Spree longitude.
* Minimum latitude is 2.4km south of the minimum satellite latitude.
* Maximum longitude is 2.4km east of the maximum satellite longitude.
* Maximum latitude is 2.4km north of the maximum satellite latitude.

These ranges were converted to cartesian and a `box` was created with resolution 0.1km.

----------

Now to calculate the probability distribution functions for each of the sources. I came across the [`shapely`](https://github.com/Toblerity/Shapely) package on Stack Overflow for analysis in cartesian geometry, which I used for calculating the distance from points to points/lines. However, its worth noting other ways to this. See `distance.py` for some functions to use maths to calculate the same thing. Note the `haversine()` function there (which I also found on Stack Overflow) can be used to do this in spherical geometry instead!

See the `pdf_point()` function for the calculations, where the following were used:

    brandenburg_point = Point(bran_coords)
    satellite_point = LineString([(sat_coords_start), (sat_coords_end)])
    spree_point = LineString(spree_coords)

I made another 'sanity check' figure with `plot_contour_distribs()`, resulting in _distribs_2d.png_ in the _results_ folder again. As before, this has been commented out of the main code as its not needed for the solution.

![2D plots of distributions](https://i.imgur.com/enZl79s.png)


----------

I then converted back to spherical geometry (see `lat_grid` and `lon_grid`) to find the most likely GPS location of the analyst.

In operational space weather forecasting, the easiest way to combine multiple forecasts to create an ensemble forecast is to use a simple linear combination. See, e.g., [Guerra et al](http://arxiv.org/abs/1504.04571), where several flare forecasts were weighted and linearly combined. Obviously I don't have historical forecast performance here so to get the total probability I simply added them all:

    total_probability = brandenburg_probability + spree_probability + satellite_probability

I created some plots of the results with `plot_pdf()`; see _distribs_all.png_ and _distrib_total.png_ in the _results_ folder.

![3D view of the distributions](https://i.imgur.com/3erHULb.png)

![3D view of the total probability distribution](https://i.imgur.com/kmgIkpm.png)


Result
---------
It is clear from the 3D view of the total probability distribution that there are multiple peaks. However, the teaser asked for the best location to send the recruiters, so to calculate this I simply found the maximum of the distribution. So (*drumroll...*), you will most likely find the analyst here:

    52.49128610520737, 13.49485721977875

I created a map of the location with `plot_solution()`, outputting _analyst_location.html_ in the _results_ folder.

![Map of analyst location](https://i.imgur.com/g1KhvmN.jpg)

The recruiters can use this Google map link to get directions:

https://www.google.com/maps/?q=52.49128610520737,13.49485721977875

I have also saved the result to _location.txt_ in the _results_ folder in case its forgotten and needed at a later point.

Its worth noting that the location looks to be some kind of factory, which is a bit odd, so it might be worth investigating the other peaks in the distribution!