zalando
=======

This is my attempt at solving the teaser posed for the Data Scientist [position](https://tech.zalando.com/jobs/data/65946-senior-data-scientist-m-f/?gh_jid=65946) at Zalando's Fashion Insight Centre.

Zalando's next top Analyst
--------------------------

The Zalando Data Intelligence Team is searching for a new top analyst. We already know of an excellent candidate with top analytical and programming skills. Unfortunately, we don't know her exact whereabouts but we only have some vague information where she might be. Can you tell us where to best send our recruiters and plot an easy to read map of your solution for them?

This is what we could extract from independent sources:

  * The candidate is likely to be close to the river Spree. The probability at any point is given by a Gaussian function of its shortest distance to the river. The function peaks at zero and has 95% of its total integral within +/-2730m
  *  A probability distribution centered around the Brandenburg Gate also informs us of the candidate's location. The distribution’s radial profile is log-normal with a mean of 4700m and a mode of 3877m in every direction.
  * A satellite offers further information: with 95% probability she is located within 2400 m distance of the satellite’s path (assuming a normal probability distribution)
  * Please make use of the additional information in the [file](http://bit.ly/19fdgVa). If you send us the solution with both code and diagram along with your application, it will help us better assess your skills. However, the task is not mandatory.


Coordinates
-----------

`earth_radius = 6371 #units in km`

`brandenburg_gate_gps = [52.516288, 13.377689]`

_##Satellite path is a great circle path between coordinates:_

`satellite_path_start = [52.590117, 13.39915] ; satellite_path_end = [52.437385, 13.553989]`

_##River Spree can be approximated as piecewise linear between the following coordinates:_
`spree_coords = [[52.529198, 13.274099], [52.531835, 13.29234], [52.522116, 13.298541], [52.520569, 13.317349],
                [52.524877, 13.322434], [52.522788, 13.329], [52.517056, 13.332075], [52.522514, 13.340743],
                [52.517239, 13.356665], [52.523063, 13.372158], [52.519198, 13.379453], [52.522462, 13.392328],
                [52.520921, 13.399703], [52.515333, 13.406054], [52.514863, 13.416354], [52.506034, 13.435923],
                [52.496473, 13.461587], [52.487641, 13.483216], [52.488739, 13.491456], [52.464011, 13.503386]]`

_##You can (but don’t have to) use following simple projection for getting GPS coordinates into an orthogonal coordinate system. The projection is reasonably accurate for the Berlin area. Result is an XY coordinate system with the origin (0,0) at the South-West corner of the area we are interested in. The X axis corresponds to East-West and is given in kilometres. The Y axis corresponds to North-South and is also given in kilometres._

_##South-west corner of the area we are interested in:_

`SW_lat = 52.464011 ; SW_lon = 13.274099`

_##The x and y coordinates of a GPS coordinate P with (P_lat, P_lon) can be calculated using:_

`P_x = (P_lon − SW_lon) ∗ cos(SW_lat * pi / 180) ∗ 111.323 ; P_y = (P_lat − SW_lat) ∗ 111.323`


Solution
--------
