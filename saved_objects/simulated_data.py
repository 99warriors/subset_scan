import tensor_scan.tensor_scan.fxns as fxns
import python_utils.python_utils.utils as utils
import itertools
import numpy as np
import functools
import pdb


"""
simulated data for agglomerative method.  distributions for background should be wider
"""
background_agg_N = 300
pattern_agg_N = 300
agg_background_time_f = functools.partial(np.random.uniform, 0.0, 10.0)
agg_pattern_time_f = functools.partial(np.random.uniform, 4.9, 5.1)
agg_background_location_f = functools.partial(np.random.normal, (1.0, 1.0), 10.0)
agg_pattern_location_f = functools.partial(np.random.normal, (-1.0, -1.0), 0.2)
agg_background_x_f = utils.series_f(\
                                    functools.partial(utils.random_categorical, [0.1, 0.1, 0.8]),\
                                    functools.partial(utils.random_categorical, [0.1, 0.8, 0.1]),\
                                    )
agg_pattern_x_f = utils.series_f(\
                                 functools.partial(utils.random_categorical, [0.1, 0.1, 0.8]),\
                                 functools.partial(utils.random_categorical, [0.1, 0.8, 0.1]),\
)

background_agg_data = [fxns.datum(i, agg_background_time_f(), agg_background_location_f(), agg_background_x_f(), 0) for i in xrange(background_agg_N)]
pattern_time_diff_agg_data = [fxns.datum(i, agg_pattern_time_f(), agg_background_location_f(), agg_background_x_f(), 1) for i in xrange(pattern_agg_N)]
pattern_location_diff_agg_data = [fxns.datum(i, agg_background_time_f(), agg_pattern_location_f(), agg_background_x_f(), 1) for i in xrange(pattern_agg_N)]
pattern_x_diff_agg_data = [fxns.datum(i, agg_background_time_f(), agg_background_location_f(), agg_pattern_x_f(), 1) for i in xrange(pattern_agg_N)]


"""
simulated data for subsetscan.  have 2 different time distributions for back/foreground.  should be mostly disjoint.  location_f should be quite concentrated for foreground, so that it's higher at its mode than background.  
"""
background_ss_N = 1900
pattern_ss_N = 100
ss_background_time_f = utils.generator_f(itertools.chain(iter(xrange(0,1000)), iter(xrange(1100,2000))))
ss_pattern_time_f = utils.generator_f(iter(xrange(1000, 1100)))
ss_background_location_f = functools.partial(utils.multivariate_random_uniform, [(0,10),(0,10)])
ss_pattern_location_f = functools.partial(utils.multivariate_random_uniform, [(-10,-5),(-10,-5)])
ss_background_x_f = utils.series_f(\
                                   functools.partial(utils.random_categorical, [0.1]),\
#                                    functools.partial(utils.random_categorical, [0.1, 0.1, 0.8]),\
#                                    functools.partial(utils.random_categorical, [0.1, 0.8, 0.1]),\
)
ss_pattern_x_f = utils.series_f(\
                                functools.partial(utils.random_categorical, [0.1]),\
#                                functools.partial(utils.random_categorical, [0.1, 0.1, 0.8]),\
#                                functools.partial(utils.random_categorical, [0.1, 0.8, 0.1]),\
)

background_ss_data = [fxns.datum(i, ss_background_time_f(), ss_background_location_f(), ss_background_x_f(), 0) for i in xrange(background_ss_N)]
pattern_location_diff_ss_data = [fxns.datum(i, ss_pattern_time_f(), ss_pattern_location_f(), ss_background_x_f(), 1) for i in xrange(pattern_ss_N)]
#pattern_x_diff_ss_data = [fxns.datum(i, ss_pattern_time_f(), ss_background_location_f(), ss_pattern_x_f(), 1) for i in xrange(pattern_ss_N)]
