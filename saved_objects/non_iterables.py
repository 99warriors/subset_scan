import crime_data.constants
import pdb
import tensor_scan.tensor_scan.fxns as tensor_scan_fxns
import crime_data.crime_data.fxns as crime_data_fxns
import python_utils.python_utils.utils as utils
import itertools
import numpy as np



"""
scratch data
"""

location_f = crime_data_fxns.house_break_f('latlng')
year_f = crime_data_fxns.house_break_f('year')
data_id_iterable = list(itertools.ifilter(lambda id: year_f(id) >= 2003 and year_f(id) <= 2005 and location_f(id) in utils.latlng_grid_region(crime_data.constants.cambridge_min_lat, crime_data.constants.cambridge_max_lat, crime_data.constants.cambridge_min_lng, crime_data.constants.cambridge_max_lng), crime_data_fxns.AllHouseBurglaryIterable()))
#data_id_iterable = list(itertools.ifilter(lambda id: location_f(id) in utils.latlng_grid_region(crime_data.constants.cambridge_min_lat, crime_data.constants.cambridge_max_lat, crime_data.constants.cambridge_min_lng, crime_data.constants.cambridge_max_lng), crime_data_fxns.AllHouseBurglaryIterable()))
cat_fs = [\
          utils.categorical_f(crime_data_fxns.house_break_f('location_of_entry'), [utils.equals_bin('Door: Front'), utils.equals_bin('Window: Ground'), utils.equals_bin('Door: Rear')]),\
          utils.categorical_f(crime_data_fxns.house_break_f('means_of_entry'), [utils.equals_bin('Pried'), utils.equals_bin('Unlocked'), utils.equals_bin('Shoved/Forced'), utils.equals_bin('Broke')]),\
#          utils.categorical_f(crime_data_fxns.house_break_f('categorization'), [utils.equals_bin('Professional'), utils.equals_bin('Unprofessional'), utils.equals_bin('Attempt')]),\
]
int_cat_fs = [utils.int_f_from_categorical_f(cat_f) for cat_f in cat_fs]
x_f = utils.series_f(*int_cat_fs)
#x_f = utils.series_f(utils.hard_code_f(0))
time_f = crime_data_fxns.house_break_f('date_num')
in_pattern_f = crime_data_fxns.in_pattern_f()
pattern_f = crime_data_fxns.house_break_f('pattern')
scratch_data = [tensor_scan_fxns.datum(id, time_f(id), location_f(id), x_f(id), in_pattern_f(id), pattern_f(id)) for id in data_id_iterable]

"""
scratch pattern_finder
"""
lat_min, lat_max, lng_min, lng_max = crime_data.constants.cambridge_min_lat, crime_data.constants.cambridge_max_lat, crime_data.constants.cambridge_min_lng, crime_data.constants.cambridge_max_lng
num_lat, num_lng = 16, 16
regions_F = utils.latlng_grid_regions_F(num_lat, num_lng)
background_count_F = tensor_scan_fxns.region_x_independent_tensor_count_F(tensor_scan_fxns.bin_region_count_F(0.00001), tensor_scan_fxns.joint_x_distribution_F(utils.independent_categorical_joint_distribution_F()))
foreground_count_F = tensor_scan_fxns.empirical_tensor_count_F()
optimizer_F = utils.iterative_argmax_F(utils.get_initial_subset_x_random(1.0), utils.cycle_through_coord_iterative_step(), 10, 0.001)
p_value_F = tensor_scan_fxns.identity_test_stat_F()
pattern_F = tensor_scan_fxns.pattern_F(background_count_F, foreground_count_F, optimizer_F, tensor_scan_fxns.pattern_test_stat, p_value_F)
pattern_finder_regions_F = tensor_scan_fxns.background_and_foreground_regions_F(regions_F)
num_windows = 20
raw_pattern_finder_F = tensor_scan_fxns.raw_pattern_finder_F(tensor_scan_fxns.many_windows_iterator(num_windows), pattern_finder_regions_F, pattern_F)
scratch_oneshot_ranked_pattern_finder_F = tensor_scan_fxns.oneshot_ranked_pattern_finder_F(raw_pattern_finder_F)
scratch_pattern_finder_F = tensor_scan_fxns.most_significant_pattern_finder_F(raw_pattern_finder_F)
iterative_max_iter = 3
scratch_iterative_ranked_pattern_finder_F = tensor_scan_fxns.iterative_ranked_pattern_finder_F(iterative_max_iter, scratch_pattern_finder_F)

"""
scratch baseline_method
"""
location_dist_F = tensor_scan_fxns.location_dist_F('euclidean')
time_dist_F = tensor_scan_fxns.time_dist_F('cityblock')
x_dist_F = tensor_scan_fxns.x_dist_F('hamming')
location_weight = 0.05
time_weight = 0.9 
x_weight = 1.0 - (location_weight + time_weight)
datum_dist_F = tensor_scan_fxns.distance_F(location_weight, time_weight, x_weight, location_dist_F, time_dist_F, x_dist_F)
linkage = 'single'
scratch_agglomerative_pattern_finder_F = tensor_scan_fxns.agglomerative_pattern_finder_F(datum_dist_F, linkage)
