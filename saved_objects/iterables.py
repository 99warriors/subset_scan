import crime_data.constants
import crime_data.crime_data.fxns as crime_data_fxns
import tensor_scan.tensor_scan.fxns as tensor_scan_fxns
import python_utils.python_utils.utils as utils
import itertools
import numpy as np
import pdb
import functools

"""
scratch data_iterable
"""

data_id_iterable = crime_data_fxns.AllHouseBurglaryIterable()
cat_fs = [\
          utils.categorical_f(crime_data_fxns.house_break_f('location_of_entry'), [utils.equals_bin('Door: Front'), utils.equals_bin('Window: Ground'), utils.equals_bin('Door: Rear')]),\
          utils.categorical_f(crime_data_fxns.house_break_f('categorization'), [utils.equals_bin('Professional'), utils.equals_bin('Unprofessional'), utils.equals_bin('Attempt')]),\
]
int_cat_fs = [utils.int_f_from_categorical_f(cat_f) for cat_f in cat_fs]
int_cat_fs_set_iterable = utils.get_powerset_iterator(int_cat_fs)
x_f_iterable = itertools.starmap(utils.series_f, int_cat_fs_set_iterable)
location_f = crime_data_fxns.house_break_f('latlng')
time_f = crime_data_fxns.house_break_f('date_num')
in_pattern_f = crime_data_fxns.in_pattern_f()
scratch_data_iterable = itertools.imap(lambda x_f: map(lambda id: tensor_scan_fxns.datum(id, time_f(id), location_f(id), x_f(id), in_pattern_f(id)), data_id_iterable), x_f_iterable)


"""
scratch pattern_finder_iterable
"""
num_lat_iterable = [5, 10, 15, 20, 30]
num_lng_iterable = num_lat_iterable
num_lat_num_lng_iterable = itertools.izip(num_lat_iterable, num_lng_iterable)
regions_F_iterable = itertools.starmap(utils.latlng_grid_regions_F, num_lat_num_lng_iterable)
pseudocounts_iterable = [0.001]
bin_region_count_F_iterable = itertools.imap(tensor_scan_fxns.bin_region_count_F, pseudocounts_iterable)
raw_joint_distribution_F_iterable = [utils.independent_categorical_joint_distribution_F()]
joint_x_distribution_F_iterable = itertools.imap(tensor_scan_fxns.joint_x_distribution_F, raw_joint_distribution_F_iterable)
background_count_F_iterable = itertools.starmap(tensor_scan_fxns.region_x_independent_tensor_count_F, itertools.product(bin_region_count_F_iterable, joint_x_distribution_F_iterable))
foreground_count_F_iterable = [tensor_scan_fxns.empirical_tensor_count_F()]
optimizer_F_iterable = [utils.iterative_argmax_F(utils.get_initial_subset_x_random(0.5), utils.cycle_through_coord_iterative_step(), 20, 0.001)]
p_value_F_iterable = [tensor_scan_fxns.identity_test_stat_F()]
pattern_test_stat_iterable = [tensor_scan_fxns.pattern_test_stat]
pattern_F_iterable = itertools.starmap(tensor_scan_fxns.pattern_F, itertools.product(background_count_F_iterable, foreground_count_F_iterable, optimizer_F_iterable, pattern_test_stat_iterable, p_value_F_iterable))
pattern_finder_regions_F_iterable = itertools.imap(tensor_scan_fxns.background_and_foreground_regions_F, regions_F_iterable)
num_blocks_iterable = [100]
background_foreground_iterator_F_iterable = [tensor_scan_fxns.many_windows_iterator(num_blocks) for num_blocks in num_blocks_iterable]
raw_pattern_finder_F_iterable = itertools.starmap(tensor_scan_fxns.raw_pattern_finder_F, itertools.product(background_foreground_iterator_F_iterable, pattern_finder_regions_F_iterable, pattern_F_iterable))
scratch_oneshot_ranked_pattern_finder_F_iterable = itertools.imap(tensor_scan_fxns.oneshot_ranked_pattern_finder_F, raw_pattern_finder_F_iterable)
iterative_max_iter = 3
scratch_ss_pattern_finder_F_iterable = itertools.imap(tensor_scan_fxns.most_significant_pattern_finder_F, raw_pattern_finder_F_iterable)
scratch_iterative_ranked_pattern_finder_F_iterable = itertools.imap(functools.partial(tensor_scan_fxns.iterative_ranked_pattern_finder_F, iterative_max_iter), scratch_ss_pattern_finder_F_iterable)

"""
scratch baseline_method_iterable
"""
location_dist_F_iterable = [tensor_scan_fxns.location_dist_F('euclidean')]
time_dist_F_iterable = [tensor_scan_fxns.time_dist_F('cityblock')]
x_dist_F_iterable = [tensor_scan_fxns.x_dist_F('hamming')]
location_weight_iterable = np.linspace(0.2, 0.8, 4)
time_weight_iterable = np.linspace(0.2, 0.8, 4)
x_weight_iterable = np.linspace(0.2, 0.8, 4)
weights_iterable = itertools.starmap((lambda *args: (args[0]/sum(args), (args[1]/sum(args)), (args[2]/sum(args)))), itertools.product(location_weight_iterable, time_weight_iterable, x_weight_iterable))
datum_dist_F_iterable = itertools.starmap(tensor_scan_fxns.distance_F, itertools.imap(utils.flatten, itertools.product(weights_iterable, itertools.product(location_dist_F_iterable, time_dist_F_iterable, x_dist_F_iterable))))
linkage_iterable = ['single']
scratch_agglomerative_pattern_finder_f_iterable = itertools.starmap(tensor_scan_fxns.agglomerative_pattern_finder_F, itertools.product(datum_dist_F_iterable, linkage_iterable))
