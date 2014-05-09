import python_utils.python_utils.utils as utils
import numpy as np
import pandas as pd
import itertools
import functools
"""

"""


class tensor(object):

    def get_marginal(self, axis, restriction):
        """
        
        """
        pass


class region_x_independent_tensor(tensor):
    """
    has no knowledge of what labels of axes are, ie what region index i of region mode is
    """
    def __init__(self, _region_counts, x_joint_distribution):
        self._region_counts, self.x_joint_distribution = _region_counts, x_joint_distribution
        self.log_scale = 0.0

    @property
    def shape(self):
        return np.concatenate([np.array([len(self.region_counts)]), np.array(self.x_joint_distribution.shape)])

    def sum(self):
        return self.get_restriction([range(x) for x in self.shape])

    def get_restriction(self, restriction):
        """
        returns actual count, not log
        """
        x_restriction = restriction[1:]
        region_restriction = restriction[0]
        from scipy.misc import logsumexp
        ans = logsumexp([self.region_counts[i] for i in region_restriction]) + self.x_joint_distribution.get_restriction(x_restriction)
        return self.log_scale + ans

    @property
    def region_counts(self):
        return self._region_counts

    def get_marginal(self, axis, restriction):
        """
        - assume the 0-th mode corresponds to regions
        - returns actual counts, not log counts
        - restriction has length equal to the dimension of the tensor, but one of them (the mode to get marginals for) is ignored
        """
        if axis == 0:
            """
            p(x|region) = p(x).  thus, count(x,region) = count(region) * p(x)
            """
            x_restriction = restriction[1:]
            return self.log_scale + self.region_counts + self.x_joint_distribution.get_restriction(x_restriction)
        else:
            old_restriction = restriction[axis]
            axis_length = self.shape[axis]
            ans = np.zeros(axis_length)
            for i in range(axis_length):
                restriction[axis] = [i]
                x_restriction = restriction[1:]
                ans[i] = self.x_joint_distribution.get_restriction(x_restriction)
            restriction[axis] = old_restriction
            from scipy.misc import logsumexp
            region_restriction = restriction[0]
            region_restriction_prob = logsumexp([self.region_counts[i] for i in region_restriction])
            return self.log_scale + ans + region_restriction_prob


class empirical_tensor(tensor):

    def __init__(self, shape, tuples):
        self.d = {}
        self._shape = shape
        for tup in tuples:
            try:
                self.d[tup] = self.d[tup] + 1
            except KeyError:
                self.d[tup] = 1
    
    @property
    def shape(self):
        return self._shape

    def get_restriction(self, restriction):
        count = 0
        for key, val in self.d.iteritems():
            ok = True
            for s, coord in zip(restriction, key):
                if coord not in s:
                    ok = False
                    break
            if ok:
                count += val
        return np.log(count)

    def get_marginal(self, axis, restriction):
        ans = np.zeros(self.shape[axis])
        for tup, count in self.d.iteritems():
            try:
                ans[tup[axis]] += count
            except IndexError:
                print axis, tup
                pdb.set_trace()
        return np.log(ans)

    def sum(self):
        #asdf = np.log(np.sum([count for (key, count) in self.d.iteritems()]))
        #print asdf, 'EMPIRICAL SUM'
        #pdb.set_trace()
        return np.log(np.sum([count for (key, count) in self.d.iteritems()]))


class region_x_independent_tensor_count_F(utils.F):
    """
    returns a tensor object
    """
    def __init__(self, region_count_F, joint_distribution_F):        
        self.region_count_F, self.joint_distribution_F = region_count_F, joint_distribution_F

    def train(self, train_data):
        self.region_count_F.train(train_data)
        self.joint_distribution_F.train(train_data)

#    @utils.timeit_method_decorator()
    def __call__(self, regions, num_cats, test_data):
        region_counts = self.region_count_F(regions, test_data)
        x_joint_distribution = self.joint_distribution_F(num_cats, test_data)
        return region_x_independent_tensor(region_counts, x_joint_distribution)
        

class empirical_tensor_count_F(utils.unsupervised_F):

    def __init__(self):
        pass

    def __call__(self, regions, num_cats, test_data):
        shape = tuple([len(regions)] + num_cats)
        tups = []
        for datum in test_data:
            try:
                tups.append((regions.point_to_region_index(datum.location),) + tuple(datum.x))
            except utils.NoRegionException:
                pass
        return empirical_tensor(shape, tups)


class background_and_foreground_regions_F(utils.unsupervised_F):

    def __init__(self, backing_regions_F):
        self.backing_regions_F = backing_regions_F

    def train(self, training_data):
        self.backing_regions_F.train([datum.location for datum in training_data])

    def __call__(self, background_data, foreground_data):
        all_data = background_data + foreground_data
        return self.backing_regions_F([datum.location for datum in all_data])


class pattern_test_stat(object):

    def __init__(self, B, C):
        self.B, self.C = B, C

    @property
    def x_dims(self):
        return self.B.shape

    def __call__(self, x):
        return self._helper_f(np.exp(self.B.get_restriction(x)), np.exp(self.C.get_restriction(x)))

    def _helper_f(self, B, C):
        try:
            return C * (np.log(C) - np.log(B)) + B - C
        except:
            print B,C
            pdb.set_trace()

#    @utils.timeit_method_decorator()
    def coord_ascent(self, x, i):
        """
        returns the new x_i
        """
        marg_B = np.array([np.exp(val) for val in self.B.get_marginal(i, x)])
        marg_C = np.array([np.exp(val) for val in self.C.get_marginal(i, x)])

        def priority(b,c):
            try:
                return np.log(c) - np.log(b)
            except ValueError:
                print b,c, 'bad'
                pdb.set_trace()
                if utils.is_zero(b) and utils.is_zero(c):
                    return 0.0
                if utils.is_zero(b) and not utils.is_zero(c):
                    return 1000
                if utils.is_zero(c) and not utils.is_zero(b):
                    return -1000
                assert False

        priorities = [-1.0 * priority(b,c) for (b,c) in itertools.izip(marg_B, marg_C)]

        in_order = np.argsort(priorities)

        sorted_marg_B = marg_B[in_order]
        sorted_marg_C = marg_C[in_order]
        sorted_marg_B_cum_sum = np.cumsum(sorted_marg_B)
        sorted_marg_C_cum_sum = np.cumsum(sorted_marg_C)

        vals = [self._helper_f(marg_B_cum_sum, marg_C_cum_sum) for marg_B_cum_sum, marg_C_cum_sum in itertools.izip(sorted_marg_B_cum_sum, sorted_marg_C_cum_sum)]
        
        return in_order[0:(np.argmax(vals)+1)]


class identity_test_stat_F(utils.unsupervised_F):

    def __init__(self):
        pass

    def __call__(self, B, C, test_stat_val):
        return test_stat_val


class pattern_F(utils.F):
    """
    returns the subset of indicies for each mode in pattern
    assumes that sum of output of background_tensor_count_F is 1, so need to normalize that
    """
    def __init__(self, background_tensor_count_F, foreground_tensor_count_F, optimizer_F, get_objective_function, get_p_value):
        self.background_tensor_count_F, self.foreground_tensor_count_F, self.optimizer_F = background_tensor_count_F, foreground_tensor_count_F, optimizer_F
        self.get_objective_function, self.get_p_value = get_objective_function, get_p_value

    def train(self, training_data):
        self.background_tensor_count_F.train(training_data)
        self.foreground_tensor_count_F.train(training_data)

#    @utils.timeit_method_decorator()
#    @caching.default_cache_method_decorator
#    @caching.default_read_method_decorator
#    @caching.default_write_method_decorator
    def __call__(self, regions, num_cats, background_data, foreground_data):

        B = self.background_tensor_count_F(regions, num_cats, background_data)
        C = self.foreground_tensor_count_F(regions, num_cats, foreground_data)
        B.log_scale = C.sum()
        f = self.get_objective_function(B, C)
        opt_subsets = self.optimizer_F(f)
        opt_region_subset = opt_subsets[0]
        opt_x_subsets = opt_subsets[1:]
        test_stat_value = f(opt_subsets)
        p_value = self.get_p_value(B, C, test_stat_value)
        return p_value, opt_region_subset, opt_x_subsets
            

class raw_pattern_finder_F(utils.F):
    """

    """
    def __init__(self, background_foreground_iterator_F, regions_F, pattern_F):
        self.background_foreground_iterator_F, self.regions_F, self.pattern_F = background_foreground_iterator_F, regions_F, pattern_F

    def train(self, training_data):
        self.regions_F.train(training_data)
        self.background_foreground_iterator_F.train(training_data)
        self.pattern_F.train(training_data)

    @utils.timeit_method_decorator()
#    @caching.default_cache_method_decorator
#    @caching.default_read_method_decorator
#    @caching.default_write_method_decorator
    def __call__(self, test_data):
        results = []
        count = 0
        count_total = self.background_foreground_iterator_F.num_blocks * (self.background_foreground_iterator_F.num_blocks - 1)
        for (background_data, foreground_data) in self.background_foreground_iterator_F(test_data):
            if count % 500 == 0:
                print '%d/%d' % (count, count_total)
            count += 1
            regions = self.regions_F(background_data, foreground_data)
            K = len(iter(background_data).next().x)
            num_cats = [np.max([datum.x[k] for datum in background_data + foreground_data]) + 1 for k in range(K)]
            p_value, opt_region_subset, opt_x_subsets = self.pattern_F(regions, num_cats, background_data, foreground_data)

            def x_ok(datum):
                return all([datum.x[i] in opt_x_subset for i, opt_x_subset in enumerate(opt_x_subsets)])

            def region_ok(datum):
                try:
                    return regions.point_to_region_index(datum.location) in opt_region_subset
                except utils.NoRegionException:
                    return False

            pattern_data = [datum for datum in foreground_data if x_ok(datum) and region_ok(datum)]
            results.append([p_value, set(background_data), set(foreground_data), set(pattern_data), utils.simple_region_list([region for i, region in enumerate(regions) if i in opt_region_subset]), opt_x_subsets])

        return results, test_data

    @staticmethod
    def output_to_list_of_plot_pattern_informative_inputs(raw_results, test_data):
        return [(raw_result, test_data) for raw_result in raw_results]


class ranked_pattern_finder_F(utils.F):

    @staticmethod
    def output_to_perf_curve_input(l):
        test_data = iter(l).next()[1]
        last_pos = len(l)
        positions = {datum:last_pos for datum in test_data}
        for pos, (raw_result, test_data) in enumerate(l):
            p_value, background_data, foreground_data, pattern_data, opt_regions, opt_xs = raw_result
            for datum in pattern_data:
                if positions[datum] > pos:
                    positions[datum] = pos

        truth_and_scores = [(datum.in_pattern, (1.0 - (pos/float(len(l))))) for (datum, pos) in positions.iteritems()]
        return [x[0] for x in truth_and_scores], [x[1] for x in truth_and_scores]


class oneshot_ranked_pattern_finder_F(ranked_pattern_finder_F):

    def __init__(self, raw_pattern_finder_F):
        self.raw_pattern_finder_F = raw_pattern_finder_F

    def train(self, train_data):
        self.raw_pattern_finder_F.train(train_data)

    def __call__(self, test_data):
        raw_results, test_data = self.raw_pattern_finder_F(test_data)
        return [(result, test_data) for result in sorted(raw_results, key = lambda x:x[0])]


class iterative_ranked_pattern_finder_F(ranked_pattern_finder_F):

    def __init__(self, max_iter, single_pattern_finder_F):
        self.max_iter, self.single_pattern_finder_F  = max_iter, single_pattern_finder_F

    def train(self, train_data):
        self.single_pattern_finder_F.train(train_data)

    def __call__(self, test_data):
        results = []
        for i in range(self.max_iter):
            print 'iterative iter: %d' % i
            single_result, test_data = self.single_pattern_finder_F(test_data)
            p_value, background_data, foreground_Data, pattern_data, opt_regions, opt_xs = single_result
            results.append((single_result, test_data))
            test_data = list(set(test_data).difference(pattern_data))
            if len(test_data) == 0:
                break
        return results


class single_pattern_finder_F(utils.F):

    @staticmethod
    def output_to_perf_point_input((p_value, background_data, foreground_data, pattern_data, opt_regions, opt_x_subsets), test_data):
        retrieved_set = set([datum.id for datum in pattern_data])
        relevant_set = set([datum.id for datum in test_data if datum.in_pattern])
        total_set = set([datum.id for datum in test_data])
        return retrieved_set, relevant_set, total_set

    @staticmethod
    def output_to_plot_pattern_input((p_value, background_data, foreground_data, pattern_data, opt_regions, opt_x_subsets), test_data):
        return (set(background_data), set(foreground_data), set(pattern_data)), test_data

    @staticmethod
    def output_to_plot_pattern_informative_input((p_value, background_data, foreground_data, pattern_data, opt_regions, opt_x_subsets), test_data):
        """
        identity function
        """
        return (p_value, background_data, foreground_data, pattern_data, opt_regions, opt_x_subsets), test_data


class most_significant_pattern_finder_F(single_pattern_finder_F):

    def __init__(self, raw_pattern_finder_F):
        self.raw_pattern_finder_F = raw_pattern_finder_F

    def train(self, train_data):
        self.raw_pattern_finder_F.train(train_data)

    def __call__(self, test_data):
        raw_results, test_data = self.raw_pattern_finder_F(test_data)
        return max(raw_results, key = lambda x: x[0]), test_data


class datum(object):

    def __init__(self, id, time, location, x, in_pattern, which_pattern = None):
        self.id, self.time, self.location, self.x, self.in_pattern, self.which_pattern = id, time, location, x, in_pattern, which_pattern


class joint_x_distribution_F(utils.F):
    """
    wrapper. given test_data, returns joint distribution of x's
    """
    def __init__(self, joint_distribution_F):
        self.joint_distribution_F = joint_distribution_F

    def train(self, train_data):
        xs = pd.DataFrame([datum.x for datum in train_data])
        self.joint_distribution_F.train(xs)

#    @utils.timeit_method_decorator()
    def __call__(self, num_cats, test_data):
        xs = np.asarray(np.matrix([datum.x for datum in test_data]))
        return self.joint_distribution_F(num_cats, xs)


class KD_region_count_F(utils.unsupervised_F):
    """
    accepts list of datums, extracts the locations from them to feed to a raw kde 
    """
    def __init__(self, kd_helper_F):
        self.kd_helper_F = kd_helper_F

    def __call__(self, regions, test_data):
        locations = [datum.location for datum in train_data]
        self.kd_helper_F.train(locations)
        return [self.kd_helper_F(region.point_rep()) for region in regions]


class bin_region_count_F(utils.unsupervised_F):
    """
    
    """
    def __init__(self, pseudo_count):
        self.pseudo_count = pseudo_count
    
#    @utils.timeit_method_decorator()
    def __call__(self, regions, test_data):
        locations = [datum.location for datum in test_data]
        counts = np.ones(len(regions)) * self.pseudo_count
        for location in locations:
            try:
                counts[regions.point_to_region_index(location)] += 1
            except utils.NoRegionException:
                pass
        return np.log(counts / np.sum(counts))


class location_dist_F(utils.F):

    def __repr__(self):
        return 'loc_dist_F_%s' % self.metric

    def __init__(self, metric):
        self.metric = metric
        self.horse = utils.scaled_dist_mat_F(metric)

    def train(self, train_data):
        train_X = np.matrix([datum.location for datum in train_data])
        self.horse.train(train_X)

    def __call__(self, test_data):
        test_X = np.matrix([datum.location for datum in test_data])
        return self.horse(test_X)


class x_dist_F(utils.F):

    def __repr__(self):
        return 'x_dist_F_%s' % self.metric

    def __init__(self, metric):
        self.metric = metric
        self.horse = utils.scaled_dist_mat_F(metric)

    def train(self, train_data):
        train_X = np.matrix([datum.x for datum in train_data])
        self.horse.train(train_X)

    def __call__(self, test_data):
        test_X = np.matrix([datum.location for datum in test_data])
        return self.horse(test_X)


class time_dist_F(utils.F):

    def __repr__(self):
        return 'time_dist_F_%s' % self.metric

    def __init__(self, metric):
        self.metric = metric
        self.horse = utils.scaled_scalar_dist_mat_F(metric)

    def train(self, train_data):
        train_X = np.array([datum.time for datum in train_data])
        self.horse.train(train_X)

    def __call__(self, test_data):
        test_X = np.array([datum.time for datum in test_data])
        return self.horse(test_X)


class distance_F(utils.F):
    """
    linear combination of spatial, time, and feature distance metrics. assumes features, locations are vectors
    returns condensed matrix (not a scalar)
    """
    def __init__(self, location_weight, time_weight, feature_weight, location_dist_F, time_dist_F, feature_dist_F):
        self.location_dist_F, self.time_dist_F, self.feature_dist_F = location_dist_F, time_dist_F, feature_dist_F
        self.location_weight, self.time_weight, self.feature_weight = location_weight, time_weight, feature_weight

    def __repr__(self):
        return '%.2f*%s+%.2f*%s+%.2f*%s'% (self.location_weight, repr(self.location_dist_F), self.time_weight, repr(self.time_dist_F), self.feature_weight, repr(self.feature_dist_F))

    def train(self, train_data):
        self.location_dist_F.train(train_data)
        self.time_dist_F.train(train_data)
        self.feature_dist_F.train(train_data)

    def __call__(self, test_data):
        return self.location_weight * self.location_dist_F(test_data) + self.time_weight * self.time_dist_F(test_data) + self.feature_weight * self.feature_dist_F(test_data)


class scored_pattern_finder_F(utils.F):

    @staticmethod
    def output_to_perf_curve_input(datum_anomaly_score_tuples):
        return [x[0].in_pattern for x in datum_anomaly_score_tuples], [x[1] for x in datum_anomaly_score_tuples]


class agglomerative_pattern_finder_F(scored_pattern_finder_F):

    def __init__(self, distance_F, method):
        self.distance_F, self.method = distance_F, method

    def __repr__(self):
        return 'agg_clustering_%s_%s' % (repr(self.distance_F), self.method)

    def train(self, train_data):
        self.distance_F.train(train_data)

    @utils.timeit_method_decorator()
    def __call__(self, test_data):
        N = len(test_data)
        dist_mat_list = self.distance_F(test_data)
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import squareform
        dist_mat = squareform(dist_mat_list, force = 'tomatrix')
        Z = hierarchy.linkage(dist_mat_list, method = self.method)
        close_to_bottom = [z for i in xrange(Z.shape[0]) for z in Z[i,0:2] if z < N ]
        assert len(close_to_bottom) == N
        anomaly_scores = np.array(np.argsort([x for x in reversed(close_to_bottom)])) / float(N)
        ans = [(datum, anomaly_score) for (datum, anomaly_score) in itertools.izip(test_data, anomaly_scores)]
        return ans


class cut_in_half_by_time_iterator(utils.unsupervised_F):

    def __init__(self):
        pass

    def __call__(self, data):
        min_time = np.min([datum.time for datum in data])
        max_time = np.max([datum.time for datum in data])
        width = (1 + max_time - min_time) / 2
        med_time = min_time + width
        return iter([\
                     [[datum for datum in data if datum.time < med_time], [datum for datum in data if datum.time >= med_time]],\
                     [[datum for datum in data if datum.time >= med_time], [datum for datum in data if datum.time < med_time]],\
                     ]
                    )


class many_windows_iterator(utils.unsupervised_F):

    def __init__(self, num_blocks):
        self.num_blocks = num_blocks

    def __call__(self, data):
        N = len(data)
        sorted_data = sorted(data, key = lambda datum: datum.time)
        idx_boundaries = [int(x) for x in np.linspace(0, N, self.num_blocks+1)]

        def start_end_idx_to_background_foreground(start_idx, end_idx):
            return sorted_data[0:start_idx] + sorted_data[end_idx:], sorted_data[start_idx:end_idx]

        #return [start_end_idx_to_background_foreground(1000,1100)]
        return itertools.ifilter(lambda (background_data, foreground_data): len(background_data) != 0 and len(foreground_data) != 0 and len(foreground_data) < N / 2, itertools.starmap(start_end_idx_to_background_foreground, itertools.combinations(idx_boundaries,2)))


def plot_pattern((background_data, foreground_data, pattern_data), test_data):
    """
    plot true foreground patterns with x's, background blue, predicted pattern red, others green
    """

    marker_size = 2
    alpha = 0.8

    not_pattern_data = foreground_data.difference(pattern_data)
    true_pattern_data = [datum for datum in foreground_data if datum.in_pattern]
    true_not_pattern_data = [datum for datum in foreground_data if not datum.in_pattern]

    pattern_xys = [utils.latlng_to_xy(*datum.location) for datum in pattern_data]
    not_pattern_xys = [utils.latlng_to_xy(*datum.location) for datum in not_pattern_data]
    true_pattern_xys = [utils.latlng_to_xy(*datum.location) for datum in true_pattern_data]
    true_not_pattern_xys = [utils.latlng_to_xy(*datum.location) for datum in true_not_pattern_data]

    background_xys = [utils.latlng_to_xy(*datum.location) for datum in background_data]

    predicted_fig, predicted_ax = plt.subplots()
    predicted_ax.scatter([xy[0] for xy in background_xys], [xy[1] for xy in background_xys], label = 'background crimes', color = 'black', s = marker_size, alpha = alpha)
    predicted_ax.scatter([xy[0] for xy in pattern_xys], [xy[1] for xy in pattern_xys], label = 'pattern crimes', color = 'green', s = marker_size, alpha = alpha)
    predicted_ax.scatter([xy[0] for xy in not_pattern_xys], [xy[1] for xy in not_pattern_xys], label = 'not pattern crimes', color = 'red', s = marker_size, alpha = alpha)
    predicted_fig.suptitle('predicted')
    predicted_ax.legend()

    true_fig, true_ax = plt.subplots()
    true_ax.scatter([xy[0] for xy in background_xys], [xy[1] for xy in background_xys], label = 'background crimes', color = 'black', s = marker_size, alpha = alpha)
    true_ax.scatter([xy[0] for xy in true_pattern_xys], [xy[1] for xy in true_pattern_xys], label = 'true pattern crimes', color = 'green', s = marker_size, alpha = alpha)
    true_ax.scatter([xy[0] for xy in true_not_pattern_xys], [xy[1] for xy in true_not_pattern_xys], label = 'true not pattern crimes', color = 'red', s = marker_size, alpha = alpha)
    true_fig.suptitle('truth')
    true_ax.legend()

    return (predicted_fig, predicted_ax), (true_fig, true_ax)
    
def plot_pattern_informative((p_value, background_data, foreground_data, pattern_data, opt_regions, opt_x_subsets), test_data):
    (predicted_fig, predicted_ax), (true_fig, true_ax) = plot_pattern((background_data, foreground_data, pattern_data), test_data)
    for region in opt_regions:
        region.plot(predicted_ax)
        region.plot(true_ax)
    min_time = min([datum.time for datum in pattern_data])
    max_time = max([datum.time for datum in pattern_data])
    predicted_fig.suptitle('test_stat: %.2f time_range: (%.2f,%.2f)' % (p_value, min_time, max_time))
    true_fig.suptitle('opt_xs: %s' % str(opt_x_subsets))
    return (predicted_fig, predicted_ax), (true_fig, true_ax)


pattern_window_color, pattern_color, background_color = 'red', 'cyan', 'black'


def plot_pattern_info(pattern_id, data):
    # find the window corresponding to pattern
    sorted_data = sorted(data, key = lambda datum: datum.time)
    pattern_idxs = [i for (i, datum) in enumerate(sorted_data) if datum.which_pattern == pattern_id]
    pattern_start_idx, pattern_end_idx = min(pattern_idxs), max(pattern_idxs)
    pattern_window_data = sorted_data[pattern_start_idx:pattern_end_idx + 1]
    background_data = sorted_data[:pattern_start_idx] + sorted_data[pattern_end_idx + 1:]
    pattern_window_background_data = [datum for datum in pattern_window_data if datum.which_pattern != pattern_id]
    pattern_data = [datum for datum in pattern_window_data if datum.which_pattern == pattern_id]

    # plot spatial distribution of points inside pattern_window, with pattern points colored differently, as well as distribution in background window
    pattern_window_background_xys = [utils.latlng_to_xy(*datum.location) for datum in pattern_window_background_data]
    pattern_xys = [utils.latlng_to_xy(*datum.location) for datum in pattern_data]
    background_xys = [utils.latlng_to_xy(*datum.location) for datum in background_data]

    marker_size = 3
    alpha = 0.8

    spatial_fig, spatial_ax = plt.subplots()
    spatial_ax.scatter([xy[0] for xy in background_xys], [xy[1] for xy in background_xys], s = marker_size, alpha = alpha, color = background_color)
    spatial_ax.scatter([xy[0] for xy in pattern_window_background_xys], [xy[1] for xy in pattern_window_background_xys], s = marker_size, alpha = alpha, color = pattern_window_color)
    spatial_ax.scatter([xy[0] for xy in pattern_xys], [xy[1] for xy in pattern_xys], s = marker_size, alpha = alpha, color = pattern_color)



    pattern_start_time, pattern_end_time = pattern_data[0].time, pattern_data[-1].time
    spatial_ax.set_title('start: %.2f, end: %.2f, len/window: %d/%d' % (pattern_start_time, pattern_end_time, len(pattern_data), len(pattern_window_background_data)))
    """
    hope that density of pattern_window is higher than that of background in region where the pattern is
    """

    # for each feature, plot distribution of features indie pattern_window, with pattern counts overload differently, as well as background distribution
    feature_figs = []
    background_feature_counts = get_data_feature_counts(background_data)
    pattern_window_feature_counts = get_data_feature_counts(pattern_window_data)
    pattern_feature_counts = get_data_feature_counts(pattern_data)
    num_features = len(iter(data).next().x)
    num_cats = [np.max([datum.x[k] for datum in data]) + 1 for k in range(num_features)]
    for background_feature_count, pattern_window_feature_count, pattern_feature_count, num_cat in itertools.izip(background_feature_counts, pattern_window_feature_counts, pattern_feature_counts, num_cats):
        feature_fig = plt.figure()
        feature_fig.suptitle(background_feature_count.name)
        non_pattern_ax = feature_fig.add_subplot(2,1,1)
        pattern_ax = feature_fig.add_subplot(2,1,2)
        expanded_background_feature_count = (pd.Series(np.zeros(num_cat)) + background_feature_count).fillna(0) / float(background_feature_count.sum())
        expanded_pattern_window_feature_count = (pd.Series(np.zeros(num_cat)) + pattern_window_feature_count).fillna(0) / float(pattern_window_feature_count.sum())
        expanded_pattern_feature_count = (pd.Series(np.zeros(num_cat)) + pattern_feature_count).fillna(0) / float(pattern_feature_count.sum())
        utils.plot_bar_chart(non_pattern_ax, map(str, range(num_cat)), expanded_pattern_window_feature_count, label = 'pattern_window', alpha = 0.5, color = pattern_window_color, offset = 0.4, width = 0.35)
        utils.plot_bar_chart(non_pattern_ax, map(str, range(num_cat)), expanded_background_feature_count, label = 'background', alpha = 0.5, color = background_color, offset = 0.0, width = 0.35)
        non_pattern_ax.set_title('sizes: %d/%d' % (len(pattern_window_data), len(background_data)))
        non_pattern_ax.set_ylim((0,1))
        non_pattern_ax.legend()
        utils.plot_bar_chart(pattern_ax, map(str, range(num_cat)), expanded_pattern_feature_count, label = 'pattern', alpha = 0.5, color = pattern_color, offset = 0.4, width = 0.35)
        pattern_ax.set_title('size: %d' % len(pattern_data))
        pattern_ax.set_ylim((0,1))
        pattern_ax.legend()
        feature_figs.append(feature_fig)
    return spatial_fig, feature_figs
