import python_utils.python_utils.utils as utils
import numpy as np
import pandas as pd
import itertools
import functools
import python_utils.python_utils.caching as caching
import pdb
import matplotlib.pyplot as plt
import python_utils.python_utils.exceptions as exceptions
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

        def ok(_tup):
            for i, t in zip(range(len(restriction)), _tup):
                if i != axis:
                    if t not in restriction[i]:
                        return False
            return True

        ans = np.zeros(self.shape[axis])
        for tup, count in self.d.iteritems():
            try:
                if ok(tup):
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
                for idx in regions.point_to_region_indicies(datum.location):
                    tups.append((idx,) + tuple(datum.x))
            except utils.NoRegionException:
                pass
        return empirical_tensor(shape, tups)


class background_and_foreground_regions_F(utils.unsupervised_F):

    def get_cache_self_attr_repr(self):
        return self.backing_regions_F.get_cache_self_repr()

    def __init__(self, backing_regions_F):
        self.backing_regions_F = backing_regions_F

    def train(self, training_data):
        self.backing_regions_F.train([datum.location for datum in training_data])

    def __call__(self, background_data, foreground_data):
        all_data = background_data + foreground_data
        return self.backing_regions_F([datum.location for datum in all_data])


def print_data_cat_counts(data):
    d = pd.DataFrame({datum.id:datum.x for datum in data})
    from IPython.display import display_html
    for name, row in d.T.iteritems():
        print name
        display_html(utils.pd.DataFrame(row.value_counts()).to_html(), raw=True)


class pattern_test_stat(object):

    def __init__(self, B, C, pseudo_count = None):
        self.B, self.C = B, C
        if pseudo_count != None:
            self.pseudo_count = pseudo_count

    @property
    def x_dims(self):
        return self.B.shape

    def __call__(self, x):
        #print np.exp(self.B.get_restriction(x)), np.exp(self.C.get_restriction(x))
        return self._helper_f(np.exp(self.B.get_restriction(x)), np.exp(self.C.get_restriction(x)))

    def _helper_f(self, B, C):
        try:
            if C < B:
                return 0.0
            else:
#                return C * (np.log(C) - np.log(B)) + B - C
                try:
                    if B < self.pseudo_count:
                        B = self.pseudo_count
                except AttributeError:
                    pass
                return C * (np.log(C) - np.log(B)) + B - C
        except:
            print B,C
            pdb.set_trace()

#    @utils.timeit_method_decorator()
    def coord_ascent(self, _x, _i, constraints):
        """
        returns the new x_i
        here is where asusmption that constraints are box constraints is made (duck typing)
        """

        def unconstrained_coord_ascent_brute(x, i):
            
            def subset_to_test_stat(subset):
                _marg_B = np.array([np.exp(val) for val in self.B.get_marginal(i, x)])
                _marg_C = np.array([np.exp(val) for val in self.C.get_marginal(i, x)])
                #print _marg_B
                #print _marg_C
                B_val = sum([b_val for (b_i, b_val) in enumerate(_marg_B) if b_i in subset])
                C_val = sum([c_val for (c_i, c_val) in enumerate(_marg_C) if c_i in subset])

                return self._helper_f(B_val, C_val)
 #           return [(set(_subset), subset_to_test_stat(_subset)) for _subset in utils.get_powerset_iterator(range(self.B.shape[i]))]
            return max([(set(_subset), subset_to_test_stat(_subset)) for _subset in utils.get_powerset_iterator(range(self.B.shape[i]))], key = lambda x: x[1])[0]


        def unconstrained_coord_ascent(x, i):

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

            max_idx = np.argmax(vals)

            max_val = vals[max_idx]

            ans = set(in_order[0:(max_idx+1)])
            #assert ans == unconstrained_coord_ascent_brute(x, i)
            #if ans != unconstrained_coord_ascent_brute(x, i):
            #    print ans, unconstrained_coord_ascent_brute(x, i)

            if self(x) > max_val + .0001:
                #pdb.set_trace()
                return x[i]

            return ans

        def constrained_coord_ascent(x, i, allowable_subsets):
            _x_copy = x[:]
            def with_coord(x_copy, j, coord):
                x_copy[j] = coord
                return x_copy
            vals = [self(with_coord(_x_copy, i, subset)) for subset in allowable_subsets]
            temp = filter(lambda x: not np.isnan(x[0]), zip(vals, allowable_subsets))
            if len(temp) != 0:
                return max(temp, key = lambda y: y[0])[1]
            else:
                import random
                try:
                    return random.sample(allowable_subsets, 1)[0]
                except:
                    pdb.set_trace()

        if constraints[_i] == None:
            return unconstrained_coord_ascent(_x, _i)
        else:
            return constrained_coord_ascent(_x, _i, constraints[_i])

class identity_test_stat_F(utils.unsupervised_F):

    def __init__(self):
        pass

    def __call__(self, B, C, test_stat_val):
        return test_stat_val


class normalized_test_stat_F(utils.unsupervised_F):

    def __init__(self):
        pass

    def __call__(self, B, C, test_stat_val):
        return test_stat_val / np.exp(C.sum())



class pattern_F(utils.F):
    """
    returns the subset of indicies for each mode in pattern
    assumes that sum of output of background_tensor_count_F is 1, so need to normalize that
    """

    def get_cache_self_attr_repr(self):
        return '%d_%.3f' % (self.optimizer_F.num, self.get_objective_function.keywords['pseudo_count'])

    def __init__(self, background_tensor_count_F, foreground_tensor_count_F, optimizer_F, get_objective_function, get_p_value):
        self.background_tensor_count_F, self.foreground_tensor_count_F, self.optimizer_F = background_tensor_count_F, foreground_tensor_count_F, optimizer_F
        self.get_objective_function, self.get_p_value = get_objective_function, get_p_value

    def train(self, training_data):
        self.background_tensor_count_F.train(training_data)
        self.foreground_tensor_count_F.train(training_data)

    def get_max_try(self, informative):
        return self.optimizer_F.get_max_try(informative)

    def get_avg_steps(self, informative):
        return self.optimizer_F.get_avg_steps(informative)

#    @utils.timeit_method_decorator()
#    @caching.default_cache_method_decorator
#    @caching.default_read_method_decorator
#    @caching.default_write_method_decorator
    def call_informative(self, regions, num_cats, background_data, foreground_data, allowable_region_subsets = None, allowable_x_subsets = None):

        if allowable_x_subsets == None:
            allowable_x_subsets = [None for i in xrange(len(num_cats))]

        allowable_subsets = [allowable_region_subsets] + allowable_x_subsets

        B = self.background_tensor_count_F(regions, num_cats, background_data)
        C = self.foreground_tensor_count_F(regions, num_cats, foreground_data)
        B.log_scale = C.sum()
        f = self.get_objective_function(B=B, C=C)
        uninformative, informative = self.optimizer_F.uninformative_informative(f, allowable_subsets)
        opt_subsets, test_stat_value = uninformative
        opt_region_subset = opt_subsets[0]
        opt_x_subsets = opt_subsets[1:]
        #test_stat_value = f(opt_subsets)
        p_value = self.get_p_value(B, C, test_stat_value)
        return (p_value, opt_region_subset, opt_x_subsets), informative
            

class data_F(utils.F):

    def get_cache_arg_repr(self, data):
        return 'datalen_%d_checksum_%d' % (len(data), sum([datum.id for datum in data]))

class raw_pattern_finder_F(data_F, utils.F):
    """

    """
    
    def get_cache_self_attr_repr(self):
        return '%s_%s_%s_%s' % (self.background_foreground_iterator_F.get_cache_self_attr_repr(), self.regions_F.get_cache_self_attr_repr(), self.pattern_F.get_cache_self_attr_repr(), self.region_subset_F.get_cache_self_attr_repr())

    def get_max_tries(self, informatives):
        return [self.pattern_F.get_max_try(informative) for informative in informatives]

    def get_avg_steps(self, informatives):
        return [self.pattern_F.get_avg_steps(informative) for informative in informatives]

    def __init__(self, background_foreground_iterator_F, regions_F, pattern_F, region_subset_F = None):
        self.background_foreground_iterator_F, self.regions_F, self.pattern_F, self.region_subset_F = background_foreground_iterator_F, regions_F, pattern_F, region_subset_F

    def train(self, training_data):
        self.regions_F.train(training_data)
        self.background_foreground_iterator_F.train(training_data)
        self.pattern_F.train(training_data)

    @utils.timeit_method_decorator()
#    @caching.default_cache_method_decorator
#    @caching.default_read_method_decorator
#    @utils.raise_exception_method_decorator(exceptions.TooLazyToComputeException)
#    @caching.default_write_method_decorator
    def call_informative(self, test_data):
        results = []
        informative_results = []
        count = 0
#        count_total = self.background_foreground_iterator_F.num_blocks * (self.background_foreground_iterator_F.num_blocks - 1)
        for (background_data, foreground_data) in self.background_foreground_iterator_F(test_data):
            if count % 1 == 0:
                print count
                import sys
                sys.stdout.flush()
            count += 1
            regions = self.regions_F(background_data, foreground_data)
            K = len(iter(background_data).next().x)
            num_cats = [np.max([datum.x[k] for datum in background_data + foreground_data]) + 1 for k in range(K)]
            if not self.region_subset_F == None:
                latlngs = [datum.location for datum in itertools.chain(background_data, foreground_data)]
                allowable_region_subsets = self.region_subset_F(regions, latlngs)
            else:
                if regions.is_constrained:
                    allowable_region_subsets = regions.get_allowable_subsets()
                else:
                    allowable_region_subsets = None
            
            pattern_F_uninformative_results, pattern_F_informative_results = self.pattern_F.uninformative_informative(regions, num_cats, background_data, foreground_data, allowable_region_subsets)
            p_value, opt_region_subset, opt_x_subsets = pattern_F_uninformative_results
            informative_results.append(pattern_F_informative_results)

            def x_ok(datum):
                return all([datum.x[i] in opt_x_subset for i, opt_x_subset in enumerate(opt_x_subsets)])

            def region_ok(datum):
                try:
                    for region_idx in regions.point_to_region_indicies(datum.location):
                        if region_idx in opt_region_subset:
                            return True
                    return False
                    #return regions.point_to_region_index(datum.location) in opt_region_subset
                except utils.NoRegionException:
                    return False

            pattern_data = [datum for datum in foreground_data if x_ok(datum) and region_ok(datum)]
            results.append([p_value, set(background_data), set(foreground_data), set(pattern_data), utils.simple_region_list([region for i, region in enumerate(regions) if i in opt_region_subset]), opt_x_subsets])


        return (results, test_data), informative_results

    @staticmethod
    def output_to_list_of_plot_pattern_informative_inputs(raw_results, test_data):
        return [(raw_result, test_data) for raw_result in raw_results]


print 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF'
gg11 = 34
pdb.set_trace()


def ranked_pattern_finder_F_output_to_perf_curve_input(l):
    test_data = iter(l).next()[1]
    last_pos = len(l)
    positions = {datum.id:last_pos for datum in test_data}
    truths = {datum.id:datum.in_pattern for datum in test_data}
    for pos, (raw_result, test_data) in enumerate(l):
        p_value, background_data, foreground_data, pattern_data, opt_regions, opt_xs = raw_result
        for datum in pattern_data:
            if positions[datum.id] > pos:
                positions[datum.id] = pos

    truth_and_scores = [(truths[_id], (1.0 - (pos/float(len(l))))) for (_id, pos) in positions.iteritems()]
    return [x[0] for x in truth_and_scores], [x[1] for x in truth_and_scores]

def ranked_pattern_finder_F_output_to_avg_objectwise_input(raw_results):
    return [single_pattern_finder_F.output_to_objectwise_precision_recall_input(*d) for d in ranked_pattern_finder_F.output_to_list_of_single_pattern_finder_F_output(raw_results)]


class ranked_pattern_finder_F(utils.F):

    @staticmethod
    def display_pattern_timeline(raw_results):
        l = ranked_pattern_finder_F.output_to_list_of_single_pattern_finder_F_output(raw_results)
        time_windows = []
        for single_result in l:
            (p_value, background_data, foreground_data, pattern_data, opt_regions, opt_xs), test_data = single_result
            min_time = min([datum.time for datum in foreground_data])
            max_time = max([datum.time for datum in foreground_data])
            time_windows.append((min_time, max_time))
        fig, ax = plt.subplots()
        time_windows = sorted(time_windows, key = lambda x: x[0])
        for i, (lower, upper) in enumerate(time_windows):
            ax.plot([lower, upper], [i,i])
        utils.display_fig_inline(fig)

    output_to_perf_curve_input = staticmethod(ranked_pattern_finder_F_output_to_perf_curve_input)

    @staticmethod
    def output_to_list_of_single_pattern_finder_F_output(l):
        original_test_data = l[0][1]
        return [(info, original_test_data) for (info, test_data) in l]

    output_to_avg_objectwise_input = staticmethod(ranked_pattern_finder_F_output_to_avg_objectwise_input)

    @staticmethod
    def display_results(k, raw_results, cat_fs = None, just_fails = False):
        """
        display performance curves, as well as informative plots of top k patterns
        """
        l = ranked_pattern_finder_F.output_to_list_of_single_pattern_finder_F_output(raw_results)
        perf_input = ranked_pattern_finder_F.output_to_perf_curve_input(raw_results)
        roc_fig, roc_ax = utils.get_roc_curve_fig(*perf_input)
        prec_fig, prec_ax = utils.get_precision_recall_curve_fig(*perf_input)
        utils.display_fig_inline(roc_fig)
        utils.display_fig_inline(prec_fig)

#        avg_objectwise_input = ranked_pattern_finder_F.output_to_avg_objectwise_input(raw_results)
#        avg_precision, avg_recall, avg_precision_with_zeros, avg_recall_with_zeros, total_correct, total_relevant, total_recalled, num_zeros, prunced_precision, prunced_recall, total_in_pattern = get_average_objectwise_precision_recall(avg_objectwise_input)
#        print 'average precision: %.2f, average recall: %.2f, \naverage precision with zeros: %.2f, average recall with zeros: %.2f, \ntotal correct: %d, total relevants: %d, total recalled: %d, \nnum_zeros: %d' % (avg_precision, avg_recall, avg_precision_with_zeros, avg_recall_with_zeros, total_correct, total_relevant, total_recalled, num_zeros)

        pattern_lengths = [len(pattern_data) for ((p_value, background_data, foreground_data, pattern_data, opt_regions, opt_x_subsets), test_data) in l]
        print 'pattern lengths: ', pattern_lengths
        print 'total in pattern: %d' % sum(pattern_lengths)
        print 'number of patterns: %d' % len(pattern_lengths)

        for s, i in zip(l, range(k)):
            pattern_data, test_data = single_pattern_finder_F.output_to_objectwise_precision_recall_input(*s)
            closest_pattern_id, closest_pattern_size, intersection_size = get_objectwise_precision_recall_info(pattern_data, test_data)

            if True:
            #if (just_fails and closest_pattern_id == None):
                print '\nRANK %d\n' % i

                try:
                    print 'intersection_size: %d, retrieved_size: %d, closest_pattern_size: %d, closest_pattern_id: %d' % (intersection_size, len(pattern_data), closest_pattern_size, closest_pattern_id)
                except:
                    print 'NO INTERSECTION, retrieved_size: %d' % len(pattern_data)
                (predicted_fig, predicted_ax), (true_fig, true_ax) = plot_pattern_informative(*single_pattern_finder_F.output_to_plot_pattern_informative_input(*s))
                utils.display_fig_inline(predicted_fig)
                utils.display_fig_inline(true_fig)
                if cat_fs == None:
                    single_pattern_finder_F.display_pattern_verbose(*single_pattern_finder_F.output_to_plot_pattern_informative_input(*s))
                else:
                    the_input = list(single_pattern_finder_F.output_to_plot_pattern_informative_input(*s)) + [cat_fs]
                    single_pattern_finder_F.display_pattern_verbose(*the_input)

    @staticmethod
    def display_textual_results(k, raw_results, cat_fs = None):
        l = ranked_pattern_finder_F.output_to_list_of_single_pattern_finder_F_output(raw_results)
        for s, i in zip(l, range(k)):
            if cat_fs == None:
                single_pattern_finder_F.display_textual_results(*s)
            else:
                single_pattern_finder_F.display_textual_results(*(list(s) + [cat_fs]))



class oneshot_ranked_pattern_finder_F(ranked_pattern_finder_F, data_F):

    def get_cache_self_attr_repr(self):
        return self.raw_pattern_finder_F.get_cache_self_attr_repr()

    def __init__(self, raw_pattern_finder_F):
        self.raw_pattern_finder_F = raw_pattern_finder_F

    def train(self, train_data):
        self.raw_pattern_finder_F.train(train_data)

    @caching.read_method_decorator(caching.read_pickle, utils.F.get_cache_path, 'pickle')
    @caching.write_method_decorator(caching.write_pickle, utils.F.get_cache_path, 'pickle')
#    @caching.default_cache_method_decorator
#    @caching.default_read_method_decorator
#    @utils.raise_exception_method_decorator(exceptions.TooLazyToComputeException)
#    @caching.default_write_method_decorator
    def __call__(self, test_data):
        import pdb
        print self.raw_pattern_finder_F
        raw_results, test_data = self.raw_pattern_finder_F(test_data)
        return [(result, test_data) for result in sorted(raw_results, key = lambda x:-1.0 * x[0])]


class iterative_ranked_pattern_finder_F(ranked_pattern_finder_F, data_F):

    def get_cache_self_attr_repr(self):
        return '%d_%s' % (self.max_iter, self.single_pattern_finder_F.get_cache_self_attr_repr())

    def __init__(self, max_iter, single_pattern_finder_F):
        self.max_iter, self.single_pattern_finder_F = max_iter, single_pattern_finder_F

    def train(self, train_data):
        self.single_pattern_finder_F.train(train_data)

    @utils.timeit_method_decorator()
#    @caching.default_cache_method_decorator
#    @caching.read_method_decorator(caching.read_pickle, utils.F.get_cache_path, 'pickle')
#    @caching.write_method_decorator(caching.write_pickle, utils.F.get_cache_path, 'pickle')
#    @caching.default_read_method_decorator
#    @utils.raise_exception_method_decorator(exceptions.TooLazyToComputeException)
#    @caching.default_write_method_decorator
    def __call__(self, test_data):
        results = []
        import python_utils.python_utils.exceptions as exceptions
        for i in range(self.max_iter):
            print 'iterative iter: %d' % i
            try:
                single_result, test_data = self.single_pattern_finder_F(test_data)
                p_value, background_data, foreground_data, pattern_data, opt_regions, opt_xs = single_result
                results.append((single_result, test_data))
                if len(pattern_data) == 0:
                    break
                test_data = list(set(test_data).difference(pattern_data))
            except exceptions.TooLazyToComputeException:
                break
            if len(test_data) == 0:
                break
        return results


def single_pattern_finder_F_output_to_perf_point_input((p_value, background_data, foreground_data, pattern_data, opt_regions, opt_x_subsets), test_data):
    retrieved_set = set([datum.id for datum in pattern_data])
    relevant_set = set([datum.id for datum in test_data if datum.in_pattern])
    total_set = set([datum.id for datum in test_data])
    return retrieved_set, relevant_set, total_set


class single_pattern_finder_F(utils.F):

    output_to_perf_point_input = staticmethod(single_pattern_finder_F_output_to_perf_point_input)

    @staticmethod
    def display_textual_results((p_value, background_data, foreground_data, pattern_data, opt_regions, opt_x_subsets), test_data, cat_fs = None):
        """
        does act of printing out results in text
        """
        print 'p_value: %.2f' % p_value
        print 'background_len: %d' % len(background_data)
        print 'foreground_len: %d' % len(foreground_data)
        print 'accuracy: %d/%d' % (len([datum for datum in pattern_data if datum.in_pattern]), len(pattern_data))
        print 'opt_regions:', opt_regions
        if cat_fs == None:
            print 'opt_x_subsets', opt_x_subsets
        else:
            import string
            string.join(['%s/%d' % (opt_x_subset, len(cat_f)) for (opt_x_subset, cat_f) in zip(opt_x_subsets, cat_fs)], sep = ', ')
        closest_pattern_id, closest_pattern_size, intersection_size = get_objectwise_precision_recall_info(pattern_data, test_data)
        print 'pattern_ids: ', [datum.id for datum in pattern_data]
        try:
            print 'closest_pattern: %d, closest_pattern_size: %d, intersection_size: %d' % (closest_pattern_id, closest_pattern_size, intersection_size)
        except:
            print 'NONE'

    @staticmethod
    def output_to_plot_pattern_input((p_value, background_data, foreground_data, pattern_data, opt_regions, opt_x_subsets), test_data):
        return (set(background_data), set(foreground_data), set(pattern_data)), test_data

    @staticmethod
    def output_to_objectwise_precision_recall_input((p_value, background_data, foreground_data, pattern_data, opt_regions, opt_x_subsets), test_data):
        return (set(pattern_data), test_data)

    @staticmethod
    def output_to_plot_pattern_informative_input((p_value, background_data, foreground_data, pattern_data, opt_regions, opt_x_subsets), test_data):
        """
        identity function
        """
        return (p_value, background_data, foreground_data, pattern_data, opt_regions, opt_x_subsets), test_data

    @staticmethod
    def display_pattern_verbose((p_value, background_data, foreground_data, pattern_data, opt_regions, opt_x_subsets), test_data, cat_fs = None):
        import crime_data.crime_data.fxns as crime_data_fxns
        from IPython.display import display_html
        import string
        if cat_fs != None:
            for cat_f, subset in zip(cat_fs, opt_x_subsets):
                def i_to_str(i):
                    if i == len(cat_f._bins):
                        return 'other'
                    if i < len(cat_f._bins):
                        return repr(cat_f._bins[i])
                    assert False
                print 'feature: %s.  categories: %s' % (repr(cat_f.f), string.join([i_to_str(i) for i in subset], sep = ', '))
        for i, datum in enumerate(sorted(pattern_data, key = lambda x:x.time)):
            verbose_info = crime_data_fxns.pinc_to_verbose_data(datum.id)
            print '\n\n\n NUMBER IN PATTERN: %d' % i
            print 'pattern_ids: ', [datum.id for datum in pattern_data]
            print 'CRIME INFO:'
            display_html(verbose_info[0].to_html(), raw=True)
            print 'PERSONS INFO:'
            try:
                display_html(verbose_info[1].to_html(), raw=True)
            except AttributeError:
                print 'NONE'
            try:
                import crime_data.crime_data.fxns as crime_data_fxns
                print crime_data_fxns.pinc_to_narrative(datum.id).iloc[0]
            except Exception, e:
                print e
                pdb.set_trace()
            #print 'PROPERTY INFO:'
            #try:
            #    display_html(verbose_info[2].to_html(), raw=True)
            #except AttributeError:
            #    print 'None'
            #import string


class most_significant_pattern_finder_F(data_F, single_pattern_finder_F):

    def get_cache_self_attr_repr(self):
        return self.raw_pattern_finder_F.get_cache_self_attr_repr()

    def get_max_tries(self, informatives):
        return self.raw_pattern_finder_F.get_max_tries(informatives)

    def get_avg_steps(self, informatives):
        return self.raw_pattern_finder_F.get_avg_steps(informatives)

    def __init__(self, raw_pattern_finder_F):
        self.raw_pattern_finder_F = raw_pattern_finder_F

    def train(self, train_data):
        self.raw_pattern_finder_F.train(train_data)

    @utils.timeit_method_decorator()
    @caching.read_method_decorator(caching.read_pickle, utils.F.get_cache_path, 'pickle')
    @caching.write_method_decorator(caching.write_pickle, utils.F.get_cache_path, 'pickle')
#    @caching.default_cache_method_decorator
#    @caching.default_read_method_decorator
    @utils.raise_exception_method_decorator(exceptions.TooLazyToComputeException)
#    @caching.default_write_method_decorator
#    def call_informative(self, test_data):
    def call_informative(self, test_data):
        (raw_results, test_data), informative = self.raw_pattern_finder_F.uninformative_informative(test_data)
        ans = max(raw_results, key = lambda x: x[0]), test_data
        return ans, informative


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
                for idx in regions.point_to_region_indicies(location):
                    counts[idx] += 1
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


def tong_data_ids():
    from tensor_scan import constants
    f_inc = open(constants.tong_fixed_incident_number_file, 'r')
    f_inc.next()
    pos_to_pincnum = [int(line.strip()) for line in f_inc]
    f_inc.close()
    return pos_to_pincnum


class tong_distance_F(utils.unsupervised_F):

    def __init__(self, daysapart=0.1600, geo_dis=0.3500, locofentry=0.1000, mnsofentry=0.0500, premises=0.0500, ransacked=0.0300, residentin=0.0500, timeofday=0.0200, dayofweek=0.0550, suspect=0.0550, victim=0.0800):
        self.daysapart, self.geo_dis, self.locofentry, self.mnsofentry, self.premises, self.ransacked, self.residentin, self.timeofday, self.dayofweek, self.suspect, self.victim = daysapart, geo_dis, locofentry, mnsofentry, premises, ransacked, residentin, timeofday, dayofweek, suspect, victim

    def __call__(self, test_data):
        # cheat. have position to pinc number list.  go through row by col, adding tuples of pincnum/pincnum->distance to a dictionary.  then in order of pincnums, use d to populate matrix
        from tensor_scan import constants
        pos_to_pincnum = tong_data_ids()
        data_pincnums = [datum.id for datum in test_data]
        data_pincnums_set = set(data_pincnums)
        relevant = [(pincnum in data_pincnums_set) for pincnum in pos_to_pincnum]
        N = len(data_pincnums)


        pos_to_pincnum_set = set(pos_to_pincnum)
        print 'sum:', pd.Series([datum.id in pos_to_pincnum_set for datum in test_data]).sum()

        for datum in test_data:
            if datum.id not in pos_to_pincnum_set:
                print datum.id
                pdb.set_trace()


        def read_relevant_entries_to_dict(path):
            d = {}
            f = open(path, 'r')
            f.next()
            for line, pincnum1, ok1 in zip(f, pos_to_pincnum, relevant):
                if ok1:
                    for sim, pincnum2, ok2 in zip(map(float, line.strip().split())[1:], pos_to_pincnum, relevant):
                        if ok2:
                            d[(pincnum1, pincnum2)] = sim
                            assert type(sim) == float
            f.close()
            mat = np.zeros(shape=(N, N))
            for i, pincnum1 in enumerate(data_pincnums):
                for j, pincnum2 in enumerate(data_pincnums):
                    try:
                        mat[i,j] = d[(pincnum1, pincnum2)]
                    except:
                        print 'bad', pincnum1, pincnum2
                        pdb.set_trace()
            return mat

        square_mat =\
            self.daysapart * read_relevant_entries_to_dict(constants.tong_daysapart_file) +\
            self.geo_dis * read_relevant_entries_to_dict(constants.tong_geo_dis_file) +\
            self.locofentry * read_relevant_entries_to_dict(constants.tong_locofentry_file) +\
            self.mnsofentry * read_relevant_entries_to_dict(constants.tong_mnsofentry_file) +\
            self.premises * read_relevant_entries_to_dict(constants.tong_premises_file) +\
            self.ransacked * read_relevant_entries_to_dict(constants.tong_ransacked_file) +\
            self.residentin * read_relevant_entries_to_dict(constants.tong_residentin_file) +\
            self.timeofday * read_relevant_entries_to_dict(constants.tong_timeofday_file) +\
            self.dayofweek * read_relevant_entries_to_dict(constants.tong_dayofweek_file) +\
            self.suspect * read_relevant_entries_to_dict(constants.tong_suspect_file) +\
            self.victim * read_relevant_entries_to_dict(constants.tong_victim_file)
       
        from scipy.spatial.distance import squareform

        return -1.0 * squareform(square_mat)


def scored_pattern_finder_F_output_to_perf_curve_input(datum_anomaly_score_tuples):
    return [x[0].in_pattern for x in datum_anomaly_score_tuples], [x[1] for x in datum_anomaly_score_tuples]
                  

class scored_pattern_finder_F(utils.F):

    @staticmethod
    def output_to_perf_curve_input(datum_anomaly_score_tuples):
        return [x[0].in_pattern for x in datum_anomaly_score_tuples], [x[1] for x in datum_anomaly_score_tuples]


    @staticmethod
    def display_results(results):
        perf_curve_input = scored_pattern_finder_F.output_to_perf_curve_input(results)
        roc_fig, roc_ax = utils.get_roc_curve_fig(*perf_curve_input)
        prec_fig, prec_ax = utils.get_precision_recall_curve_fig(*perf_curve_input)
        utils.display_fig_inline(roc_fig)
        utils.display_fig_inline(prec_fig)


class agglomerative_pattern_finder_F(scored_pattern_finder_F):

    def __init__(self, raw_F):
        self.raw_F = raw_F

    def __repr__(self):
        return repr(self.raw_F)
        return 'agg_clustering_%s_%s' % (repr(self.distance_F), self.method)

    def train(self, train_data):
        self.distance_F.train(train_data)

    @utils.timeit_method_decorator()
    @caching.default_cache_method_decorator
    @caching.default_read_method_decorator
    @caching.default_write_method_decorator
    def __call__(self, test_data):
        Z = self.raw_F(test_data)
        N = len(test_data)
        close_to_bottom = [z for i in xrange(Z.shape[0]) for z in Z[i,0:2] if z < N ]
        assert len(close_to_bottom) == N
        anomaly_scores = np.array(np.argsort([x for x in reversed(close_to_bottom)])) / float(N)
        ans = [(datum, anomaly_score) for (datum, anomaly_score) in itertools.izip(test_data, anomaly_scores)]
        return ans


class hac_raw_F(utils.F):
    """
    gets the hierarchical clustering tree
    """
    def __init__(self, distance_F, method):
        self.distance_F, self.method = distance_F, method

    def __repr__(self):
        return '%s_%s' % (repr(self.distance_F), self.method)

    @utils.timeit_method_decorator()
    @caching.default_cache_method_decorator
    @caching.default_read_method_decorator
    @caching.default_write_method_decorator
    def __call__(self, test_data):
        N = len(test_data)
        dist_mat_list = self.distance_F(test_data)
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import squareform
        dist_mat = squareform(dist_mat_list, force = 'tomatrix')
        Z = hierarchy.linkage(dist_mat_list, method = self.method)
        return Z

def hac_cluster_to_k_clusters_F_output_to_avg_objectwise_precision_recall(l):
    patterns, test_data = l
    return [(pattern_data, test_data) for pattern_data in patterns]


class hac_cluster_to_k_clusters_F(utils.F):

    output_to_avg_objectwise_precision_recall = staticmethod(hac_cluster_to_k_clusters_F_output_to_avg_objectwise_precision_recall)

    def __init__(self, raw_F, cluster_min_size, up_shift = 1.0):
        self.raw_F, self.cluster_min_size, self.up_shift = raw_F, cluster_min_size, up_shift

    def __call__(self, test_data, num_clusters):
        Z = self.raw_F(test_data)
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import squareform
        Z[:,2] = Z[:,2] + self.up_shift
        assignments = hierarchy.fcluster(Z, num_clusters, criterion = 'maxclust')
        d = {}
        for (datum, assignment) in zip(test_data, assignments):
            try:
                d[assignment].add(datum)
            except KeyError:
                d[assignment] = set([datum])
#        print pd.Series(assignments).value_counts()
        return [val for (key, val) in d.iteritems() if len(val) >= self.cluster_min_size], test_data

"""
def plot_k_clusters_score(k_clusters_F, ks, data):

    #assumes that input F (which takes in k) returns the input to avg_objectwise_precision_recall.  so for hac, would compose it with the converter
    #if this were more flexible, it would take the per-k output and the function that computes a score from it

    d = {}
    raw_scores = [get_average_objectwise_precision_recall(k_clusters_F(data, k)) for k in ks]
    avg_precisions, avg_recalls, avg_precisions_with_zeros, avg_recalls_with_zeros, total_corrects, total_relevant, total_recalled, num_zeros, pruned_precisions, pruned_recalls, total_in_patterns = zip(*raw_scores)
    running_avgs = [total / k for (k, total) in enumerate(np.cumsum(avg_precisions))]
    frac_in_patterns = [total / float(len(data)) for total in total_in_patterns]
    fig, ax = plt.subplots()
#    ax.plot(ks, running_avgs, label = 'running_avg')
    ax.scatter(ks, avg_precisions_with_zeros, label = 'avg precisions', color = 'r')
    ax.scatter(ks, pruned_precisions, label = 'pruned precisions', color = 'b')
    ax.scatter(ks, avg_recalls_with_zeros, label = 'avg recalls', color = 'g')
    ax.scatter(ks, pruned_recalls, label = 'pruned recalls', color = 'k')
    ax.plot(ks, frac_in_patterns, label = 'frac retrieved', color = 'c')
    ax.legend(prop = {'size':9})
    ax.set_xlim((0, None))
    ax.set_xlabel('number of clusters')
    fig.suptitle('HAC results')
    fig.tight_layout()
    return fig
"""



"""
for a cluster, compute: objectwise precision, objectwise recall, retrieved size, best pattern size(could be none), number in any pattern, number not in any pattern, cluster size
"""

def get_cluster_raw_info(pattern_data, test_data):
    """
    used by both
    """
    retrieved_pattern_size = len(pattern_data)
    in_pattern_pattern_data = [datum for datum in pattern_data if datum.in_pattern]
    num_in_any_pattern = len(in_pattern_pattern_data)
    num_not_in_any_pattern = retrieved_pattern_size - num_in_any_pattern
    counts = pd.Series({datum.id:datum.which_pattern for datum in in_pattern_pattern_data}).value_counts()
    if len(counts) == 0:
        objectwise_precision = 0.0
        objectwise_recall = 0.0
        best_pattern_size = None
        best_pattern_id = None
        intersection_size = 0
    else:
        best_pattern_id, intersection_size = max(counts.iteritems(), key = lambda x: x[1])
        best_pattern_size = len([datum for datum in test_data if datum.which_pattern == best_pattern_id])
        objectwise_precision = intersection_size / float(retrieved_pattern_size)
        objectwise_recall = intersection_size / float(best_pattern_size)

    return objectwise_precision, objectwise_recall, retrieved_pattern_size, best_pattern_size, num_in_any_pattern, num_not_in_any_pattern, best_pattern_id, intersection_size

def get_hac_raw_clustering_info(l):
    """
    accepts the output of k_cluster_F, which is a list of (pattern_data, test_data)
    """

    objectwise_precisions, objectwise_recalls = [], []
    total_retrieved = 0
    total_retrieved_in_any_pattern = 0

    for (pattern_data, test_data) in l:
        objectwise_precision, objectwise_recall, retrieved_pattern_size, best_pattern_size, num_in_any_pattern, num_not_in_any_pattern, best_pattern_id, intersection_size = get_cluster_raw_info(pattern_data, test_data)
        objectwise_precisions.append(objectwise_precision)
        objectwise_recalls.append(objectwise_recall)
        total_retrieved += retrieved_pattern_size
        total_retrieved_in_any_pattern += num_in_any_pattern

    avg_objectwise_precision = np.mean(objectwise_precisions)
    avg_objectwise_recall = np.mean(objectwise_recalls)
    total_retrieved_not_in_any_pattern = total_retrieved - total_retrieved_in_any_pattern

    return avg_objectwise_precision, avg_objectwise_recall, total_retrieved_in_any_pattern, total_retrieved, len(l)


def get_hac_results(k_cluster_F, ks, test_data):
    """
    returns the list of dictionaries that will be used for plotting
    assumes k_cluster_F returns input to hac_cluster_to_k_clusters_F_output_to_avg_objectwise_precision_recall
    """
    total_in_any_pattern = sum([datum.in_pattern for datum in test_data])
    total_not_in_any_pattern = len(test_data) - total_in_any_pattern

    results = []
    for k in ks:
        l = k_cluster_F(test_data, k)
        avg_objectwise_precision, avg_objectwise_recall, total_retrieved_in_any_pattern, total_retrieved, num_clusters = get_hac_raw_clustering_info(l)
        d = {\
            'clustering_avg_objectwise_precision': avg_objectwise_precision,\
                'clustering_avg_objectwise_recall': avg_objectwise_recall,\
                'clustering_in_any_pattern_retrieved_frac': total_retrieved_in_any_pattern / float(total_in_any_pattern),\
                'clustering_not_in_any_pattern_retrieved_frac': (total_retrieved - total_retrieved_in_any_pattern) / float(total_not_in_any_pattern),\
                'clustering_total_retrieved': total_retrieved,\
                'clustering_frac_retrieved': total_retrieved / float(len(test_data)),\
                'clustering_num_clusters': num_clusters,\
                }
        results.append(d)

    return sorted(results, key = lambda d: d['clustering_num_clusters'])


def get_subset_scan_results(raw_results):
    """
    assumes ss_F returns the input to hac_cluster_to_k_clusters_F_output_to_avg_objectwise_precision_recall
    """
    #l = hac_cluster_to_k_clusters_F_output_to_avg_objectwise_precision_recall(raw_results)
    l = ranked_pattern_finder_F_output_to_avg_objectwise_input(raw_results)
    test_data = l[0][1]
    all_test_data = l[0][1]
    asdf = len(l[0][1])
    total_in_any_pattern = sum([datum.in_pattern for datum in test_data])
    total_not_in_any_pattern = len(test_data) - total_in_any_pattern

    objectwise_precisions, objectwise_recalls, retrieved_pattern_sizes, best_pattern_sizes, num_in_any_patterns, num_not_in_any_patterns, best_pattern_ids, intersection_sizes = zip(*[get_cluster_raw_info(pattern_data, all_test_data) for (pattern_data, test_data) in l])

    def get_cum_avg(d):
        return [(v / float(i+1)) for (i, v) in enumerate(np.cumsum(d))]

    cumulative_avg_objectwise_precisions = get_cum_avg(objectwise_precisions)
    cumulative_avg_objectwise_recalls = get_cum_avg(objectwise_recalls)
    cumulative_retrieved_pattern_sizes = np.cumsum(retrieved_pattern_sizes)
    cumulative_num_in_any_patterns = np.cumsum(num_in_any_patterns)
    cumulative_num_not_in_any_patterns = cumulative_retrieved_pattern_sizes - cumulative_num_in_any_patterns

    cumulative_in_any_pattern_retrieved_fracs = cumulative_num_in_any_patterns / float(total_in_any_pattern)
    cumulative_not_in_any_pattern_retrieved_fracs = cumulative_num_not_in_any_patterns / float(total_not_in_any_pattern)

    cumulative_num_clusterss = range(len(l))


    results = [{\
            'cumulative_avg_objectwise_precision': cumulative_avg_objectwise_precision,\
                'cumulative_avg_objectwise_recall': cumulative_avg_objectwise_recall,\
                'cumulative_in_any_pattern_retrived_frac': cumulative_in_any_pattern_retrieved_frac,\
                'cumulative_not_in_any_pattern_retrieved_frac': cumulative_not_in_any_pattern_retrieved_frac,\
                'cumulative_total_retrieved': cumulative_retrieved_pattern_size,\
                'cumulative_frac_retrieved': cumulative_retrieved_pattern_size / float(asdf),\
                'cumulative_num_clusters': cumulative_num_clusters,\
                'objectwise_precision': objectwise_precision,\
                'objectwise_recall': objectwise_recall,\
                'retrieved_pattern_size': retrieved_pattern_size,\
                'best_pattern_size': best_pattern_size,\
                'intersection_size': intersection_size,\
                }\
                   for (cumulative_avg_objectwise_precision,\
                            cumulative_avg_objectwise_recall,\
                            cumulative_in_any_pattern_retrieved_frac,\
                            cumulative_not_in_any_pattern_retrieved_frac,\
                            cumulative_retrieved_pattern_size,\
                            cumulative_num_clusters,\
                            objectwise_precision,\
                            objectwise_recall,\
                            retrieved_pattern_size,\
                            best_pattern_size,\
                            intersection_size)\
                   in zip(\
            cumulative_avg_objectwise_precisions,\
                cumulative_avg_objectwise_recalls,\
                cumulative_in_any_pattern_retrieved_fracs,\
                cumulative_not_in_any_pattern_retrieved_fracs,\
                cumulative_retrieved_pattern_sizes,\
                cumulative_num_clusterss,\
                objectwise_precisions,\
                objectwise_recalls,\
                retrieved_pattern_sizes,\
                best_pattern_sizes,\
                intersection_sizes)\
                   ]
    return results

subset_scan_precision_color = 'red'
subset_scan_recall_color = 'blue'
hac_precision_color = 'black'
hac_recall_color = 'green'

ss_linestyle = '-'
hac_linestyle = '--'
precision_marker = '>'
recall_marker = '.'



import itertools
color_cycle = itertools.cycle(['b','g','r','c','m','y','k'])

def plot_subset_scan_results(ax, results, by_which, label):
    if by_which == 'num_clusters':
        xs = [result['cumulative_num_clusters'] for result in results]
        ax.set_xlabel('# clusters retrieved')
    if by_which == 'frac_retrieved':
        ax.set_xlabel('frac retrieved')
        xs = [result['cumulative_frac_retrieved'] for result in results]
    #color = color_cycle.next()
    color = 'b'
    ax.plot(xs, [result['cumulative_avg_objectwise_precision'] for result in results], color = color, label = 'ss_precision #%s' % label, marker = precision_marker, linestyle = ss_linestyle)
    ax.plot(xs, [result['cumulative_avg_objectwise_recall'] for result in results], color = 'g', label = 'ss_recall #%s' % label,  marker = recall_marker, linestyle = ss_linestyle)
    ax.scatter(xs, [result['objectwise_precision'] for result in results], color = color)
    ax.scatter(xs, [result['objectwise_recall'] for result in results], color = 'g')


def plot_hac_results(ax, results, by_which, label):
    if by_which == 'num_clusters':
        xs = [result['clustering_num_clusters'] for result in results]
        ax.set_xlabel('# clusters retrieved')
    if by_which == 'frac_retrieved':
        xs = [result['clustering_frac_retrieved'] for result in results]
        ax.set_xlabel('frac retrieved')
    #color = color_cycle.next()
    color = 'r'
    #ax.plot(xs, [result['clustering_avg_objectwise_precision'] for result in results], color = hac_precision_color, label = label)
    #ax.plot(xs, [result['clustering_avg_objectwise_recall'] for result in results], color = hac_recall_color)
    ax.plot(xs, [result['clustering_avg_objectwise_precision'] for result in results], color = color, label = 'hac_precision #%s' % label, linestyle = hac_linestyle, marker = precision_marker)
    ax.plot(xs, [result['clustering_avg_objectwise_recall'] for result in results], color = color, label = 'hac_recall #%s' % label, linestyle = hac_linestyle, marker = recall_marker)


fontsize = 6

def plot_subset_scan_precrec(ax, results, label):
    #ax.plot([result['cumulative_avg_objectwise_recall'] for result in results], [result['cumulative_avg_objectwise_precision'] for result in results], color = subset_scan_precision_color, label = 'ss ' + label)
    ax.set_xlabel('objectwise recall')
    ax.set_ylabel('objectwise precision')
    color = 'r'
    #color = color_cycle.next()
    ax.plot([result['cumulative_avg_objectwise_recall'] for result in results], [result['cumulative_avg_objectwise_precision'] for result in results], linestyle = ss_linestyle, label = 'ss ' + label, color = color)
#    ax.scatter([result['cumulative_avg_objectwise_recall'] for result in results], [result['cumulative_avg_objectwise_precision'] for result in results], label = 'ss ' + label, color = color, s = [result['cumulative_frac_retrieved'] * 50 for result in results])
    for result in results:
        ax.text(result['cumulative_avg_objectwise_recall'], result['cumulative_avg_objectwise_precision'], color = 'k', s = '%.2f' % result['cumulative_frac_retrieved'], fontsize = fontsize)


def plot_hac_precrec(ax, results, label):
    #ax.plot([result['clustering_avg_objectwise_recall'] for result in results], [result['clustering_avg_objectwise_precision'] for result in results], color = hac_precision_color, label = 'hac ' + label)
    ax.set_xlabel('objectwise recall')
    ax.set_ylabel('objectwise precision')
    color = color_cycle.next()
    ax.plot([result['clustering_avg_objectwise_recall'] for result in results], [result['clustering_avg_objectwise_precision'] for result in results], linestyle = hac_linestyle, label = 'hac ' + label, color = color)
#    ax.scatter([result['clustering_avg_objectwise_recall'] for result in results], [result['clustering_avg_objectwise_precision'] for result in results], linestyle = hac_linestyle, label = 'hac ' + label, color = color, s = [result['clustering_frac_retrieved'] * 50 for result in results])
    for result in results:
        ax.text(result['clustering_avg_objectwise_recall'], result['clustering_avg_objectwise_precision'], color = 'k', s = '%.2f' % result['clustering_frac_retrieved'], fontsize = fontsize)


def plot_subset_scan_objectwise_results(l):
    """
    takes same input as get_average_objectwise_precision_recall
    """
    fig = plt.figure()


    frac_ax = fig.add_subplot(1,1,1)

    asdf_fig, num_ax = plt.subplots()
#    num_ax = fig.add_subplot(2,1,2)
    best_pattern_sizes = []
    intersection_sizes = []
    pattern_sizes = []
    precisions = []
    recalls = []
    
    data_length = len(iter(l).next()[1])

    for (i, r) in enumerate(l):
        pattern_data, test_data = r
        best_pattern_id, best_pattern_size, intersection_size = get_objectwise_precision_recall_info(pattern_data, test_data)
        if best_pattern_id == None:
            best_pattern_sizes.append(np.inf)
            intersection_sizes.append(np.inf)
            pattern_sizes.append(len(pattern_data))
            precisions.append(0.0)
            recalls.append(0.0)
        else:
            best_pattern_sizes.append(best_pattern_size)
            intersection_sizes.append(intersection_size)
            pattern_sizes.append(len(pattern_data))
            precisions.append(intersection_size / float(len(pattern_data)))
            recalls.append(intersection_size / float(best_pattern_size))

    N = len(l)
    num_ax.scatter(range(N), best_pattern_sizes, color = 'r', label = 'best_pattern_size')
    num_ax.scatter(range(N), intersection_sizes, color = 'g', label = 'intersection_pattern_size')
    num_ax.scatter(range(N), pattern_sizes, color = 'b', label = 'retrieved_pattern_size')
    frac_ax.scatter(range(N), precisions, color = 'b', label = 'precision')
    frac_ax.scatter(range(N), recalls, color = 'r', label = 'recall')

    precision_running_avg = [(p / float(k+1)) for (k, p) in enumerate(np.cumsum(precisions))]
    recall_running_avg = [(p / float(k+1)) for (k, p) in enumerate(np.cumsum(recalls))]

    pct_retrieved = np.cumsum(pattern_sizes) / float(data_length)
    frac_ax.plot(range(N), pct_retrieved, color = 'k', label = 'frac retrieved')


    frac_ax.plot(range(N), precision_running_avg, color = 'b')
    frac_ax.plot(range(N), recall_running_avg, color = 'r')

    frac_ax.set_xlabel('pattern rank')
    num_ax.set_xlabel('pattern rank')

    num_ax.legend(prop = {'size':6})
    frac_ax.legend(prop = {'size':6})

    frac_ax.set_title('precision/recalls')
    num_ax.set_title('raw numbers')
    frac_ax.set_xlim((0, None))
    num_ax.set_xlim((0, None))
 
    fig.suptitle('subsetscan results')

    return fig, frac_ax
    return fig, (num_ax, frac_ax)
        


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
        return itertools.ifilter(lambda (background_data, foreground_data): len(background_data) != 0 and len(foreground_data) != 0 and len(foreground_data) < N / 5, itertools.starmap(start_end_idx_to_background_foreground, itertools.combinations(idx_boundaries,2)))


def data_to_raw_pattern_lengths(data):
    id_to_years = {}
    for datum in data:
        if datum.in_pattern:
            try:
                id_to_years[datum.which_pattern].append(datum.time)
            except KeyError:
                id_to_years[datum.which_pattern] = [datum.time]

    return {_id:len(years) for (_id, years) in id_to_years.iteritems()}
                
    for key, val in id_to_years.iteritems():
        id_to_years[key] = sorted(val)


def get_cheating_windows(data):
    id_to_years = {}
    for datum in data:
        if datum.in_pattern:
            try:
                id_to_years[datum.which_pattern].append(datum.time)
            except KeyError:
                id_to_years[datum.which_pattern] = [datum.time]

    id_to_foreground_time_windows = {_id:(min(years), max(years)) for (_id, years) in id_to_years.iteritems()}

    def year_window_to_split(min_year, max_year):
        foreground_data, background_data = [], []
        for datum in data:
            if datum.time >= min_year and datum.time <= max_year:
                foreground_data.append(datum)
            else:
                background_data.append(datum)
        return (background_data, foreground_data)

    return [year_window_to_split(low_year, high_year) for (_id, (low_year, high_year)) in id_to_foreground_time_windows.iteritems()]



class cheating_windows_iterator(utils.unsupervised_F):
    """
    for now, hardcode what the patterns are.  later on, could have a function defining what it means to be in a pattern
    """
    def __init__(self, min_window_length):
        self.min_window_length = min_window_length

    def __call__(self, data):
        return [(background, foreground) for (background, foreground) in get_cheating_windows(data) if len(foreground) > self.min_window_length]


class chained_windows_iterator(utils.unsupervised_F):

    def __init__(self, iterators):
        # calls a bunch of functions on same data (to get several iterators), then calls a single function on the results
        self.iterators = iterators

    def __call__(self, data):
        return list(itertools.chain(*[iterator(data) for iterator in self.iterators]))

class sliding_windows_iterator(utils.unsupervised_F):
    """
    if max pattern width is T, then having a window of size T+eps with eps increments will ensure that some window encloses the pattern
    """

    def get_cache_self_attr_repr(self):
        import string
        return string.join([('%.3f_%.3f' % (width, increment)) for (width, increment) in self.time_width_time_increments], sep = '_')

    def __init__(self, time_width_time_increments, min_window_length, min_time = None, max_time = None):
        self.time_width_time_increments, self.min_window_length = time_width_time_increments, min_window_length
        self.min_time, self.max_time = min_time, max_time

    def __call__(self, data):
        N = len(data)
        sorted_data = sorted(data, key = lambda datum: datum.time)
        if self.min_time == None or self.max_time == None:
            min_time, max_time = sorted_data[0].time, sorted_data[-1].time
        else:
            max_time, min_time = self.max_time, self.min_time

        ans = []

        for time_width, time_increment in self.time_width_time_increments:
            
            #print time_width, time_increment
            num_blocks = int((max_time - min_time) / time_increment)
            #print num_blocks
            start_times = np.linspace(min_time, min_time + num_blocks * time_increment, num_blocks)
            #print start_times
            end_times = [start_time + time_width for start_time in start_times]

            start_idxs = []
            start_idx = len(sorted_data) - 1

            actual_start_times = []

            for start_time in reversed(start_times):
                while sorted_data[start_idx].time >= start_time:
                    if start_idx == 0:
                        break
                    start_idx = start_idx - 1
                start_idxs.append(start_idx)
                actual_start_times.append(sorted_data[start_idx].time)
            start_idxs = list(reversed(start_idxs))
            actual_start_times = list(reversed(actual_start_times))

            end_idxs = []
            end_idx = 0

            actual_end_times = []

            for end_time in end_times:
                while sorted_data[end_idx].time <= end_time:
                    if end_idx == len(sorted_data) - 1:
                        break
                    end_idx = end_idx + 1
                end_idxs.append(end_idx)
                actual_end_times.append(sorted_data[end_idx].time)

            def start_end_idx_to_background_foreground(start_idx, end_idx):
                return sorted_data[0:start_idx] + sorted_data[end_idx:], sorted_data[start_idx:end_idx]

            #print len(start_idxs)
            #pdb.set_trace()
            print 'aLEN:', len(start_idxs)

            #print [(start, end, end-start) for (start, end) in zip(actual_start_times, actual_end_times)]
            #print [(actual_start, start, start-actual_start) for (actual_start, start) in zip(actual_start_times, start_times)]
            #print [(actual_end, end, end-actual_end) for (actual_end, end) in zip(actual_end_times, end_times)]

            ans = ans + [start_end_idx_to_background_foreground(s,e) for (s, e) in itertools.izip(start_idxs, end_idxs)]

        #pdb.set_trace()
        print 'LEN: ', len(ans)
        #print zip(start_idxs, end_idxs)

        #pdb.set_trace()

        return [x for x in ans if len(x[0]) > self.min_window_length]

        #return [start_end_idx_to_background_foreground(s,e) for (s, e) in itertools.izip(start_idxs, end_idxs)]

        for_now = zip(start_idxs, end_idxs)

        pdb.set_trace()

        # for each window, extend the endpoint if necessary
        newer = []
        for i, (s, e) in enumerate(for_now):
            new_e = e
            j = 0
            while new_e < s + self.min_idx_window or i != len(for_now)-1:
                j = j + 1
                new_e = for_now[j]
            newer.append((s, new_e))

        pdb.set_trace()

        return [start_end_idx_to_background_foreground(s,e) for (s, e) in itertools.izip(start_idxs, end_idxs)]



pattern_window_color, pattern_color, background_color = 'red', 'blue', 'black'


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
    predicted_ax.scatter([xy[0] for xy in background_xys], [xy[1] for xy in background_xys], label = 'background crimes', color = background_color, s = marker_size, alpha = alpha)
    predicted_ax.scatter([xy[0] for xy in pattern_xys], [xy[1] for xy in pattern_xys], label = 'pattern crimes', color = pattern_color, s = 2.0 * marker_size, alpha = alpha)
    predicted_ax.scatter([xy[0] for xy in not_pattern_xys], [xy[1] for xy in not_pattern_xys], label = 'not pattern crimes', color = pattern_window_color, s = marker_size, alpha = alpha)
    predicted_fig.suptitle('predicted')
    predicted_ax.legend()

    true_fig, true_ax = plt.subplots()
    true_ax.scatter([xy[0] for xy in background_xys], [xy[1] for xy in background_xys], label = 'background crimes', color = background_color, s = marker_size, alpha = alpha)
    true_ax.scatter([xy[0] for xy in true_pattern_xys], [xy[1] for xy in true_pattern_xys], label = 'true pattern crimes', color = pattern_color, s = 2.0 * marker_size, alpha = alpha)
    true_ax.scatter([xy[0] for xy in true_not_pattern_xys], [xy[1] for xy in true_not_pattern_xys], label = 'true not pattern crimes', color = pattern_window_color, s = marker_size, alpha = alpha)
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
    predicted_fig.suptitle('test_stat: %.2f time_range: (%.2f,%.2f) foreground_len: %d pattern_len: %d' % (p_value, min_time, max_time, len(foreground_data), len(pattern_data)))
    true_fig.suptitle('opt_xs: %s' % str(opt_x_subsets))
    return (predicted_fig, predicted_ax), (true_fig, true_ax)


def get_objectwise_precision_recall_info(pattern_data, test_data):
    """
    find most popular pattern.  if none, answer is 0.  return best_pattern_id, best_pattern_size, precision, recall
    """
    counts = pd.Series({datum.id:datum.which_pattern for datum in pattern_data if datum.which_pattern != None}).value_counts()
    if len(counts) == 0:
        return (None, None, None)
    [(_id, count) for (_id, count) in counts.iteritems()]
    best_pattern_id, intersection_size = max(counts.iteritems(), key = lambda x: x[1])
    print counts
    best_pattern_data = set([datum for datum in test_data if datum.which_pattern == best_pattern_id])
    #print len(best_pattern_data)
    if len(best_pattern_data) == 0:
        print 'FADAD'
        pdb.set_trace()

    return best_pattern_id, len(best_pattern_data), intersection_size
#    return best_pattern_id, len(best_pattern_data), intersection_size / float(len(pattern_data)), intersection_size / float(len(best_pattern_size))


def get_subset_scan_all_info_horse(l):
    """
    accepts the input to get_average_objectwise_precision_recall.  use the converter in wrapper function
    number of patterns retrieved to that point
    objectwise precision to that point
    objectwise recall to that point
    fraction of data retrieved to that point
    fraction of total negatives retrieved to that point, aka false negatives in roc curve
    fraction of total positives retrieved to that point, aka true positive rate in roc curve
    """
    pass


def get_subset_scan_all_info(ranked_F, data):
    """
    accepts ranked pattern finder, without any converters
    """
    return get_subset_scan_all_info_horse(ranked_pattern_finder_F_output_to_avg_objectwise_input(ranked_F(data)))




def get_average_objectwise_precision_recall(l):
    """
    by definition takes in a list of the inputs for above fxn
    """
    min_pattern_length = 1

    precisions, recalls = [], []
    precisions_with_zeros, recalls_with_zeros = [], []
    pruned_precisions, pruned_recalls = [], []
    total_corrects, total_relevant, total_recalled = 0, 0, 0
    total_in_a_pattern = 0
    total_in_pattern = 0
    num_empty = 0
    for r in l:
        pattern_data, test_data = r
        total_recalled += len(pattern_data)
        best_pattern_id, best_pattern_size, intersection_size = get_objectwise_precision_recall_info(pattern_data, test_data)
        #print best_pattern_size, intersection_size, len(pattern_data)
        if best_pattern_id == None:
            num_empty += 1
            precisions_with_zeros.append(0.0)
            recalls_with_zeros.append(0.0)
            if len(pattern_data) > min_pattern_length:
                pruned_precisions.append(0.0)
                pruned_recalls.append(0.0)
                total_in_pattern += len(pattern_data)
        else:
            precisions.append(float(intersection_size) / len(pattern_data))
            recalls.append(float(intersection_size) / best_pattern_size)
            precisions_with_zeros.append(float(intersection_size) / len(pattern_data))
            recalls_with_zeros.append(float(intersection_size) / best_pattern_size)
            total_corrects += intersection_size
            if len(pattern_data) > min_pattern_length:
                pruned_precisions.append(float(intersection_size) / len(pattern_data))
                pruned_recalls.append(float(intersection_size) / best_pattern_size)
                total_in_pattern += len(pattern_data)

    #print np.mean(precisions_with_zeros), np.mean(recalls_with_zeros), num_empty, len(l)
    # also return the number of negatives retrieved, and number of
    total_not_in_a_pattern = 0
    return np.mean(precisions), np.mean(recalls), np.mean(precisions_with_zeros), np.mean(recalls_with_zeros), total_in_a_pattern, total_not_in_a_pattern , total_recalled, num_empty, np.mean(pruned_precisions), np.mean(pruned_recalls), total_in_pattern




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
#    spatial_ax.scatter([xy[0] for xy in background_xys], [xy[1] for xy in background_xys], s = marker_size, alpha = alpha, color = 'r', label = 'background', marker = '.')
#    spatial_ax.scatter([xy[0] for xy in pattern_window_background_xys], [xy[1] for xy in pattern_window_background_xys], s = marker_size, alpha = alpha, color = 'r', label = 'foreground', marker = '.')
    spatial_ax.scatter([xy[0] for xy in pattern_xys], [xy[1] for xy in pattern_xys], s = marker_size, alpha = 0.4, color = pattern_color, label = 'pattern')



    pattern_start_time, pattern_end_time = pattern_data[0].time, pattern_data[-1].time
    spatial_ax.set_title('start: %.2f, end: %.2f, len/window: %d/%d' % (pattern_start_time, pattern_end_time, len(pattern_data), len(pattern_window_background_data)))
    spatial_ax.legend(prop = {'size':3})
    """
    hope that density of pattern_window is higher than that of background in region where the pattern is
    """

    def get_data_feature_counts(data):
        df = pd.DataFrame({datum.id:datum.x for datum in data})
        counts = [row.value_counts() for (name, row) in df.iterrows()]
        return counts
        

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
    return (spatial_fig, spatial_ax), feature_figs
