"""
Adapted from Allen from:
https://github.com/AllenInstitute/ecephys_spike_sorting.git
"""
import click
import warnings
from pdb import set_trace
import numpy as np
import os
from collections import OrderedDict

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

from scipy.spatial.distance import cdist
from scipy.stats import chi2
from scipy.ndimage.filters import gaussian_filter1d

import pandas as pd


# TODO metrics goes on biowulf

def get_phy_simple(fid='hp28_butter', path='/home/m/ssd/res/maze/raw/', fs=30000):
    phy_folder = path + fid + '/' + fid + '.GUI/'
    groupfname = os.path.join(phy_folder, 'cluster_groups.csv')

    groups = pd.read_table(groupfname, delimiter='\t')
    # TODO this will be based on Metrics class
    takegroups = groups[(groups.group == 'good')].cluster_id

    # load spike times and cluster IDs
    with open(phy_folder + 'spike_clusters.npy', 'rb') as f:
        ids = np.load(f).flatten()
    with open(phy_folder + 'spike_times.npy', 'rb') as f:
        ts = np.load(f).flatten()

    # only take spikes that are in our "good" group
    takespikes = np.array([], dtype='int')
    for i in takegroups:
        takespikes = np.append(takespikes, (ids == i).nonzero()[0])
    ids = ids[takespikes]
    ts = ts[takespikes]
    ts = ts.astype(float) / fs

    # Remove non-unique timestamps
    ts, ii = np.unique(ts, return_index=True)
    ids = ids[ii]
    print(len(np.unique(ids)))
    Ts = []
    Ids = []
    # Loop over ids
    for iu in np.unique(ids):
        thists = ts[ids == iu]
        ihere = np.where(np.diff(thists) > .0008)[0]
        # skip spikes that are too close together (likely an alignment issue):
        # TODO this should be done by Allen's helper
        if sum(np.diff(thists[ihere]) < .001) > 20:
            continue
        else:  # take good spikes
            Ts.extend(thists[ihere])
            Ids.extend(np.repeat(iu, len(ihere)))
    # convert to numpy
    ts = np.array(Ts)
    ids = np.array(Ids)
    return ids, ts


class KsortPostprocessing:
    """

    Clean up Kilosort outputs by removing putative double-counted spikes.

    Kilosort occasionally fits a spike template to the residual of another spike. See this discussion for more information.

    This module aims to correct for this by removing spikes from the same unit or neighboring units that occur within 5 samples (0.16 ms) of one another. This is not ideal, since it can potentially remove legitimate spike times, but on the whole it seems worth it to avoid having spurious zero-time-lag correlation between units.

    We are not currently taking into account spike amplitude when removing spikes; the module just deletes one spike from an overlapping pair that occurs later in time.
    """

    @staticmethod
    def remove_double_counted_spikes(spike_times, spike_clusters, spike_templates, amplitudes, channel_map, templates,
                                     pc_features, sample_rate, within_unit_overlap_window=0.000166,
                                     between_unit_overlap_window=0.000166, between_unit_channel_distance=5
                                     ):

        """ Remove putative double-counted spikes from Kilosort outputs
        Inputs:
        ------
        spike_times : numpy.ndarray (num_spikes x 0)
            Spike times in samples
        spike_clusters : numpy.ndarray (num_spikes x 0)
            Cluster IDs for each spike time
        spike_templates : numpy.ndarray (num_spikes x 0)
            Template IDs for each spike time
        amplitudes : numpy.ndarray (num_spikes x 0)
            Amplitude value for each spike time
        channel_map : numpy.ndarray (num_units x 0)
            Original data channel for pc_feature_ind array
        templates : numpy.ndarray (num_units x num_channels x num_samples)
            Spike templates for each unit
        pc_features : numpy.ndarray (num_spikes x num_pcs x num_channels)
            Pre-computed PCs for blocks of channels around each spike
        pc_feature_ind : numpy.ndarray (num_units x num_channels)
            Channel indices of PCs for each unit
        sample_rate : Float
            Sample rate of spike times
        'within_unit_overlap_window' :
            time window for removing overlapping spikes
        'between_unit_overlap_window' :
            time window for removing overlapping spikes
        'between_unit_channel_distance' :
             number of channels over which to search for overlapping spikes

        Outputs:
        --------
        spike_times : numpy.ndarray (num_spikes x 0)
            Spike times in seconds (same timebase as epochs)
        spike_clusters : numpy.ndarray (num_spikes x 0)
            Cluster IDs for each spike time
        spike_templates : numpy.ndarray (num_spikes x 0)
            Template IDs for each spike time
        amplitudes : numpy.ndarray (num_spikes x 0)
            Amplitude value for each spike time
        pc_features : numpy.ndarray (num_spikes x num_pcs x num_channels)
            Pre-computed PCs for blocks of channels around each spike
        overlap_matrix : numpy.ndarray (num_clusters x num_clusters)
            Matrix indicating number of spikes removed for each pair of clusters
        """

        def find_within_unit_overlap(spike_train, overlap_window=5):

            """
            Finds overlapping spikes within a single spike train.
            Parameters
            ----------
            spike_train : numpy.ndarray
                Spike times (in samples)
            overlap_window : int
                Number of samples to search for overlapping spikes
            Outputs
            -------
            spikes_to_remove : numpy.ndarray
                Indices of overlapping spikes in spike_train
            """

            spikes_to_remove = np.where(np.diff(spike_train) < overlap_window)[0]

            return spikes_to_remove

        def find_between_unit_overlap(spike_train1, spike_train2, overlap_window=5):

            """
            Finds overlapping spikes between two spike trains
            Parameters
            ----------
            spike_train1 : numpy.ndarray
                Spike times (in samples)
            spike_train2 : numpy.ndarray
                Spike times (in samples)
            overlap_window : int
                Number of samples to search for overlapping spikes
            Outputs
            -------
            spikes_to_remove1 : numpy.ndarray
                Indices of overlapping spikes in spike_train1
            spikes_to_remove2 : numpy.ndarray
                Indices of overlapping spikes in spike_train2
            """

            spike_train = np.concatenate((spike_train1, spike_train2))
            original_inds = np.concatenate((np.arange(len(spike_train1)), np.arange(len(spike_train2))))
            cluster_ids = np.concatenate((np.zeros((len(spike_train1),)), np.ones((len(spike_train2),))))

            order = np.argsort(spike_train)
            sorted_train = spike_train[order]
            sorted_inds = original_inds[order][1:]
            sorted_cluster_ids = cluster_ids[order][1:]

            spikes_to_remove = np.diff(sorted_train) < overlap_window

            spikes_to_remove1 = sorted_inds[spikes_to_remove * (sorted_cluster_ids == 0)]
            spikes_to_remove2 = sorted_inds[spikes_to_remove * (sorted_cluster_ids == 1)]

            return spikes_to_remove1, spikes_to_remove2

        def remove_spikes(spike_times, spike_clusters, spike_templates, amplitudes, pc_features, spikes_to_remove):

            """
            Removes spikes from Kilosort outputs
            Inputs:
            ------
            spike_times : numpy.ndarray (num_spikes x 0)
                Spike times in samples
            spike_clusters : numpy.ndarray (num_spikes x 0)
                Cluster IDs for each spike time
            spike_templates : numpy.ndarray (num_spikes x 0)
                Template IDs for each spike time
            amplitudes : numpy.ndarray (num_spikes x 0)
                Amplitude value for each spike time
            pc_features : numpy.ndarray (num_spikes x num_pcs x num_channels)
                Pre-computed PCs for blocks of channels around each spike
            spikes_to_remove : numpy.ndarray
                Indices of spikes to remove
            Outputs:
            --------
            spike_times : numpy.ndarray (num_spikes - spikes_to_remove x 0)
            spike_clusters : numpy.ndarray (num_spikes - spikes_to_remove x 0)
            spike_templates : numpy.ndarray (num_spikes - spikes_to_remove x 0)
            amplitudes : numpy.ndarray (num_spikes - spikes_to_remove x 0)
            pc_features : numpy.ndarray (num_spikes - spikes_to_remove x num_pcs x num_channels)
            """

            spike_times = np.delete(spike_times, spikes_to_remove, 0)
            spike_clusters = np.delete(spike_clusters, spikes_to_remove, 0)
            spike_templates = np.delete(spike_templates, spikes_to_remove, 0)
            amplitudes = np.delete(amplitudes, spikes_to_remove, 0)
            pc_features = np.delete(pc_features, spikes_to_remove, 0)

            return spike_times, spike_clusters, spike_templates, amplitudes, pc_features

        unit_list = np.arange(np.max(spike_clusters) + 1)

        peak_channels = np.squeeze(channel_map[np.argmax(np.max(templates, 1) - np.min(templates, 1), 1)])


        order = np.argsort(peak_channels)
        # Skip order if they are not present in unit_list:
        order=order[np.in1d(order,unit_list)]
        overlap_matrix = np.zeros((peak_channels.size, peak_channels.size))

        within_unit_overlap_samples = int(within_unit_overlap_window * sample_rate)
        between_unit_overlap_samples = int(between_unit_overlap_window * sample_rate)
        print('Removing within-unit overlapping spikes...')
        spikes_to_remove = np.zeros((0,))

        for idx1, unit_id1 in enumerate(unit_list[order]):
            for_unit1 = np.where(spike_clusters == unit_id1)[0]
            to_remove = find_within_unit_overlap(spike_times[for_unit1], within_unit_overlap_samples)
            overlap_matrix[idx1, idx1] = len(to_remove)
            spikes_to_remove = np.concatenate((spikes_to_remove, for_unit1[to_remove]))

        spike_times, spike_clusters, spike_templates, amplitudes, pc_features = remove_spikes(spike_times,
                                                                                              spike_clusters,
                                                                                              spike_templates,
                                                                                              amplitudes,
                                                                                              pc_features,
                                                                                              spikes_to_remove)

        print('Removing between-unit overlapping spikes...')
        spikes_to_remove = np.zeros((0,))

        for idx1, unit_id1 in enumerate(unit_list[order]):

            for_unit1 = np.where(spike_clusters == unit_id1)[0]

            for idx2, unit_id2 in enumerate(unit_list[order]):

                if idx2 > idx1 and (np.abs(peak_channels[unit_id1] - peak_channels[unit_id2]) <
                                    between_unit_channel_distance):
                    for_unit2 = np.where(spike_clusters == unit_id2)[0]

                    to_remove1, to_remove2 = find_between_unit_overlap(spike_times[for_unit1], spike_times[for_unit2],
                                                                       between_unit_overlap_samples)

                    overlap_matrix[idx1, idx2] = len(to_remove1) + len(to_remove2)

                    spikes_to_remove = np.concatenate((spikes_to_remove, for_unit1[to_remove1], for_unit2[to_remove2]))

        spike_times, spike_clusters, spike_templates, amplitudes, pc_features = remove_spikes(spike_times,
                                                                                              spike_clusters,
                                                                                              spike_templates,
                                                                                              amplitudes,
                                                                                              pc_features,
                                                                                              np.unique(
                                                                                                  spikes_to_remove))

        return spike_times, spike_clusters, spike_templates, amplitudes, pc_features, overlap_matrix


class IO:
    """
    Adapted from Allen from:
    https://github.com/AllenInstitute/ecephys_spike_sorting.git
    """

    @staticmethod
    def load_kilosort_data(folder,
                           sample_rate=None,
                           convert_to_seconds=True,
                           use_master_clock=False,
                           include_pcs=False,
                           template_zero_padding=21):

        """
        Loads Kilosort output files from a directory
        Inputs:
        -------
        folder : String
            Location of Kilosort output directory
        sample_rate : float (optional)
            AP band sample rate in Hz
        convert_to_seconds : bool (optional)
            Flags whether to return spike times in seconds (requires sample_rate to be set)
        use_master_clock : bool (optional)
            Flags whether to load spike times that have been converted to the master clock timebase
        include_pcs : bool (optional)
            Flags whether to load spike principal components (large file)
        template_zero_padding : int (default = 21)
            Number of zeros added to the beginning of each template
        Outputs:
        --------
        spike_times : numpy.ndarray (N x 0)
            Times for N spikes
        spike_clusters : numpy.ndarray (N x 0)
            Cluster IDs for N spikes
        spike_templates : numpy.ndarray (N x 0)
            Template IDs for N spikes
        amplitudes : numpy.ndarray (N x 0)
            Amplitudes for N spikes
        unwhitened_temps : numpy.ndarray (M x samples x channels)
            Templates for M units
        channel_map : numpy.ndarray
            Channels from original data file used for sorting
        cluster_ids : Python list
            Cluster IDs for M units
        cluster_quality : Python list
            Quality ratings from cluster_group.tsv file
        pc_features (optinal) : numpy.ndarray (N x channels x num_PCs)
            PC features for each spike
        pc_feature_ind (optional) : numpy.ndarray (M x channels)
            Channels used for PC calculation for each unit
        """
        folder = os.path.expanduser(folder)

        def load(folder, filename):

            """
            Loads a numpy file from a folder.
            Inputs:
            -------
            folder : String
                Directory containing the file to load
            filename : String
                Name of the numpy file
            Outputs:
            --------
            data : numpy.ndarray
                File contents
            """

            return np.load(os.path.join(folder, filename))

        def read_cluster_group_tsv(filename):

            """
            Reads a tab-separated cluster_group.tsv file from disk
            Inputs:
            -------
            filename : String
                Full path of file
            Outputs:
            --------
            IDs : list
                List of cluster IDs
            quality : list
                Quality ratings for each unit (same size as IDs)
            """

            info = np.genfromtxt(filename, dtype='str')
            cluster_ids = info[1:, 0].astype('int')
            cluster_quality = info[1:, 1]

            return cluster_ids, cluster_quality

        if use_master_clock:
            spike_times = load(folder, 'spike_times_master_clock.npy')
        else:
            spike_times = load(folder, 'spike_times.npy')

        spike_clusters = load(folder, 'spike_clusters.npy')
        spike_templates = load(folder, 'spike_templates.npy')
        amplitudes = load(folder, 'amplitudes.npy')
        templates = load(folder, 'templates.npy')
        unwhitening_mat = load(folder, 'whitening_mat_inv.npy')
        channel_map = load(folder, 'channel_map.npy')

        if include_pcs:
            pc_features = load(folder, 'pc_features.npy')
            pc_feature_ind = load(folder, 'pc_feature_ind.npy')

        templates = templates[:, template_zero_padding:, :]  # remove zeros
        spike_clusters = np.squeeze(spike_clusters)  # fix dimensions
        spike_times = np.squeeze(spike_times)  # fix dimensions
        spike_templates = np.squeeze(spike_templates)  # fix dimensions

        if convert_to_seconds and sample_rate is not None:
            spike_times = spike_times / sample_rate

        unwhitened_temps = np.zeros((templates.shape))

        for temp_idx in range(templates.shape[0]):
            unwhitened_temps[temp_idx, :, :] = np.dot(np.ascontiguousarray(templates[temp_idx, :, :]),
                                                      np.ascontiguousarray(unwhitening_mat))

        try:
            cluster_ids, cluster_quality = read_cluster_group_tsv(os.path.join(folder, 'cluster_group.tsv'))
        except OSError:
            cluster_ids = np.unique(spike_clusters)
            cluster_quality = ['unsorted'] * cluster_ids.size

        if not include_pcs:
            return spike_times, spike_clusters, spike_templates, amplitudes, unwhitened_temps, channel_map, cluster_ids, cluster_quality
        else:
            return spike_times, spike_clusters, spike_templates, amplitudes, unwhitened_temps, channel_map, cluster_ids, cluster_quality, pc_features, pc_feature_ind


class Wrappers:
    @staticmethod
    def calculate_isi_violations(spike_times, spike_clusters, isi_threshold, min_isi):
        cluster_ids = np.unique(spike_clusters)

        viol_rates = []
        viol_ns = []

        for idx, cluster_id in enumerate(cluster_ids):
            for_this_cluster = (spike_clusters == cluster_id)
            viol_rate, viol_n = QualityMetrics.isi_violations(spike_times[for_this_cluster],
                                                              min_time=np.min(spike_times),
                                                              max_time=np.max(spike_times),
                                                              isi_threshold=isi_threshold,
                                                              min_isi=min_isi)
            viol_rates.append(viol_rate)
            viol_ns.append(viol_n)

        return np.array(viol_rates), np.array(viol_ns)

    @staticmethod
    def calculate_presence_ratio(spike_times, spike_clusters):
        """

        :param spike_times:
        :param spike_clusters:
        :param total_units:
        :return:
        """
        cluster_ids = np.unique(spike_clusters)

        ratios = []

        for idx, cluster_id in enumerate(cluster_ids):
            for_this_cluster = (spike_clusters == cluster_id)
            ratios.append(QualityMetrics.presence_ratio(spike_times[for_this_cluster],
                                                        min_time=np.min(spike_times),
                                                        max_time=np.max(spike_times)))

        return np.array(ratios)

    @staticmethod
    def calculate_firing_rate(spike_times, spike_clusters):
        """

        :param spike_times:
        :param spike_clusters:
        :param total_units:
        :return:
        """
        cluster_ids = np.unique(spike_clusters)
        firing_rates = []

        for idx, cluster_id in enumerate(cluster_ids):
            for_this_cluster = (spike_clusters == cluster_id)
            firing_rates.append(QualityMetrics.firing_rate(spike_times[for_this_cluster],
                                                           min_time=np.min(spike_times),
                                                           max_time=np.max(spike_times)))

        return np.array(firing_rates)

    @staticmethod
    def calculate_amplitude_cutoff(spike_clusters, amplitudes):
        """

        :param spike_clusters:
        :param amplitudes:
        :param total_units:
        :return:
        """
        cluster_ids = np.unique(spike_clusters)

        amplitude_cutoffs = []

        for idx, cluster_id in enumerate(cluster_ids):
            for_this_cluster = (spike_clusters == cluster_id)
            amplitude_cutoffs.append(QualityMetrics.amplitude_cutoff(amplitudes[for_this_cluster]))

        return np.array(amplitude_cutoffs)

    @staticmethod
    def calculate_pc_metrics(spike_clusters,
                             spike_templates,
                             total_units,
                             pc_features,
                             pc_feature_ind,
                             num_channels_to_compare,
                             max_spikes_for_cluster,
                             max_spikes_for_nn,
                             n_neighbors,
                             do_parallel=True):
        """

        :param spike_clusters:
        :param total_units:
        :param pc_features:
        :param pc_feature_ind:
        :param num_channels_to_compare:
        :param max_spikes_for_cluster:
        :param max_spikes_for_nn:
        :param n_neighbors:
        :return:
        """

        assert (num_channels_to_compare % 2 == 1)
        half_spread = int((num_channels_to_compare - 1) / 2)

        cluster_ids = np.unique(spike_clusters)

        peak_channels = np.zeros((cluster_ids.max() + 1,), dtype='uint16')
        for cluster_id in cluster_ids:
            for_unit = np.squeeze(spike_clusters == cluster_id)
            pc_max = np.argmax(np.mean(pc_features[for_unit, 0, :], 0))
            peak_channels[cluster_id] = pc_feature_ind[cluster_id, pc_max]

        # Loop over clusters:
        if do_parallel:
            from joblib import Parallel, delayed
            # from joblib import wrap_non_picklable_objects
            # @delayed
            # @wrap_non_picklable_objects
            # def calculate_pc_metrics_one_cluster(**args):
            #     meas = Wrappers.calculate_pc_metrics_one_cluster(**args)
            #     return meas

            meas = Parallel(n_jobs=-1, verbose=3)(  # -1 means use all cores
                delayed(Wrappers.calculate_pc_metrics_one_cluster)  # Function
                (peak_channels, cluster_id, half_spread, pc_features, pc_feature_ind,  # Arguments
                 spike_clusters, max_spikes_for_cluster, max_spikes_for_nn, n_neighbors
                 )
                for cluster_id in cluster_ids)  # Loop
        else:
            from tqdm import tqdm
            meas = [Wrappers.calculate_pc_metrics_one_cluster(  # Function
                peak_channels, cluster_id, half_spread, pc_features, pc_feature_ind, spike_clusters,  # Arguments
                max_spikes_for_cluster, max_spikes_for_nn, n_neighbors)
                for cluster_id in tqdm(cluster_ids, desc='Calculating isolation metrics')]  # Loop

        # Unpack:
        isolation_distances = []
        l_ratios = []
        d_primes = []
        nn_hit_rates = []
        nn_miss_rates = []
        for mea in meas:
            isolation_distance, d_prime, nn_miss_rate, nn_hit_rate, l_ratio = mea
            isolation_distances.append(isolation_distance)
            d_primes.append(d_prime)
            nn_miss_rates.append(nn_miss_rate)
            nn_hit_rates.append(nn_hit_rate)
            l_ratios.append(l_ratio)

        return (np.array(isolation_distances), np.array(l_ratios), np.array(d_primes),
                np.array(nn_hit_rates), np.array(nn_miss_rates))

    @staticmethod
    def calculate_pc_metrics_one_cluster(peak_channels, cluster_id, half_spread, pc_features, pc_feature_ind,
                                         spike_clusters, max_spikes_for_cluster, max_spikes_for_nn, n_neighbors):

        # HELPERS:
        def make_index_mask(spike_clusters, unit_id, min_num, max_num):
            """ Create a mask for the spike index dimensions of the pc_features array
            Inputs:
            -------
            spike_clusters : numpy.ndarray (num_spikes x 0)
                Contains cluster IDs for all spikes in pc_features array
            unit_id : Int
                ID for this unit
            min_num : Int
                Minimum number of spikes to return; if there are not enough spikes for this unit, return all False
            max_num : Int
                Maximum number of spikes to return; if too many spikes for this unit, return a random subsample
            Output:
            -------
            index_mask : numpy.ndarray (boolean)
                Mask of spike indices for pc_features array
            """

            index_mask = spike_clusters == unit_id

            inds = np.where(index_mask)[0]

            if len(inds) < min_num:
                index_mask = np.zeros((spike_clusters.size,), dtype='bool')
            else:
                index_mask = np.zeros((spike_clusters.size,), dtype='bool')
                order = np.random.permutation(inds.size)
                index_mask[inds[order[:max_num]]] = True

            return index_mask

        def make_channel_mask(unit_id, pc_feature_ind, channels_to_use, these_inds=None):
            """ Create a mask for the channel dimension of the pc_features array
            Inputs:
            -------
            unit_id : Int
                ID for this unit
            pc_feature_ind : np.ndarray
                Channels used for PC calculation for each unit
            channels_to_use : np.ndarray
                Channels to use for calculating metrics
            Output:
            -------
            channel_mask : numpy.ndarray
                Channel indices to extract from pc_features array

            """
            if these_inds is None:
                these_inds = pc_feature_ind[unit_id, :]
            channel_mask = [np.argwhere(these_inds == i)[0][0] for i in channels_to_use]

            # channel_mask = [np.argwhere(these_inds == i)[0][0] for i in available_to_use]

            return np.array(channel_mask)

        def get_unit_pcs(these_pc_features, index_mask, channel_mask):
            """ Use the index_mask and channel_mask to return PC features for one unit
            Inputs:
            -------
            these_pc_features : numpy.ndarray (float)
                Array of pre-computed PC features (num_spikes x num_PCs x num_channels)
            index_mask : numpy.ndarray (boolean)
                Mask for spike index dimension of pc_features array
            channel_mask : numpy.ndarray (boolean)
                Mask for channel index dimension of pc_features array
            Output:
            -------
            unit_PCs : numpy.ndarray (float)
                PCs for one unit (num_spikes x num_PCs x num_channels)
            """

            unit_PCs = these_pc_features[index_mask, :, :]

            unit_PCs = unit_PCs[:, :, channel_mask]

            return unit_PCs

        def features_intersect(pc_feature_ind, these_channels):
            """
            # Take only the channels that have calculated features out of the ones we are interested in:
            # This should reduce the occurence of 'except IndexError' below

            Args:
                these_channels: channels_to_use or units_for_channel

            Returns:
                channels_to_use: intersect of what's available in PCs and what's needed
            """
            intersect = set(pc_feature_ind[these_channels[0], :])  # Initialize
            for cluster_id2 in these_channels:
                # Make a running intersect of what is available and what is needed
                intersect = intersect & set(pc_feature_ind[cluster_id2, :])
            return np.array(list(intersect))

        # HELPERS OVER
        peak_channel = peak_channels[cluster_id]

        half_spread_down = peak_channel \
            if peak_channel < half_spread \
            else half_spread

        half_spread_up = np.max(pc_feature_ind) - peak_channel \
            if peak_channel + half_spread > np.max(pc_feature_ind) \
            else half_spread

        units_for_channel, channel_index = np.unravel_index(
            np.where(pc_feature_ind.flatten() == peak_channel)[0],
            pc_feature_ind.shape)

        # Skip peak_channels if they are not present in unit_list:
        units_for_channel = units_for_channel[np.in1d(units_for_channel,peak_channels)]

        units_in_range = (peak_channels[units_for_channel] >= peak_channel - half_spread_down) * \
                         (peak_channels[units_for_channel] <= peak_channel + half_spread_up)

        units_for_channel = units_for_channel[units_in_range]

        channels_to_use = np.arange(peak_channel - half_spread_down, peak_channel + half_spread_up + 1)

        # Use channels that are available in PCs:
        channels_to_use = features_intersect(pc_feature_ind, channels_to_use)
        # If this yields nothing, use units_for_channel:
        if len(channels_to_use) < 1:
            channels_to_use = features_intersect(pc_feature_ind, units_for_channel)

        spike_counts = np.zeros(units_for_channel.shape)

        for idx2, cluster_id2 in enumerate(units_for_channel):
            spike_counts[idx2] = np.sum(spike_clusters == cluster_id2)

        this_unit_idx = np.where(units_for_channel == cluster_id)[0]

        if spike_counts[this_unit_idx] > max_spikes_for_cluster:
            relative_counts = spike_counts / spike_counts[this_unit_idx] * max_spikes_for_cluster
        else:
            relative_counts = spike_counts

        all_pcs = np.zeros((0, pc_features.shape[1], channels_to_use.size))
        all_labels = np.zeros((0,))
        for idx2, cluster_id2 in enumerate(units_for_channel):

            try:
                channel_mask = make_channel_mask(cluster_id2, pc_feature_ind, channels_to_use)
            except IndexError:
                # Occurs when pc_feature_ind does not contain all channels of interest
                # In that case, we will exclude this unit for the calculation
                pass
            else:
                subsample = int(relative_counts[idx2])
                index_mask = make_index_mask(spike_clusters, cluster_id2, min_num=0, max_num=subsample)
                pcs = get_unit_pcs(pc_features, index_mask, channel_mask)
                labels = np.ones((pcs.shape[0],)) * cluster_id2

                all_pcs = np.concatenate((all_pcs, pcs), 0)
                all_labels = np.concatenate((all_labels, labels), 0)

        all_pcs = np.reshape(all_pcs, (all_pcs.shape[0], pc_features.shape[1] * channels_to_use.size))
        if ((all_pcs.shape[0] > 10)
                and (cluster_id in all_labels)
                and (len(channels_to_use) > 0)
            and not (all_labels== cluster_id).all()
        ):

            isolation_distance, l_ratio = QualityMetrics.mahalanobis_metrics(all_pcs, all_labels, cluster_id)

            d_prime = QualityMetrics.lda_metrics(all_pcs, all_labels, cluster_id)

            nn_hit_rate, nn_miss_rate = QualityMetrics.nearest_neighbors_metrics(all_pcs, all_labels,
                                                                                 cluster_id,
                                                                                 max_spikes_for_nn,
                                                                                 n_neighbors)
        else:  # Too few spikes or cluster doesnt exist
            isolation_distance = np.nan
            d_prime = np.nan
            nn_miss_rate = np.nan
            nn_hit_rate = np.nan
            l_ratio = np.nan
        return isolation_distance, d_prime, nn_miss_rate, nn_hit_rate, l_ratio

    @staticmethod
    def calculate_silhouette_score(spike_clusters,
                                   spike_templates,
                                   total_units,
                                   pc_features,
                                   pc_feature_ind,
                                   total_spikes,
                                   do_parallel=True):
        """

        :param spike_clusters:
        :param pc_features:
        :param pc_feature_ind:
        :param total_spikes:
        :return:
        """
        import warnings
        cluster_ids = np.unique(spike_clusters)
        random_spike_inds = np.random.permutation(spike_clusters.size)
        random_spike_inds = random_spike_inds[:total_spikes]
        num_pc_features = pc_features.shape[1]
        num_channels = np.max(pc_feature_ind) + 1
        all_pcs = np.zeros((total_spikes, num_channels * num_pc_features))

        for idx, i in enumerate(random_spike_inds):
            unit_id = spike_templates[i]
            channels = pc_feature_ind[unit_id, :]

            for j in range(0, num_pc_features):
                all_pcs[idx, channels + num_channels * j] = pc_features[i, j, :]

        cluster_labels = spike_clusters[random_spike_inds]

        SS = np.empty((total_units, total_units))
        SS[:] = np.nan
        """

    for idx1, i in enumerate(cluster_ids):
        for idx2, j in enumerate(cluster_ids):
            
            if j > i:
                inds = np.in1d(cluster_labels, np.array([i,j]))
                X = all_pcs[inds,:]
                labels = cluster_labels[inds]
                
                if len(labels) > 2 and len(np.unique(labels)) > 1:
                    SS[idx1,idx2] = silhouette_score(X, labels)                        
        """

        def score_inner_loop(i, cluster_ids):
            """
            Helper to loop over cluster_ids in one dimension. We dont want to loop over both dimensions in parallel-
            that will create too much worker overhead
            Args:
                i: index of first dimension
                cluster_ids: iterable of cluster ids

            Returns: scores for dimension j

            """
            scores_1d = []
            for j in cluster_ids:
                if j > i:
                    inds = np.in1d(cluster_labels, np.array([i, j]))
                    X = all_pcs[inds, :]
                    labels = cluster_labels[inds]
                    # len(np.unique(labels))=1 Can happen if total_spikes is low:
                    if (len(labels) > 2) and (len(np.unique(labels)) > 1):
                        scores_1d.append(silhouette_score(X, labels))
                    else:
                        scores_1d.append(np.nan)
                else:
                    scores_1d.append(np.nan)
            return scores_1d

        # Build lists
        if do_parallel:
            from joblib import Parallel, delayed
            scores = Parallel(n_jobs=-1, verbose=2)(delayed(score_inner_loop)(i, cluster_ids) for i in cluster_ids)
        else:
            scores = [score_inner_loop(i, cluster_ids) for i in cluster_ids]

        # Fill the 2d array
        for i, col_score in enumerate(scores):
            for j, one_score in enumerate(col_score):
                SS[i, j] = one_score

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = np.nanmin(SS, 0)
            b = np.nanmin(SS, 1)
        return np.array([np.nanmin([a, b]) for a, b in zip(a, b)])

    @staticmethod
    def calculate_drift_metrics(spike_times,
                                spike_clusters,
                                spike_templates,
                                pc_features,
                                pc_feature_ind,
                                interval_length,
                                min_spikes_per_interval,
                                do_parallel=True):
        """

        :param spike_times:
        :param spike_clusters:
        :param total_units:
        :param pc_features:
        :param pc_feature_ind:
        :param interval_length:
        :param min_spikes_per_interval:
        :return:
        """

        def get_spike_depths(spike_clusters, pc_features, pc_feature_ind):

            """
            Calculates the distance (in microns) of individual spikes from the probe tip
            This implementation is based on Matlab code from github.com/cortex-lab/spikes
            Input:
            -----
            spike_clusters : numpy.ndarray (N x 0)
                Cluster IDs for N spikes
            pc_features : numpy.ndarray (N x channels x num_PCs)
                PC features for each spike
            pc_feature_ind  : numpy.ndarray (M x channels)
                Channels used for PC calculation for each unit
            Output:
            ------
            spike_depths : numpy.ndarray (N x 0)
                Distance (in microns) from each spike waveform from the probe tip
            """

            pc_features_copy = np.copy(pc_features)
            pc_features_copy = np.squeeze(pc_features_copy[:, 0, :])
            pc_features_copy[pc_features_copy < 0] = 0
            pc_power = pow(pc_features_copy, 2)

            spike_feat_ind = pc_feature_ind[spike_clusters, :]
            spike_depths = np.sum(spike_feat_ind * pc_power, 1) / np.sum(pc_power, 1)

            return spike_depths * 10

        def calc_one_cluster(cluster_id):
            """
            Helper to calculate drift for one cluster
            Args:
                cluster_id:

            Returns:
                max_drift, cumulative_drift
            """
            in_cluster = spike_clusters == cluster_id
            times_for_cluster = spike_times[in_cluster]
            depths_for_cluster = depths[in_cluster]

            median_depths = []

            for t1, t2 in zip(interval_starts, interval_ends):

                in_range = (times_for_cluster > t1) * (times_for_cluster < t2)

                if np.sum(in_range) >= min_spikes_per_interval:
                    median_depths.append(np.median(depths_for_cluster[in_range]))
                else:
                    median_depths.append(np.nan)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # RuntimeWarning: All-NaN slice encountered
                median_depths = np.array(median_depths)
                max_drift = np.around(np.nanmax(median_depths) - np.nanmin(median_depths), 2)
                cumulative_drift = np.around(np.nansum(np.abs(np.diff(median_depths))), 2)
            return max_drift, cumulative_drift

        max_drifts = []
        cumulative_drifts = []

        depths = get_spike_depths(spike_templates, pc_features, pc_feature_ind)

        interval_starts = np.arange(np.min(spike_times), np.max(spike_times), interval_length)
        interval_ends = interval_starts + interval_length

        cluster_ids = np.unique(spike_clusters)

        if do_parallel:
            from joblib import Parallel, delayed
            meas = Parallel(n_jobs=-1, verbose=2)(delayed(calc_one_cluster)(cluster_id)
                                                  for cluster_id in cluster_ids)
        else:
            meas = [calc_one_cluster(cluster_id) for cluster_id in cluster_ids]

        for max_drift, cumulative_drift in meas:
            max_drifts.append(max_drift)
            cumulative_drifts.append(max_drift)
        return np.array(max_drifts), np.array(cumulative_drifts)


class QualityMetrics:
    @staticmethod
    def isi_violations(spike_train, min_time, max_time, isi_threshold, min_isi=0):
        """Calculate ISI violations for a spike train.
        Based on metric described in Hill et al. (2011) J Neurosci 31: 8699-8705
        modified by Dan Denman from cortex-lab/sortingQuality GitHub by Nick Steinmetz
        Inputs:
        -------
        spike_train : array of spike times
        min_time : minimum time for potential spikes
        max_time : maximum time for potential spikes
        isi_threshold : threshold for isi violation
        min_isi : threshold for duplicate spikes
        Outputs:
        --------
        fpRate : rate of contaminating spikes as a fraction of overall rate
            A perfect unit has a fpRate = 0
            A unit with some contamination has a fpRate < 0.5
            A unit with lots of contamination has a fpRate > 1.0
        num_violations : total number of violations
        """
        duplicate_spikes = np.where(np.diff(spike_train) <= min_isi)[0]

        spike_train = np.delete(spike_train, duplicate_spikes + 1)
        isis = np.diff(spike_train)

        num_spikes = len(spike_train)
        num_violations = sum(isis < isi_threshold)
        violation_time = 2 * num_spikes * (isi_threshold - min_isi)
        total_rate = QualityMetrics.firing_rate(spike_train, min_time, max_time)
        violation_rate = num_violations / violation_time
        fpRate = violation_rate / total_rate

        return fpRate, num_violations

    @staticmethod
    def presence_ratio(spike_train, min_time, max_time, num_bins=100):
        """Calculate fraction of time the unit is present within an epoch.
        Inputs:
        -------
        spike_train : array of spike times
        min_time : minimum time for potential spikes
        max_time : maximum time for potential spikes
        Outputs:
        --------
        presence_ratio : fraction of time bins in which this unit is spiking
        """

        h, b = np.histogram(spike_train, np.linspace(min_time, max_time, num_bins))

        return np.sum(h > 0) / num_bins

    @staticmethod
    def firing_rate(spike_train, min_time=None, max_time=None):
        """Calculate firing rate for a spike train.
        If no temporal bounds are specified, the first and last spike time are used.
        Inputs:
        -------
        spike_train : numpy.ndarray
            Array of spike times in seconds
        min_time : float
            Time of first possible spike (optional)
        max_time : float
            Time of last possible spike (optional)
        Outputs:
        --------
        fr : float
            Firing rate in Hz
        """

        if min_time is not None and max_time is not None:
            duration = max_time - min_time
        else:
            duration = np.max(spike_train) - np.min(spike_train)

        fr = spike_train.size / duration

        return fr

    @staticmethod
    def amplitude_cutoff(amplitudes, num_histogram_bins=500, histogram_smoothing_value=3):
        """ Calculate approximate fraction of spikes missing from a distribution of amplitudes
        Assumes the amplitude histogram is symmetric (not valid in the presence of drift)
        Inspired by metric described in Hill et al. (2011) J Neurosci 31: 8699-8705
        Input:
        ------
        amplitudes : numpy.ndarray
            Array of amplitudes (don't need to be in physical units)
        Output:
        -------
        fraction_missing : float
            Fraction of missing spikes (0-0.5)
            If more than 50% of spikes are missing, an accurate estimate isn't possible
        """

        h, b = np.histogram(amplitudes, num_histogram_bins, density=True)

        pdf = gaussian_filter1d(h, histogram_smoothing_value)
        support = b[:-1]

        peak_index = np.argmax(pdf)
        G = np.argmin(np.abs(pdf[peak_index:] - pdf[0])) + peak_index

        bin_size = np.mean(np.diff(support))
        fraction_missing = np.sum(pdf[G:]) * bin_size

        fraction_missing = np.min([fraction_missing, 0.5])

        return fraction_missing

    @staticmethod
    def mahalanobis_metrics(all_pcs, all_labels, this_unit_id):
        # def mahalanobis_metrics(pcs_for_this_unit, pcs_for_other_units):
        """ Calculates isolation distance and L-ratio (metrics computed from Mahalanobis distance)
        Based on metrics described in Schmitzer-Torbert et al. (2005) Neurosci 131: 1-11
        Inputs:
        -------
        all_pcs : numpy.ndarray (num_spikes x PCs)
            2D array of PCs for all spikes
        all_labels : numpy.ndarray (num_spikes x 0)
            1D array of cluster labels for all spikes
        this_unit_id : Int
            number corresponding to unit for which these metrics will be calculated
        Outputs:
        --------
        isolation_distance : float
            Isolation distance of this unit
        l_ratio : float
            L-ratio for this unit
        """

        pcs_for_this_unit = all_pcs[all_labels == this_unit_id, :]
        pcs_for_other_units = all_pcs[all_labels != this_unit_id, :]
        mean_value = np.expand_dims(np.mean(pcs_for_this_unit, 0), 0)

        try:
            VI = np.linalg.inv(np.cov(pcs_for_this_unit.T))
        except np.linalg.linalg.LinAlgError:  # case of singular matrix
            return np.nan, np.nan

        mahalanobis_other = np.sort(cdist(mean_value,
                                          pcs_for_other_units,
                                          'mahalanobis', VI=VI)[0])

        mahalanobis_self = np.sort(cdist(mean_value,
                                         pcs_for_this_unit,
                                         'mahalanobis', VI=VI)[0])

        n = np.min([pcs_for_this_unit.shape[0], pcs_for_other_units.shape[0]])  # number of spikes

        if n >= 2:

            dof = pcs_for_this_unit.shape[1]  # number of features

            l_ratio = np.sum(1 - chi2.cdf(pow(mahalanobis_other, 2), dof)) / mahalanobis_other.shape[0]
            isolation_distance = pow(mahalanobis_other[n - 1], 2)

        else:
            l_ratio = np.nan
            isolation_distance = np.nan

        return isolation_distance, l_ratio

    @staticmethod
    def lda_metrics(all_pcs, all_labels, this_unit_id):
        """ Calculates d-prime based on Linear Discriminant Analysis
        Based on metric described in Hill et al. (2011) J Neurosci 31: 8699-8705
        Inputs:
        -------
        all_pcs : numpy.ndarray (num_spikes x PCs)
            2D array of PCs for all spikes
        all_labels : numpy.ndarray (num_spikes x 0)
            1D array of cluster labels for all spikes
        this_unit_id : Int
            number corresponding to unit for which these metrics will be calculated
        Outputs:
        --------
        d_prime : float
            Isolation distance of this unit
        l_ratio : float
            L-ratio for this unit
        """

        X = all_pcs

        y = np.zeros((X.shape[0],), dtype='bool')
        y[all_labels == this_unit_id] = True

        lda = LDA(n_components=1)

        X_flda = lda.fit_transform(X, y)

        flda_this_cluster = X_flda[np.where(y)[0]]
        flda_other_cluster = X_flda[np.where(np.invert(y))[0]]

        d_prime = (np.mean(flda_this_cluster) - np.mean(flda_other_cluster)) / np.sqrt(
            0.5 * (np.std(flda_this_cluster) ** 2 + np.std(flda_other_cluster) ** 2))

        return d_prime

    @staticmethod
    def nearest_neighbors_metrics(all_pcs, all_labels, this_unit_id, max_spikes_for_nn, n_neighbors):
        """ Calculates unit contamination based on NearestNeighbors search in PCA space
        Based on metrics described in Chung, Magland et al. (2017) Neuron 95: 1381-1394
        Inputs:
        -------
        all_pcs : numpy.ndarray (num_spikes x PCs)
            2D array of PCs for all spikes
        all_labels : numpy.ndarray (num_spikes x 0)
            1D array of cluster labels for all spikes
        this_unit_id : Int
            number corresponding to unit for which these metrics will be calculated
        max_spikes_for_nn : Int
            number of spikes to use (calculation can be very slow when this number is >20000)
        n_neighbors : Int
            number of neighbors to use
        Outputs:
        --------
        hit_rate : float
            Fraction of neighbors for target cluster that are also in target cluster
        miss_rate : float
            Fraction of neighbors outside target cluster that are in target cluster
        """

        total_spikes = all_pcs.shape[0]
        ratio = max_spikes_for_nn / total_spikes
        this_unit = all_labels == this_unit_id

        X = np.concatenate((all_pcs[this_unit, :], all_pcs[np.invert(this_unit), :]), 0)

        n = np.sum(this_unit)

        if ratio < 1:
            inds = np.arange(0, X.shape[0] - 1, 1 / ratio).astype('int')
            X = X[inds, :]
            n = int(n * ratio)

        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(X)
        distances, indices = nbrs.kneighbors(X)

        this_cluster_inds = np.arange(n)

        this_cluster_nearest = indices[:n, 1:].flatten()
        other_cluster_nearest = indices[n:, 1:].flatten()

        hit_rate = np.mean(this_cluster_nearest < n)
        miss_rate = np.mean(other_cluster_nearest < n)

        return hit_rate, miss_rate


class NewWrappers:
    @staticmethod
    def calculate_pc_metrics(spike_clusters,
                                 spike_templates,
                                 total_units,
                                 pc_features,
                                 pc_feature_ind,
                                 num_channels_to_compare,
                                 max_spikes_for_cluster,
                                 max_spikes_for_nn,
                                 n_neighbors,
                                 do_parallel=True):
        """

        :param spike_clusters:
        :param total_units:
        :param pc_features:
        :param pc_feature_ind:
        :param num_channels_to_compare:
        :param max_spikes_for_cluster:
        :param max_spikes_for_nn:
        :param n_neighbors:
        :return:
        """

        assert (num_channels_to_compare % 2 == 1)
        half_spread = int((num_channels_to_compare - 1) / 2)

        cluster_ids = np.unique(spike_clusters)
        template_ids = np.unique(spike_templates)

        template_peak_channels = np.zeros((len(template_ids),), dtype='uint16')
        cluster_peak_channels = np.zeros((len(cluster_ids),), dtype='uint16')

        for idx, template_id in enumerate(template_ids):
            for_template = np.squeeze(spike_templates == template_id)
            pc_max = np.argmax(np.mean(pc_features[for_template, 0, :], 0))
            template_peak_channels[idx] = pc_feature_ind[template_id, pc_max]

        for idx, cluster_id in enumerate(cluster_ids):
            for_unit = np.squeeze(spike_clusters == cluster_id)
            templates_for_unit = np.unique(spike_templates[for_unit])
            template_positions = np.where(np.isin(template_ids, templates_for_unit))[0]
            cluster_peak_channels[idx] = np.median(template_peak_channels[template_positions])

        # Loop over clusters:
        if do_parallel:
            from joblib import Parallel, delayed
            meas = Parallel(n_jobs=-1, verbose=3)(  # -1 means use all cores
                # delayed(Wrappers.calculate_pc_metrics_one_cluster_old)  # Function
                # (template_peak_channels, cluster_id, half_spread, pc_features, pc_feature_ind, spike_clusters,  # Inputs
                #  max_spikes_for_cluster, max_spikes_for_nn, n_neighbors)
                delayed(NewWrappers.calculate_pc_metrics_one_cluster)  # Function
                (cluster_peak_channels, idx, cluster_id, cluster_ids,
                 half_spread, pc_features, pc_feature_ind,
                 spike_clusters, spike_templates,
                 max_spikes_for_cluster, max_spikes_for_nn, n_neighbors)
                for idx, cluster_id in enumerate(cluster_ids))  # Loop
        else:
            from tqdm import tqdm
            meas = []
            for idx, cluster_id in tqdm(enumerate(cluster_ids), total=cluster_ids.max(), desc='PC metrics'):  # Loop
                # meas.append(Wrappers.calculate_pc_metrics_one_cluster_old(
                #     template_peak_channels, cluster_id, half_spread, pc_features, pc_feature_ind, spike_clusters,
                #     max_spikes_for_cluster, max_spikes_for_nn, n_neighbors))
                meas.append(NewWrappers.calculate_pc_metrics_one_cluster(  # Function
                    cluster_peak_channels, idx, cluster_id, cluster_ids,
                    half_spread, pc_features, pc_feature_ind,
                    spike_clusters, spike_templates,
                    max_spikes_for_cluster, max_spikes_for_nn, n_neighbors))

        # Unpack:
        isolation_distances = []
        l_ratios = []
        d_primes = []
        nn_hit_rates = []
        nn_miss_rates = []
        for mea in meas:
            isolation_distance, d_prime, nn_miss_rate, nn_hit_rate, l_ratio = mea
            isolation_distances.append(isolation_distance)
            d_primes.append(d_prime)
            nn_miss_rates.append(nn_miss_rate)
            nn_hit_rates.append(nn_hit_rate)
            l_ratios.append(l_ratio)

        return (np.array(isolation_distances), np.array(l_ratios), np.array(d_primes),
                np.array(nn_hit_rates), np.array(nn_miss_rates))

    @staticmethod
    def calculate_pc_metrics_one_cluster(cluster_peak_channels, idx, cluster_id, cluster_ids,
                                         half_spread, pc_features, pc_feature_ind,
                                         spike_clusters, spike_templates,
                                         max_spikes_for_cluster, max_spikes_for_nn, n_neighbors):

        # def make_index_mask(spike_clusters, unit_id, min_num, max_num):
        #     """ Create a mask for the spike index dimensions of the pc_features array
        #     Inputs:
        #     -------
        #     spike_clusters : numpy.ndarray (num_spikes x 0)
        #         Contains cluster IDs for all spikes in pc_features array
        #     unit_id : Int
        #         ID for this unit
        #     min_num : Int
        #         Minimum number of spikes to return; if there are not enough spikes for this unit, return all False
        #     max_num : Int
        #         Maximum number of spikes to return; if too many spikes for this unit, return a random subsample
        #     Output:
        #     -------
        #     index_mask : numpy.ndarray (boolean)
        #         Mask of spike indices for pc_features array
        #     """
        #
        #     index_mask = spike_clusters == unit_id
        #
        #     inds = np.where(index_mask)[0]
        #
        #     if len(inds) < min_num:
        #         index_mask = np.zeros((spike_clusters.size,), dtype='bool')
        #     else:
        #         index_mask = np.zeros((spike_clusters.size,), dtype='bool')
        #         order = np.random.permutation(inds.size)
        #         index_mask[inds[order[:max_num]]] = True
        #
        #     return index_mask
        #
        # def make_channel_mask(unit_id, pc_feature_ind, channels_to_use, these_inds=None):
        #     """ Create a mask for the channel dimension of the pc_features array
        #     Inputs:
        #     -------
        #     unit_id : Int
        #         ID for this unit
        #     pc_feature_ind : np.ndarray
        #         Channels used for PC calculation for each unit
        #     channels_to_use : np.ndarray
        #         Channels to use for calculating metrics
        #     Output:
        #     -------
        #     channel_mask : numpy.ndarray
        #         Channel indices to extract from pc_features array
        #
        #     """
        #     if these_inds is None:
        #         these_inds = pc_feature_ind[unit_id, :]
        #     channel_mask = [np.argwhere(these_inds == i)[0][0] for i in channels_to_use]
        #
        #     # channel_mask = [np.argwhere(these_inds == i)[0][0] for i in available_to_use]
        #
        #     return np.array(channel_mask)
        #
        # def get_unit_pcs_old(these_pc_features, index_mask, channel_mask):
        #     """ Use the index_mask and channel_mask to return PC features for one unit
        #     Inputs:
        #     -------
        #     these_pc_features : numpy.ndarray (float)
        #         Array of pre-computed PC features (num_spikes x num_PCs x num_channels)
        #     index_mask : numpy.ndarray (boolean)
        #         Mask for spike index dimension of pc_features array
        #     channel_mask : numpy.ndarray (boolean)
        #         Mask for channel index dimension of pc_features array
        #     Output:
        #     -------
        #     unit_PCs : numpy.ndarray (float)
        #         PCs for one unit (num_spikes x num_PCs x num_channels)
        #     """
        #
        #     unit_PCs = these_pc_features[index_mask, :, :]
        #
        #     unit_PCs = unit_PCs[:, :, channel_mask]
        #
        #     return unit_PCs

        def get_unit_pcs(unit_id,
                                            spike_clusters,
                                            spike_templates,
                                            pc_feature_ind,
                                            pc_features,
                                            channels_to_use,
                                            subsample):

            """ Return PC features for one unit

            Inputs:
            -------
            unit_id : Int
                ID for this unit
            spike_clusters : np.ndarray
                Cluster labels for each spike
            spike_templates : np.ndarry
                Template labels for each spike
            pc_feature_ind : np.ndarray
                Channels used for PC calculation for each unit
            pc_features : np.ndarray
                Array of all PC features
            channels_to_use : np.ndarray
                Channels to use for calculating metrics
            subsample : Int
                maximum number of spikes to return

            Output:
            -------
            unit_PCs : numpy.ndarray (float)
                PCs for one unit (num_spikes x num_PCs x num_channels)

            """

            inds_for_unit = np.where(spike_clusters == unit_id)[0]

            spikes_to_use = np.random.permutation(inds_for_unit)[:subsample]

            unique_template_ids = np.unique(spike_templates[spikes_to_use])

            unit_PCs = []

            for template_id in unique_template_ids:

                index_mask = spikes_to_use[np.squeeze(spike_templates[spikes_to_use]) == template_id]
                these_inds = pc_feature_ind[template_id, :]

                pc_array = []

                for i in channels_to_use:

                    if np.isin(i, these_inds):
                        channel_index = np.argwhere(these_inds == i)[0][0]
                        pc_array.append(pc_features[index_mask, :, channel_index])
                    else:
                        return None

                unit_PCs.append(np.stack(pc_array, axis=-1))

            if len(unit_PCs) > 0:

                return np.concatenate(unit_PCs)
            else:
                return None

        def features_intersect(pc_feature_ind, these_channels):
            """
            # Take only the channels that have calculated features out of the ones we are interested in:
            # This should reduce the occurence of 'except IndexError' below

            Args:
                these_channels: channels_to_use or units_for_channel

            Returns:
                channels_to_use: intersect of what's available in PCs and what's needed
            """
            intersect = set(channels_to_use)  # Initialize
            for cluster_id2 in these_channels:
                # Make a running intersect of what is available and what is needed
                intersect = intersect & set(pc_feature_ind[cluster_id2, :])
            return np.array(list(intersect))

        peak_channel = cluster_peak_channels[idx]
        num_spikes_in_cluster = np.sum(spike_clusters == cluster_id)

        half_spread_down = peak_channel \
            if peak_channel < half_spread \
            else half_spread

        half_spread_up = np.max(pc_feature_ind) - peak_channel \
            if peak_channel + half_spread > np.max(pc_feature_ind) \
            else half_spread

        channels_to_use = np.arange(peak_channel - half_spread_down, peak_channel + half_spread_up + 1)
        units_in_range = cluster_ids[np.isin(cluster_peak_channels, channels_to_use)]

        # Use channels that are available in PCs:
        channels_to_use = features_intersect(pc_feature_ind, channels_to_use)
        # If this yields nothing, use units_for_channel:
        if len(channels_to_use) < 1:
            channels_to_use = features_intersect(pc_feature_ind, units_in_range)

        spike_counts = np.zeros(units_in_range.shape)

        for idx2, cluster_id2 in enumerate(units_in_range):
            spike_counts[idx2] = np.sum(spike_clusters == cluster_id2)

        if num_spikes_in_cluster > max_spikes_for_cluster:
            relative_counts = spike_counts / num_spikes_in_cluster * max_spikes_for_cluster
        else:
            relative_counts = spike_counts

        all_pcs = np.zeros((0, pc_features.shape[1], channels_to_use.size))
        all_labels = np.zeros((0,))

        # for idx2, cluster_id2 in enumerate(units_in_range):
        #
        #     try:
        #         channel_mask = make_channel_mask(cluster_id2, pc_feature_ind, channels_to_use)
        #     except IndexError:
        #         # Occurs when pc_feature_ind does not contain all channels of interest
        #         # In that case, we will exclude this unit for the calculation
        #         pass
        #     else:
        #         subsample = int(relative_counts[idx2])
        #         index_mask = make_index_mask(spike_clusters, cluster_id2, min_num=0, max_num=subsample)
        #         pcs = get_unit_pcs(pc_features, index_mask, channel_mask)
        #         # pcs = get_unit_pcs(cluster_id2, spike_clusters, spike_templates,
        #         #                                           pc_feature_ind, pc_features, channels_to_use,
        #         #                                           subsample)
        #         labels = np.ones((pcs.shape[0],)) * cluster_id2
        #
        #         all_pcs = np.concatenate((all_pcs, pcs), 0)
        #         all_labels = np.concatenate((all_labels, labels), 0)

        # New Allen implementation still misses neurons that are not on many channels, eg stereotrodes
        for idx2, cluster_id2 in enumerate(units_in_range):

            subsample = int(relative_counts[idx2])

            pcs = get_unit_pcs(cluster_id2, spike_clusters, spike_templates,
                               pc_feature_ind, pc_features, channels_to_use,
                               subsample)

            if pcs is not None and len(pcs.shape) == 3:
                labels = np.ones((pcs.shape[0],)) * cluster_id2

                all_pcs = np.concatenate((all_pcs, pcs), 0)
                all_labels = np.concatenate((all_labels, labels), 0)

        all_pcs = np.reshape(all_pcs, (all_pcs.shape[0], pc_features.shape[1] * channels_to_use.size))
        if ((all_pcs.shape[0] > 10)
                and (cluster_id in all_labels)
                and (len(channels_to_use) > 0)):
            isolation_distance, l_ratio = QualityMetrics.mahalanobis_metrics(all_pcs, all_labels, cluster_id)

            d_prime = QualityMetrics.lda_metrics(all_pcs, all_labels, cluster_id)

            nn_hit_rate, nn_miss_rate = QualityMetrics.nearest_neighbors_metrics(all_pcs, all_labels,
                                                                                 cluster_id,
                                                                                 max_spikes_for_nn,
                                                                                 n_neighbors)
        else:  # Too few spikes or cluster doesnt exist
            isolation_distance = np.nan
            d_prime = np.nan
            nn_miss_rate = np.nan
            nn_hit_rate = np.nan
            l_ratio = np.nan
        return isolation_distance, d_prime, nn_miss_rate, nn_hit_rate, l_ratio


def calculate_metrics(spike_times, spike_clusters, spike_templates, amplitudes, pc_features, pc_feature_ind,
                      output_folder=None,
                      do_parallel=True, do_pc_features=True, do_silhouette=True, do_drift=True,
                      isi_threshold=0.0015,
                      min_isi=0.000166,
                      num_channels_to_compare=5,
                      max_spikes_for_unit=1500,
                      max_spikes_for_nn=20000,
                      n_neighbors=4,
                      n_silhouette=20000,
                      drift_metrics_interval_s=51,
                      drift_metrics_min_spikes_per_interval=10
                      ):
    """ Calculate metrics for all units on one probe
    from mmy.input_output import spike_io
    ksort_folder = '~/res_ss_full/res_ss/tcloop_train_m022_1553627381_'
    spike_times, spike_clusters, spike_templates, amplitudes, templates, channel_map, clusterIDs, cluster_quality, pc_features, pc_feature_ind = \
        spike_io.QualityMetrics.load_kilosort_data(ksort_folder, 3e4, False, include_pcs=True)
    metrics = QualityMetrics.calculate_metrics(spike_times, spike_clusters, amplitudes, pc_features, pc_feature_ind, ksort_folder)



    Inputs:
    ------
    spike_times : numpy.ndarray (num_spikes x 0)
        Spike times in seconds (same timebase as epochs)
    spike_clusters : numpy.ndarray (num_spikes x 0)
        Cluster IDs for each spike time
    amplitudes : numpy.ndarray (num_spikes x 0)
        Amplitude value for each spike time
    channel_map : numpy.ndarray (num_units x 0)
        Original data channel for pc_feature_ind array
    pc_features : numpy.ndarray (num_spikes x num_pcs x num_channels)
        Pre-computed PCs for blocks of channels around each spike
    pc_feature_ind : numpy.ndarray (num_units x num_channels)
        Channel indices of PCs for each unit
    epochs : list of Epoch objects
        contains information on Epoch start and stop times
    params : dict of parameters
        'isi_threshold' : minimum time for isi violations

    Outputs:
    --------
    metrics : pandas.DataFrame
        one column for each metric
        one row per unit per epoch
    """

    # ==========================================================
    # MAIN:
    # ==========================================================

    cluster_ids = np.unique(spike_clusters)
    total_units = len(np.unique(spike_clusters))
    print("Calculating isi violations")
    print(spike_clusters)
    print(total_units)
    isi_viol_rate, isi_viol_n = Wrappers.calculate_isi_violations(spike_times, spike_clusters, isi_threshold, min_isi)

    print("Calculating presence ratio")
    presence_ratio = Wrappers.calculate_presence_ratio(spike_times, spike_clusters, )

    print("Calculating firing rate")
    firing_rate = Wrappers.calculate_firing_rate(spike_times, spike_clusters, )

    print("Calculating amplitude cutoff")
    amplitude_cutoff = Wrappers.calculate_amplitude_cutoff(spike_clusters, amplitudes, )
    metrics = pd.DataFrame(data=OrderedDict((('cluster_id', cluster_ids),
                                             ('firing_rate', firing_rate),
                                             ('presence_ratio', presence_ratio),
                                             ('isi_viol_rate', isi_viol_rate),
                                             ('isi_viol_n', isi_viol_n),
                                             ('amplitude_cutoff', amplitude_cutoff),)))
    if do_pc_features:
        print("Calculating PC-based metrics")
        try:
            (isolation_distance, l_ratio,
            d_prime, nn_hit_rate, nn_miss_rate) = NewWrappers.calculate_pc_metrics(spike_clusters,
                                                                             spike_templates,
                                                                             total_units,
                                                                             pc_features,
                                                                             pc_feature_ind,
                                                                             num_channels_to_compare,
                                                                             max_spikes_for_unit,
                                                                             max_spikes_for_nn,
                                                                             n_neighbors,
                                                                             do_parallel=do_parallel)
        except Exception:
            # Fallback
            print("Falling back to old Allen algo")
            (isolation_distance, l_ratio,
             d_prime, nn_hit_rate, nn_miss_rate) = Wrappers.calculate_pc_metrics(spike_clusters,
                                                                             spike_templates,
                                                                             total_units,
                                                                             pc_features,
                                                                             pc_feature_ind,
                                                                             num_channels_to_compare,
                                                                             max_spikes_for_unit,
                                                                             max_spikes_for_nn,
                                                                             n_neighbors,
                                                                             do_parallel=do_parallel)

        metrics0 = pd.DataFrame(data=OrderedDict((('isolation_distance', isolation_distance),
                                                  ('l_ratio', l_ratio),
                                                  ('d_prime', d_prime),
                                                  ('nn_hit_rate', nn_hit_rate),
                                                  ('nn_miss_rate', nn_miss_rate)
                                                  )))
        metrics = pd.concat([metrics, metrics0], axis=1)
    if do_silhouette:
        print("Calculating silhouette score")
        the_silhouette_score = Wrappers.calculate_silhouette_score(spike_clusters,
                                                                   spike_templates,
                                                                   total_units,
                                                                   pc_features,
                                                                   pc_feature_ind,
                                                                   n_silhouette,
                                                                   do_parallel=True)
        metrics2 = pd.DataFrame(data=OrderedDict((('silhouette_score', the_silhouette_score),)),
                                index=range(len(the_silhouette_score)))

        metrics = pd.concat([metrics, metrics2], axis=1)
    if do_drift:
        print("Calculating drift metrics")
        # TODO [in_epoch] has to go. Need to modify loading function
        max_drift, cumulative_drift = Wrappers.calculate_drift_metrics(spike_times,
                                                                       spike_clusters,
                                                                       spike_templates,
                                                                       pc_features,
                                                                       pc_feature_ind,
                                                                       drift_metrics_interval_s,
                                                                       drift_metrics_min_spikes_per_interval,
                                                                       do_parallel=do_parallel)

        metrics3 = pd.DataFrame(data=OrderedDict((('max_drift', max_drift),
                                                  ('cumulative_drift', cumulative_drift),
                                                  )))
        metrics = pd.concat([metrics, metrics3], axis=1)
    # write to output file if requested
    if output_folder is not None:
        metrics.to_csv(os.path.join(output_folder, 'quality_metrics.csv'), index=False)

    return metrics


@click.command()
@click.option('--kilosort_folder', default=None, help='kilosort_folder to read from and write to')
@click.option('--do_parallel', default=1, help='Parallel or not, 0 or 1')
@click.option('--do_silhouette', default=1, help='do_silhouette or not, 0 or 1')
@click.option('--do_drift', default=1, help='do_drift or not, 0 or 1')
@click.option('--do_pc_features', default=1, help='do_pc_features or not, 0 or 1')
def main(kilosort_folder=None, do_parallel=True, do_pc_features=True, do_silhouette=True, do_drift=True, fs=3e4):
    """ Calculate metrics for all units on one probe"""
    # kilosort_folder = '~/res_ss_full/res_ss/tcloop_train_m022_1553627381_'
    if kilosort_folder is None:
        kilosort_folder = os.getcwd()
    if do_pc_features:
        do_include_pcs = True
    else:
        do_include_pcs = False

    (the_spike_times, the_spike_clusters, the_spike_templates, the_amplitudes, the_templates,
     the_channel_map, the_clusterIDs, the_cluster_quality,
     the_pc_features, the_pc_feature_ind) = IO.load_kilosort_data(kilosort_folder,
                                                                  fs,
                                                                  False,
                                                                  include_pcs=do_include_pcs)

    (the_spike_times, the_spike_clusters, the_spike_templates,
     the_amplitudes, the_pc_features,
     the_overlap_matrix) = KsortPostprocessing.remove_double_counted_spikes(the_spike_times,
                                                                        the_spike_clusters,
                                                                        the_spike_templates,
                                                                        the_amplitudes,
                                                                        the_channel_map,
                                                                        the_templates,
                                                                        the_pc_features,
                                                                        sample_rate=fs)


    all_metrics = calculate_metrics(the_spike_times, the_spike_clusters, the_spike_templates,
                                    the_amplitudes, the_pc_features, the_pc_feature_ind,
                                    output_folder=kilosort_folder,
                                    do_pc_features=do_pc_features,
                                    do_silhouette=do_silhouette,
                                    do_drift=do_drift,
                                    do_parallel=do_parallel)


# Launch this file and drop into debug if needed
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
    # except Exception as e:
    #     print('Error. Trying to start debugger... :\n ', e)
    #     import sys, traceback, pdb
    #
    #     extype, value, tb = sys.exc_info()
    #     traceback.print_exc()
    #     pdb.post_mortem(tb)
