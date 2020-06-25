"""Show how to add a custom cluster metrics."""
from pdb import set_trace
import numpy as np
from phy import IPlugin
import spike_io
import os


class ClusterMetricsPlugin(IPlugin):
    def attach_to_controller(self, controller):
        """Note that this function is called at initialization time, *before* the supervisor is
        created. The `controller.cluster_metrics` items are then passed to the supervisor when
        constructing it."""

        def meanisi(cluster_id):
            t = controller.get_spike_times(cluster_id).data
            return np.diff(t).mean() if len(t) >= 2 else 0

        def snr(cluster_id):
            # This function takes a cluster id as input and returns a scalar.
            # data.data is a (n_spikes, n_samples, n_channels) array.
            data = controller._get_waveforms(cluster_id)
            noise_std = np.concatenate((data.data[:, :10, :], data.data[:, :10, :]), axis=1).std(axis=(0, 1))
            sig_std = data.data.mean(0).std(0)
            m = (sig_std / noise_std).max()  # max snr across channels
            # m=erf(sig_std/noise_std/2).max()*100   # max "isolation" across channels
            # m=abs(data.data.mean(0).min()) # mean over selected spikes, min over all samples and channels
            return m

        # Use this dictionary to define custom cluster metrics.
        # We memcache the function so that cluster metrics are only computed once and saved
        # within the session, and also between sessions (the memcached values are also saved
        # on disk).
        controller.cluster_metrics['meanisi'] = controller.context.memcache(meanisi)
        controller.cluster_metrics['snr'] = controller.context.memcache(snr)
        isi_threshold = 0.0015
        min_isi = 0.000166

        # params = {
        #     "isi_threshold": 0.0015,
        #     "min_isi": 0.000166,
        #     "num_channels_to_compare": 13,
        #     "max_spikes_for_unit": 500,
        #     "max_spikes_for_nn": 10000,
        #     "n_neighbors": 4,
        #     'n_silhouette': 10000,
        #     # "quality_metrics_output_file": os.path.join(os.path.expanduser(output_folder), "metrics.csv"),
        #     "drift_metrics_interval_s": 51,
        #     "drift_metrics_min_spikes_per_interval": 10
        # }

        def get_data(cluster_id):
            spike_times = controller.get_spike_times(cluster_id).data
            spike_clusters = np.zeros_like(spike_times)

            return spike_times, spike_clusters

        def presence_ratio(cluster_id):
            spike_train, spike_clusters = get_data(cluster_id)
            return spike_io.QualityMetrics.presence_ratio(spike_train, np.min(spike_train), np.max(spike_train), )

        def amplitude_cutoff(cluster_id):
            spike_ids = controller.get_spike_ids(cluster_id)
            amplitudes = controller.get_spike_raw_amplitudes(spike_ids)
            return spike_io.QualityMetrics.amplitude_cutoff(amplitudes)

        def isi_viol(cluster_id):
            spike_train, spike_clusters = get_data(cluster_id)
            # TODO minmax should really be on all spiketimes
            return spike_io.QualityMetrics.isi_violations(spike_train, np.min(spike_train), np.max(spike_train),
                                                          isi_threshold,
                                                          min_isi)[0]

        controller.cluster_metrics['isiviol'] = controller.context.memcache(isi_viol)
        controller.cluster_metrics['pres'] = controller.context.memcache(presence_ratio)
        controller.cluster_metrics['ampcut'] = controller.context.memcache(amplitude_cutoff)
        # =============================================================================================
        # The following necessitates loading of all features, which is more complicated and time consuming.

        # (spike_times, spike_clusters, spike_templates, amplitudes, templates,
        #  channel_map, clusterIDs, cluster_quality, pc_features, pc_feature_ind) = spike_io.IO.load_kilosort_data(
        #     controller.dir_path, 3e4, False, include_pcs=True)
        import pandas as pd
        # Uncomment this if you want to recompute every time phy loads
        # os.system('python ~/.phy/plugins/spike_io.py --do_drift=0')
        try:
            df = pd.read_csv(os.path.join(controller.dir_path, 'quality_metrics.csv'))
        except FileNotFoundError:
            # Force recalculation if spike measures do not exist
            os.system('python ~/.phy/plugins/spike_io.py --do_drift=0')
            df = pd.read_csv(os.path.join(controller.dir_path, 'quality_metrics.csv'))
        # total_units = np.max(spike_clusters) + 1
        # (isolation_distance, l_ratio, d_prime, nn_hit_rate,
        #  nn_miss_rate) = spike_io.Wrappers.calculate_pc_metrics(spike_clusters,
        #                                                         spike_templates,
        #                                                         total_units,
        #                                                         pc_features,
        #                                                         pc_feature_ind,
        #                                                         params['num_channels_to_compare'],
        #                                                         params['max_spikes_for_unit'],
        #                                                         params['max_spikes_for_nn'],
        #                                                         params['n_neighbors'],
        #                                                         do_parallel=False)
        #
        # silhouette_scores = spike_io.Wrappers.calculate_silhouette_score(spike_clusters,
        #                                                                  pc_features,
        #                                                                  pc_feature_ind,
        #                                                                  params['n_silhouette'],
        #                                                                  do_parallel=False)

        # instead of above, run spike_io.main (thorough os.system if you must), save, and load
        # unpickle problems due to QT in phy. so cant use joblib for parallel
        cluster_ids = df['cluster_id'].values
        df.drop('cluster_id', inplace=True, axis=1)
        # make shorthands
        [df.rename({measure: measure[:5]}, axis=1, inplace=True) for measure in df.columns]
        # Insert each measure into phy
        for measure in df.columns:
            insert = lambda cluster_id: np.round(df[measure][cluster_ids == cluster_id].values[0], 2) \
                if cluster_id in cluster_ids else np.nan
            controller.cluster_metrics[measure] = controller.context.memcache(insert)

        #
        # controller.cluster_metrics['idist'] = controller.context.memcache(
        #     lambda cluster_id: np.round(df['isolation_distance'][np.unique(df['cluster_id']) == cluster_id].values[0],
        #                                 2))
        # controller.cluster_metrics['lrat'] = controller.context.memcache(
        #     lambda cluster_id: np.round(df['l_ratio'][np.unique(df['cluster_id']) == cluster_id].values[0], 2))
        # controller.cluster_metrics['miss'] = controller.context.memcache(
        #     lambda cluster_id: np.round(df['nn_miss_rate'][np.unique(df['cluster_id']) == cluster_id].values[0], 2))
        # controller.cluster_metrics['hit'] = controller.context.memcache(
        #     lambda cluster_id: np.round(df['nn_hit_rate'][np.unique(df['cluster_id']) == cluster_id].values[0], 2))
        # controller.cluster_metrics['dpr'] = controller.context.memcache(
        #     lambda cluster_id: np.round(df['d_prime'][np.unique(df['cluster_id']) == cluster_id].values[0], 2))
        # controller.cluster_metrics['sil'] = controller.context.memcache(
        #     lambda cluster_id: np.round(df['silhouette_score'][np.unique(df['cluster_id']) == cluster_id].values[0], 2))

        # def drift_metrics(cluster_id):
        #     max_drift, cumulative_drift = spike_io.Wrappers.calculate_drift_metrics(spike_times,
        #                                                                             spike_clusters,
        #
        #                                                                             pc_features,
        #                                                                             pc_feature_ind,
        #                                                                             params['drift_metrics_interval_s'],
        #                                                                             params[
        #                                                                                 'drift_metrics_min_spikes_per_interval'])
        #     return max_drift, cumulative_drift
