"""Show how to add a custom cluster metrics."""
import numpy as np
from phy import IPlugin

import cluster_quality.quality_metrics

try:
    from cluster_quality import spike_io
except Exception as e:
    print(e)
    print('Cant find spike_io to calculate cluster_metrics! Trying again..')
    from cluster_quality import spike_io


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

        def get_data(cluster_id):
            spike_times = controller.get_spike_times(cluster_id).data
            spike_clusters = np.zeros_like(spike_times)

            return spike_times, spike_clusters

        def presence_ratio(cluster_id):
            spike_train, spike_clusters = get_data(cluster_id)
            return cluster_quality.quality_metrics.QualityMetrics.presence_ratio(spike_train, np.min(spike_train), np.max(spike_train), )

        def isi_viol(cluster_id):
            spike_train, spike_clusters = get_data(cluster_id)
            # TODO minmax should really be on all spiketimes
            return cluster_quality.quality_metrics.QualityMetrics.isi_violations(spike_train, np.min(spike_train), np.max(spike_train),
                                                                                 isi_threshold,
                                                                                 min_isi)[0]

        controller.cluster_metrics['isiviol'] = controller.context.memcache(isi_viol)
        controller.cluster_metrics['pres'] = controller.context.memcache(presence_ratio)