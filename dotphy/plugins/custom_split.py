"""Show how to write a custom split action."""

from phy import IPlugin, connect
import numpy as np


def k_means(x):
    """Cluster an array into two subclusters, using the K-means algorithm."""
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=2).fit_predict(x)


def outliers(x, algorithm='Local Outlier Factor'):
    # outliers_fraction = 0.1
    outliers_fraction = 'auto'
    if algorithm == "Robust covariance":
        from sklearn.covariance import EllipticEnvelope
        algo_fun = EllipticEnvelope(contamination=0.1)  # doesnt support auto
    elif algorithm == "One-Class SVM":
        from sklearn import svm
        algo_fun = svm.OneClassSVM(nu=.1, kernel="rbf", gamma=0.1)
    elif algorithm == "Isolation Forest":
        from sklearn.ensemble import IsolationForest
        algo_fun = IsolationForest(contamination=outliers_fraction, random_state=42)
    elif algorithm == "Local Outlier Factor":
        from sklearn.neighbors import LocalOutlierFactor
        algo_fun = LocalOutlierFactor(n_neighbors=135, contamination=outliers_fraction)

    # fit the data and tag outliers
    if (algorithm == "Local Outlier Factor"):
        y_pred = algo_fun.fit_predict(x)
    # elif(algorithm == "Isolation Forest") :
    else:
        y_pred = algo_fun.fit(x).predict(x)
    # else:
    #     y_pred = algo_fun.fit(x.predict(x))
    return y_pred


def my_amplitude_getter(controller, cluster_ids, name=None, load_all=True, get_bg_spikes=False):
    """Return the data requested by the amplitude view, wich depends on the
    type of amplitude.

    Parameters
    ----------
    cluster_ids : list
        List of clusters.
    name : str
        Amplitude name, see `controller._amplitude_functions`.
    load_all : boolean
        Whether to load all spikes from the requested clusters, or a subselection just
        for display.

    """
    if get_bg_spikes:
        n_channel_ids = 1
    else:
        n_channel_ids = 3
    all_amplitudes = []
    all_spike_ids = []
    all_spike_times = []
    n = controller.n_spikes_amplitudes if not load_all else None
    # Find the first cluster, used to determine the best channels.
    first_cluster = cluster_ids[0]  # Should be only one of them
    # Best channels of the first cluster.
    channel_ids = controller.get_best_channels(first_cluster)
    # Best channelS of the first cluster.
    for i_id in range(n_channel_ids):
        channel_id = channel_ids[i_id]
        # All clusters appearing on the first cluster's peak channel.
        other_clusters = controller.get_clusters_on_channel(channel_id)
        # Get the amplitude method.
        f = controller._get_amplitude_functions()[name]
        # Take spikes from the waveform selection if we're loading the raw amplitudes,
        # or by minimzing the number of chunks to load if fetching waveforms directly
        # from the raw data.
        # Otherwise we load the spikes randomly from the whole dataset.
        subset_chunks = subset_spikes = None
        if name == 'raw':
            if controller.model.spike_waveforms is not None:
                subset_spikes = controller.model.spike_waveforms.spike_ids
            else:
                subset_chunks = True
        # Go through each cluster in order to select spikes from each.
        cluster_id = cluster_ids[0]
        if get_bg_spikes or (cluster_id is None):
            # Background spikes.
            spike_ids = controller.selector(
                n, other_clusters, subset_spikes=subset_spikes, subset_chunks=subset_chunks)
        else:
            # Cluster spikes.
            spike_ids = controller.get_spike_ids(
                cluster_id, n=n, subset_spikes=subset_spikes, subset_chunks=subset_chunks)
        # Get the spike times.
        spike_times = controller._get_spike_times_reordered(spike_ids)
        if name in ('feature', 'raw'):
            # Retrieve the feature PC selected in the feature view
            # or the channel selected in the waveform view.
            channel_id = controller.selection.get('channel_id', channel_id)
        pc = controller.selection.get('feature_pc', None)
        # Call the spike amplitude getter function.
        amplitudes = f(
            spike_ids, channel_ids=channel_ids, channel_id=channel_id, pc=pc,
            first_cluster=first_cluster)
        if amplitudes is None:
            print('no amplitudes')
            continue
        assert amplitudes.shape == spike_ids.shape == spike_times.shape
        all_amplitudes.append(amplitudes)
        all_spike_ids.append(spike_ids)
        all_spike_times.append(spike_times)

    assert len(set(np.vstack(all_spike_ids)[:, 0])) == 1, 'Concatenation is not right!'

    return np.vstack(all_amplitudes).T, np.array(all_spike_ids[0]), np.array(all_spike_times[0])


class ExampleCustomSplitPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_gui_ready(sender, gui):
            @controller.supervisor.actions.add(shortcut='x')
            def kmeans_split():
                """Split using the K-means clustering algorithm on the template amplitudes
                of the first cluster.
                Hit x key to trigger
                """

                # Selected clusters across the cluster view and similarity view.
                cluster_ids = controller.supervisor.selected

                # Get the amplitudes, using the same controller method as what the amplitude view
                # is using.

                # Use the best three channels:
                y, spike_ids, spike_times = my_amplitude_getter(controller, cluster_ids, name='feature')

                # Uncomment the following section to cluster on single channel
                # # Note that we need load_all=True to load all spikes from the selected clusters,
                # # instead of just the selection of them chosen for display.
                # bunchs = controller._amplitude_getter(cluster_ids, name='feature', load_all=True)
                # # We get the spike ids and the corresponding spike template amplitudes.
                # # NOTE: in this example, we only consider the first selected cluster.
                # spike_ids = bunchs.spike_ids
                # y = bunchs.amplitudes
                # y=y.reshape((-1, 1))

                # We perform the clustering algorithm, which returns an integer for each
                # subcluster.
                labels = k_means(y)
                assert spike_ids.shape == labels.shape
                if len(set(labels)) > 1:
                    # We split according to the labels.
                    controller.supervisor.actions.split(spike_ids, labels)

            @controller.supervisor.actions.add(shortcut='d')
            def gac_split():
                """Split using the gac algorithm on the template amplitudes
                of the first cluster.
                Hit d key to trigger
                """

                import pyximport, sys, os
                pyximport.install(build_in_temp=False, inplace=True, setup_args={"include_dirs": np.get_include()})
                sys.path.append(os.path.expanduser('~/.phy/plugins/'))
                from gac import gac  # .pyx file
                # Selected clusters across the cluster view and similarity view.
                cluster_ids = controller.supervisor.selected

                # Get the amplitudes, using the same controller method as what the amplitude view
                # is using.
                # Use the best three channels:
                y, spike_ids, spike_times = my_amplitude_getter(controller, cluster_ids, name='feature')
                # y_template, _, _ = my_amplitude_getter(controller, cluster_ids, name='template')
                # y=np.hstack([y,y_template])

                # We perform the clustering algorithm, which returns an integer for each
                # subcluster.
                # data=Nxd_dims
                # sigma = 0.19#0.175 * np.sqrt(nd)
                labels = gac(np.ascontiguousarray(y.astype('float32')),
                             sigma=.7,
                             rmergex=0.25, rneighx=4, alpha=1.0, maxgrad=1000,
                             minmovex=0.00001, maxnnomerges=5000, minpoints=20)
                # nids from gac() are 0-based, but we want our single unit nids to be 1-based,
                # to leave room for junk cluster at 0 and multiunit clusters at nids < 0. So add 1:
                labels += 1
                assert spike_ids.shape == labels.shape
                # We split according to the labels.
                controller.supervisor.actions.split(spike_ids, labels)

            @controller.supervisor.actions.add(shortcut='s')
            def scrub_outliers():
                """
                Hit s key to trigger
                """

                # Selected clusters across the cluster view and similarity view.
                cluster_ids = controller.supervisor.selected

                # Get the amplitudes, using the same controller method as what the amplitude view
                # is using.
                # Use the best three channels:
                y, spike_ids, spike_times = my_amplitude_getter(controller, cluster_ids, name='feature')

                # Uncomment the following section to cluster on single channel
                # # Note that we need load_all=True to load all spikes from the selected clusters,
                # # instead of just the selection of them chosen for display.
                # # bunchs = controller._amplitude_getter(cluster_ids, name='feature', load_all=True)
                #
                # # We get the spike ids and the corresponding spike template amplitudes.
                # # NOTE: in this example, we only consider the first selected cluster.
                # spike_ids = bunchs[0].spike_ids
                # # print(bunchs)
                # # print(bunchs[0])
                # y = bunchs[0].amplitudes
                # y=y.reshape((-1, 1))

                # We perform the clustering algorithm, which returns an integer for each
                # subcluster.
                labels = outliers(y)
                labels = labels + 1  # -1 are outliers. make them 0 instead

                assert spike_ids.shape == labels.shape
                if len(set(labels)) > 1:
                    # We split according to the labels.
                    controller.supervisor.actions.split(spike_ids, labels)

            @controller.supervisor.actions.add(shortcut='t')
            def time_split():
                """
                Hit t key to trigger
                """

                import ssm
                # Selected clusters across the cluster view and similarity view.
                cluster_ids = controller.supervisor.selected

                # Get the amplitudes, using the same controller method as what the amplitude view
                # is using.
                # Use the best three channels:
                y, spike_ids, spike_times = my_amplitude_getter(controller,
                                                                cluster_ids,
                                                                name='feature')

                do_get_background_spikes = False
                if do_get_background_spikes:
                    # Last feature will be backround spike activity on the channel:
                    y_bg, spike_ids_bg, spike_times_bg = my_amplitude_getter(controller, cluster_ids,
                                                                             name='feature',
                                                                             get_bg_spikes=True)
                    y_bg = np.squeeze(y_bg)
                    # bin background spiketimes according to spikestimes of neuron of interest
                    digitized = np.digitize(spike_times_bg, spike_times, right=False)
                    # mean of background spikes' features in each bin
                    y_bg = np.array([y_bg[digitized == i].mean() for i in range(1, len(spike_times))])
                    y_bg = np.hstack([y_bg, y_bg[-1]])  # since right=False by default in digitize, append last bin
                    # Last feature will be backround spike activity on the channel:
                    y = np.hstack([y, np.expand_dims(y_bg, 1)])
                assert spike_ids.shape[0] == y.shape[0]
                # Uncomment the following section to cluster on single channel
                # # Note that we need load_all=True to load all spikes from the selected clusters,
                # # instead of just the selection of them chosen for display.
                # # bunchs = controller._amplitude_getter(cluster_ids, name='feature', load_all=True)
                #
                # # We get the spike ids and the corresponding spike template amplitudes.
                # # NOTE: in this example, we only consider the first selected cluster.
                # spike_ids = bunchs[0].spike_ids
                # # print(bunchs)
                # # print(bunchs[0])
                # y = bunchs[0].amplitudes
                # y=y.reshape((-1, 1))

                # We perform the clustering algorithm, which returns an integer for each
                # subcluster.
                noise_only = False
                if noise_only:
                    the_hmm = ssm.HMM(K=3, D=y.shape[1], observations='gaussian', transitions='standard')
                else:
                    the_hmm = ssm.HMM(K=3, D=y.shape[1], observations='gaussian', transitions='standard')

                the_hmm.fit(y, method="em", num_iters=2000)
                hmm_labels = the_hmm.most_likely_states(y)
                if noise_only:
                    # Burst, non-burst, and noise=3 labels.
                    # The one with least number of occurences is noise
                    noise_label = np.argmin([(hmm_labels == hmm_label).sum() for hmm_label in set(hmm_labels)])
                    # All zeros except for noise label:
                    labels = np.zeros_like(hmm_labels)
                    labels[hmm_labels == noise_label] = 1
                else:
                    labels = hmm_labels

                assert spike_ids.shape == labels.shape
                if len(set(labels)) > 1:
                    # We split according to the labels.
                    controller.supervisor.actions.split(spike_ids, labels)
