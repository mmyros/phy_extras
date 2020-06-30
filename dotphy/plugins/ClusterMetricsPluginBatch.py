"""Show how to add a custom cluster metrics."""
from pdb import set_trace
import numpy as np
from phy import IPlugin
import os


class ClusterMetricsPluginBatch(IPlugin):
    def attach_to_controller(self, controller):
        """Note that this function is called at initialization time, *before* the supervisor is
        created. The `controller.cluster_metrics` items are then passed to the supervisor when
        constructing it."""

        # =============================================================================================
        # The following necessitates loading of all features, which is more complicated and time consuming.
        import pandas as pd
        print('Remember to delete .phy directory if you are running cluster metrics for a second time and changed sorting')

        # unpickle problems due to QT in phy. so cant use joblib for parallel
        # Uncomment this if you want to recompute every time phy loads (or remove quality_metrics.csv)
        # os.system('python ~/.phy/plugins/spike_io.py --do_drift=0')
        os.system('cluster_quality --do_drift=0')
        try:
            df = pd.read_csv(os.path.join(controller.dir_path, 'quality_metrics.csv'))
        except FileNotFoundError:
            # Force recalculation if spike measures do not exist
            os.system('cluster_quality --do_drift=0')
            df = pd.read_csv(os.path.join(controller.dir_path, 'quality_metrics.csv'))

        # run spike_io.main (thorough os.system if you must), save, and load
        cluster_ids = df['cluster_id'].values
        df.drop('cluster_id', inplace=True, axis=1)
        df.drop('isi_viol_rate', inplace=True, axis=1)
        df.drop('isi_viol_n', inplace=True, axis=1)
        # make shorthands
        # [df.rename({measure: measure[:5]}, axis=1, inplace=True) for measure in df.columns]

        # Insert each measure into phy
        for measure in df.columns:
            insert = lambda cluster_id: np.round(df[measure][cluster_ids == cluster_id].values[0], 2) \
                if cluster_id in cluster_ids else np.nan
            controller.cluster_metrics[measure] = controller.context.memcache(insert)
