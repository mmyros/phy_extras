# You can also put your plugins in ~/.phy/plugins/.

from phy import IPlugin

# Plugin example:
#
# class MyPlugin(IPlugin):
#     def attach_to_cli(self, cli):
#         # you can create phy subcommands here with click
#         pass
from phy import IPlugin, Bunch
from phy.cluster.views import HistogramView
from pathlib import Path

c = get_config()
c.Plugins.dirs = [Path('~/.phy/plugins').expanduser()]
# list of plugin names to load in the TemplateGUIExampleClusterStatsPlugin:
c.TemplateGUI.plugins = ['RawDataFilterPluginMeanAndHighpass',
                         # 'ClusterMetricsPlugin',
                         # 'ClusterMetricsPluginBatch',
                         'ExampleCustomSplitPlugin']
