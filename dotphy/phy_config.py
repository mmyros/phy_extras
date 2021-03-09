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

c = get_config()
c.Plugins.dirs = [r'/home/m/.phy/plugins']
# list of plugin names to load in the TemplateGUIExampleClusterStatsPlugin:
c.TemplateGUI.plugins = ['RawDataFilterPluginMeanAndHighpass',
                         # 'ClusterMetricsPlugin',
                         # 'ClusterMetricsPluginBatch',
                         'ExampleCustomSplitPlugin']
