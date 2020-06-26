"""Show how to add a custom raw data filter for the TraceView and Waveform View

Use Alt+R in the GUI to toggle the filter.

"""

from scipy.signal import butter, filtfilt

from phy import IPlugin
import numpy as np


class RawDataFilterPluginMeanAndHighpass(IPlugin):
    def attach_to_controller(self, controller):
        b, a = butter(3, 150.0 / controller.model.sample_rate * 2.0, 'high')

        @controller.raw_data_filter.add_filter
        def high_pass_then_mean(arr, axis=0, mean_axis=1):
            arr = filtfilt(b, a, arr, axis=axis)
            arr = arr - np.mean(arr, axis=mean_axis, keepdims=True)
            return arr

        @controller.raw_data_filter.add_filter
        def median_then_high_pass(arr, axis=0, mean_axis=1):
            arr = arr - np.median(arr, axis=mean_axis, keepdims=True)
            return filtfilt(b, a, arr, axis=axis)

        @controller.raw_data_filter.add_filter
        def mean_then_high_pass(arr, axis=0, mean_axis=1):
            arr = arr - np.mean(arr, axis=mean_axis, keepdims=True)
            return filtfilt(b, a, arr, axis=axis)

        @controller.raw_data_filter.add_filter
        def high_pass_then_median(arr, axis=0, mean_axis=1):
            arr = filtfilt(b, a, arr, axis=axis)
            arr = arr - np.median(arr, axis=mean_axis, keepdims=True)
            return arr

        @controller.raw_data_filter.add_filter
        def mean_only(arr, axis=0, mean_axis=1):
            return arr - np.mean(arr, axis=mean_axis, keepdims=True)

        @controller.raw_data_filter.add_filter
        def median_only(arr, axis=0, mean_axis=1):
            return arr - np.median(arr, axis=mean_axis, keepdims=True)
