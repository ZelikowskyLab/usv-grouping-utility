import numpy as np
import pandas as pd
from datacleaner import audio
from datacleaner import datautils as util


class Behavior:
    def __init__(self, path_to_data):
        self.data = get_data(path_to_data)
        self.behaviors = self.get_behaviors()

    def get_behaviors(self):
        behaviors_list = []
        for behavior in self.data['behavior']:
            if behavior not in behaviors_list:
                behaviors_list.append(behavior)
        try:
            behaviors_list.remove(np.nan)
        except ValueError:
            pass
        return behaviors_list

    def get_behavior_intervals(self):
        start_data = self.data[self.data['event'] == 'State start']
        stop_data = self.data[self.data['event'] == 'State stop']
        bins = np.asarray([[x, y] for x, y in zip(start_data.to_numpy(), stop_data.to_numpy())])
        return bins

    def plot_intervals(self):
        def plot_interval(start_df, stop_df, color, ax_index):
            for start, stop in zip(start_df, stop_df):
                ax[ax_index].axvspan(start, stop, alpha=0.5, color=color)


    # def get_behavior_triggered_averages(self, neuro_data):
    #     """Returns a numpy array containing the average fluorescence for each cell during the specified behavior.
    #
    #         Parameters
    #         ----------
    #         neuro_data : data
    #             Neural data for which we want to compute average activity
    #     """
    #     behavior_triggered_averages = {}
    #     for behavior in self.behaviors:
    #         behavior_triggered_averages[behavior] = get_behavior_triggered_average(behavior, self.data, neuro_data)
    #     return behavior_triggered_averages


def get_data(path_to_data):
    df = pd.read_csv(path_to_data)
    df = df[['Time_Relative_sf', 'Behavior', 'Event_Type']]
    df = df.rename(
        columns={
            "Time_Relative_sf": "time",
            "Behavior": "behavior",
            "Event_Type": "event"
        }
    )
    return df


def get_intervals(df):
    start_data = df[df['event'] == 'State start']['time']
    stop_data = df[df['event'] == 'State stop']['time']
    bins = np.asarray([[x, y] for x, y in zip(start_data.to_numpy(), stop_data.to_numpy())])
    return bins


def get_behaviors(df):
    behaviors_list = []
    for behavior in df['behavior']:
        if behavior not in behaviors_list:
            behaviors_list.append(behavior)
    try:
        behaviors_list.remove(np.nan)
    except ValueError:
        pass
    behaviors_list.sort()
    return behaviors_list









