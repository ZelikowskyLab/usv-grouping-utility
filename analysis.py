#  takes AudioData, BehaviorData, NeuroData objects and passes them to analysis functions
import numpy as np
import pandas as pd
from datacleaner import behavior as beh
from datacleaner import audio as aud
from datacleaner import datautils as util
import seaborn as sns
from matplotlib import pyplot as plt


def percentile_overlap_behavior_usv(behavior_data, audio_data, behavior):
    behavior_data = behavior_data[behavior_data['behavior'] == behavior]
    behavior_bins = beh.get_intervals(behavior_data)
    audio_bins = aud.get_intervals(audio_data)
    count = 0
    percentage_overlap_of_behavior = []
    percentage_overlap_of_audio = []
    actual_overlap = []
    for behavior_interval in behavior_bins:
        has_overlapping_usv = False
        for audio_interval in audio_bins:
            this_overlap = util.overlap(behavior_interval, audio_interval)
            if this_overlap != 0:
                count += 1
                if has_overlapping_usv:
                    actual_overlap[-1] += this_overlap
                    percentage_overlap_of_behavior[-1] += util.overlap_percentage(audio_interval, behavior_interval)
                else:
                    actual_overlap.append(this_overlap)
                    percentage_overlap_of_behavior.append(util.overlap_percentage(audio_interval, behavior_interval))
    average_percentage_of_overlap = np.average(percentage_overlap_of_behavior)
    average_actual_overlap = np.average(actual_overlap)
    return [behavior, count, average_actual_overlap, average_percentage_of_overlap]


def filter_audio_by_behavior(behavior_data, audio_data, behavior):
    headers = util.get_header_list(audio_data)
    behaviorDF = util.filter_dataframe(behavior_data, "behavior", behavior)
    audio_list = audio_data.to_numpy().tolist()
    audio_intervals = aud.get_intervals(audio_data)
    behavior_intervals = beh.get_intervals(behaviorDF)
    filtered_rows = []
    for behavior_interval in behavior_intervals:
        for audio_interval, row in zip(audio_intervals, audio_list):
            if util.overlap(audio_interval, behavior_interval) != 0:
                row.append(behavior)
                filtered_rows.append(row)
                continue
    headers.append("Behavior")
    return pd.DataFrame(filtered_rows, columns=headers)


def get_all_behavior_usv_overlaps_percentage(behavior_data, audio_data):
    behavior_usv_overlaps = []
    for behavior in beh.get_behaviors(behavior_data):
        behavior_usv_overlaps.append(
            percentile_overlap_behavior_usv(behavior_data, audio_data, behavior)
        )
    return pd.DataFrame(behavior_usv_overlaps, columns=["Behavior", "Tot#USVs", "AvgOvrlp", "AvgOvrlp%"])


def get_all_behavior_usv_overlaps(behavior_data, audio_data):
    behavior_usv_overlaps = []
    for behavior in beh.get_behaviors(behavior_data):
        behavior_usv_overlaps.append(
            get_behavior_usv_overlaps(behavior_data, audio_data, behavior)
        )
    return pd.DataFrame(behavior_usv_overlaps)


def get_behavior_triggered_average(behavior_data, neural_data, behavior, offset: int = 0):
    """Returns a numpy array containing the average fluorescence for each cell during the specified behavior.

        Parameters
        ----------
            :param behavior:
            :param behavior_data:
            :param neural_data:
            :param offset: before/after behavior start
        """
#  CHECK INACTIVE AND PLOT
    neural_data = neural_data.to_numpy()
    beh_data = behavior_data[behavior_data['behavior'] == behavior]
    bins = beh.get_intervals(beh_data)
    if offset < 0:
        bins = np.asarray([[interval[0] - 3, interval[0]] for interval in bins])
    elif offset > 0:
        bins = np.asarray([[interval[1], interval[1]+3] for interval in bins])
    behavior_specific_data = []
    for time_interval in bins:
        for row in neural_data:
            if (time_interval[0] >= row[0]) and (row[0] < time_interval[1]): #flip signs back
                behavior_specific_data.append(row)
                continue
    behavior_specific_data = np.asarray(behavior_specific_data)
    averages = [np.average(cell) for cell in behavior_specific_data.transpose()[1:]]
    return pd.DataFrame(averages, columns=[behavior])


def get_all_behavior_triggered_averages(behavior_data, neural_data, offset):
    df = pd.DataFrame()
    for behavior in beh.get_behaviors(behavior_data):
        bta = get_behavior_triggered_average(behavior_data, neural_data, behavior, offset)
        df = pd.concat([df, bta], axis=1)
    return df


def get_behavior_usv_overlaps(behavior_data, audio_data, behavior):
    behavior_data = behavior_data[behavior_data['behavior'] == behavior]
    behavior_bins = beh.get_intervals(behavior_data)
    audio_bins = aud.get_intervals(audio_data)
    count = 0
    for behavior_interval in behavior_bins:
        for audio_interval in audio_bins:
            if util.in_range(audio_interval[0], behavior_interval) or util.in_range(audio_interval[1],
                                                                                    behavior_interval):
                count += 1
                continue
    return [behavior, count]




#  bin across categories
def get_usv_triggered_averages(audio_data, neural_data):
    """Returns a numpy array containing the average fluorescence for each cell during the specified usv.

        Parameters
        ----------
        audio_data : data
            Audio data to use as a trigger for computing average neural activity
        neural_data : data
            Neural data for which we want to compute average activity
        """
    neural_data = neural_data.to_numpy()
    usv_groups = aud.create_usv_group_dataframe(audio_data)
    audio_specific_data = []
    for time_interval in zip(usv_groups['Start'], usv_groups['End']):
        usv_specific_data = []
        for row in neural_data:
            if (time_interval[0] <= row[0]) and (row[0] < time_interval[1]):
                usv_specific_data.append(row)
                continue
        audio_specific_data.append(usv_specific_data)
    audio_specific_data = np.asarray(audio_specific_data)
    print(audio_specific_data)
    averages = [np.average(cell) for cell in audio_specific_data.transpose()[1:]]
    return pd.DataFrame(averages, columns=usv_groups['group_id'])


def plot_behavior_triggered_averages(mouse):
    df = mouse.get_analysis(get_all_behavior_triggered_averages)
    behaviors = mouse.behavior.get_behaviors()

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    colors = ["red", "green", "blue", "indigo", "violet", "gray"]
    num_behaviors = len(behaviors)

    #  plot all behaviors and cells overlaid together
    for i in range(num_behaviors):
        tempDF = np.array([(i, y) for y in df[behaviors[i]]])
        sns.barplot(data=tempDF.transpose(), color=colors[i], errorbar=None, alpha=.5, dodge=True)
        ax.set_ylabel(behaviors[i])
        ax.set_xlabel("Cell ID")
        ax.set_xticklabels(df.index.values.tolist())

    #  plot labels on big graph
    ax.scatter([], [], c='red', alpha=.5, s=50, label="Face Sniffing")
    ax.scatter([], [], c='green', alpha=.5, s=50, label="Anogenital Sniffing")
    ax.scatter([], [], c='blue', alpha=.5, s=50, label="Allogrooming")
    ax.scatter([], [], c='indigo', alpha=.5, s=50, label="Chasing")
    ax.scatter([], [], c='violet', alpha=.5, s=50, label="Digging")
    ax.legend(scatterpoints=1, framealpha=0.5, fancybox=True, shadow=False, borderpad=1, frameon=True, labelspacing=0.5,
              loc="upper right", title='Behavior')
    fig, axes = plt.subplots(num_behaviors, 1, figsize=(8, 20))

    # plot individual behaviors
    for i in range(num_behaviors):
        tempDF = np.array([(i, y) for y in df[behaviors[i]]])
        sns.barplot(data=tempDF.transpose(), color=colors[i], errorbar=None, alpha=1, dodge=True, ax=axes[i])
        axes[i].set_ylabel(behaviors[i])
        axes[i].set_xlabel("Cell ID")
