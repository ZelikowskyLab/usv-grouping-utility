import numpy as np
import pandas as pd
import jenkspy as jk


class Audio:
    def __init__(self, path_to_data):
        self.data = get_data(path_to_data)
        # self.data['Curvature'] = classify_curvature(self.data)
        self.usv_groups = create_usv_group_dataframe(self.data)


def get_data(path_to_data):
    df = pd.read_csv(path_to_data)
    return df


def get_intervals(df):
    start_data = df['Begin Time (s)']
    stop_data = df['End Time (s)']
    bins = np.asarray([[x, y] for x, y in zip(start_data.to_numpy(), stop_data.to_numpy())])
    return bins


def get_all_spaces(df):
    """
    gets the time in between all USVs and adds them to array
    :return: array of all spaces
    """
    bins = get_intervals(df)
    all_spaces = np.asarray([bins[i + 1][0] - bins[i][1] for i in range(len(bins) - 1)])
    return all_spaces


def group_usvs(df):
    all_spaces = get_all_spaces(df)  # get the scalar value of each gap between usvs
    all_spaces_log = np.log(all_spaces)  # the distribution is highly skewed do a log transformation
    standard_deviation_log = np.std(all_spaces_log)  # compute the standard deviation of the log
    all_spaces_below_std_log = [x for x in all_spaces if x < standard_deviation_log]  # get a list of spaces
    # below 1 standard deviation of the log
    jnb = jk.JenksNaturalBreaks(2)  # create a natural breaks object with two classes
    jnb.fit(all_spaces_below_std_log)  # fit the natural breaks object to the smaller dataset

    group_ids = []  # create an empty list to store group ids
    group_id = 1  # initialize the first group id (the first usv call will always belong to group id 1)
    group_ids.append(group_id)  # append the first group id to the list
    # (the first pass through the for loop will look at the space between the first and second usv calls
    # to determine whether the second usv is close enough to the first to be considered part of group one,
    # if not it will be assigned the next group)
    for space in all_spaces:
        if jnb.predict(
                space).item() == 0:  # if the space is assigned class 0 by the jnb object, keep with the current group
            group_ids.append(group_id)
        else:  # otherwise, add it to a new group
            group_id += 1
            group_ids.append(group_id)
    df['group_id'] = group_ids  # appends group ids to data frame
    return group_ids


def average_column_over_groups(df: pd.DataFrame, column: str, number_of_groups: int):
    averages = []

    for i in range(1, number_of_groups + 1):  # compute the average sinuosity of each group
        group = df[df['group_id'] == i]
        group_average = np.mean(np.asarray(group[column]))
        averages.append(group_average)
    return averages


def get_group_time(df, group_id):
    group = df[df['group_id'] == group_id]
    start = group.iloc[0]['Begin Time (s)']
    end = group.iloc[-1]['End Time (s)']
    return [start, end]


def get_group_spacing(df, group_id):
    group = df[df['group_id'] == group_id]
    start_data = group['Begin Time (s)']
    stop_data = group['End Time (s)']

    bins = np.asarray([[x, y] for x, y in zip(start_data.to_numpy(), stop_data.to_numpy())])
    if len(group) > 1:
        return np.mean(np.asarray([bins[i + 1][0] - bins[i][1] for i in range(len(bins) - 1)]))
    else:
        return 0


def get_num_usvs_in_group(df, group_id):
    group = df[df['group_id'] == group_id]
    return len(group)


def create_usv_group_dataframe(df: pd.DataFrame):
    try:
        group_ids = df['group_id']  # get the group ids from the individual usv data frame passed in
    except KeyError:
        group_ids = group_usvs(df)  # if the group ids column is not present, group the usv data frame to generate
    number_of_groups = max(group_ids)  # the number of groups is the max of the number of group IDs
    groupDF = pd.DataFrame(range(1, number_of_groups + 1),
                           columns=['group_id'])  # create a dataframe with just the list of group IDs
    start_times = []
    end_times = []
    average_spacing = []
    num_usvs_in_group = []
    for i in range(1, number_of_groups + 1):
        num_usvs_in_group.append(get_num_usvs_in_group(df, i))
        average_spacing.append(get_group_spacing(df, i))
        start_times.append(get_group_time(df, i)[0])
        end_times.append(get_group_time(df, i)[1])
    column_labels = ['Call Length (s)', 'Principal Frequency (kHz)', 'Low Freq (kHz)', 'High Freq (kHz)',
                     'Delta Freq (kHz)',
                     'Frequency Standard Deviation (kHz)', 'Slope (kHz/s)', 'Sinuosity', 'Mean Power (dB/Hz)',
                     'Tonality',
                     'Peak Freq (kHz)']
    groupDF['Start'] = start_times
    groupDF['End'] = end_times
    groupDF['Group Length'] = groupDF['End'] - groupDF['Start']
    groupDF['Num USVs'] = num_usvs_in_group
    groupDF['Average Spacing'] = average_spacing

    for label in column_labels:
        groupDF['Average ' + label] = average_column_over_groups(df, label, number_of_groups)
    curvature = classify_curvature(groupDF[['Average Sinuosity']])
    groupDF['Curvature'] = curvature
    return groupDF


def classify_curvature(df):
    df = df['Average Sinuosity']  # Get sinuosity data for the data frame
    jnb = jk.JenksNaturalBreaks(5)
    jnb.fit(df)
    curvature = [jnb.predict(x).item() for x in df]
    return curvature


if __name__ == "__main__":
    from sys import argv
    import os
    """
    Should take a path as an argument
    if path is a directory should run the script for all files in that directory
    if path is a file just run the script on that file
    File path MUST end in .csv or .xlsx
    This script will work on usv data formatted identically to deepsqueak xlsx output
    """
    path = argv[1]
    if os.path.isfile(path):  # check to see if path is file or dir
        print()

    elif os.path.isdir(path):
        print()
    else:
        print('Usage <%s>: <path to file or dir>' % argv[0])
        exit(1)
