
def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def overlap_percentage(a, b):
    return overlap(a, b)/(b[1] - b[0])


def get_header_list(df):
    return df.columns.values.tolist()


def filter_dataframe(df, column, value):
    return df[df[column] == value]


def in_range(x, interval):
    return interval[0] <= x <= interval[1]
