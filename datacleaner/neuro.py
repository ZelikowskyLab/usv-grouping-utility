
import pandas as pd
from scipy.stats import zscore


class Neuro:
    def __init__(self, path_to_data):
        self.data = get_data(path_to_data)


def get_data(path_to_data):
    df = pd.read_csv(path_to_data)
    df['Time(s)/Cell Status'] -= df['Time(s)/Cell Status'][0]
    # df.set_index('Time(s)/Cell Status', inplace=True)
    timeColumn = df.pop('Time(s)/Cell Status')
    df = df.apply(zscore)
    df.insert(0, 'Time(s)/Cell Status', timeColumn)
    return df


def z_score_dataframe(df):
    df = df.apply(zscore)
# population ensembles


if __name__ == '__main__':
    neuro_data = Neuro(r'../EndoZetaData/NeuralData/Test_1/GroupHoused/mouse_1.csv')
    print(neuro_data.data)

