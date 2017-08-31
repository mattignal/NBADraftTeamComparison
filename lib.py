# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def sorted_value(df, metric):
    df = df.sort_values(['Year', '{}'.format(metric)], ascending=[True, False])
    metric_sorted = list(df['{}'.format(metric)])
    df = df.sort_values(['Year', 'Pk'])
    df['Sorted'] = metric_sorted
    df['Difference'] = df['{}'.format(metric)] - df['Sorted']
    # Take average pick difference
    df['Avg_Pk_Difference'] = df.groupby('Pk')['Difference'].transform(lambda x:
                                                                       x.mean())

    # Line fitting to account for difference between picks
    X = df['Pk']
    Y = df['Difference']
    z = np.polyfit(X, Y, 1)
    f = np.poly1d(z)

    # Account for difference
    df = df.sort_values('Pk')
    x = df['Pk']
    df['Line_Fit_Pk_Difference'] = f[1] * x + f[0]

    # Calculate the difference once more
    df['Adj_Difference'] = df['Difference'] - df['Line_Fit_Pk_Difference']
    df['Adj_Difference'] = StandardScaler().fit_transform(df['Adj_Difference'])
    sv = pd.DataFrame(df.groupby('Tm').mean()['Adj_Difference'])
    sv.rename(columns={'Adj_Difference': 'Sorted Value {}'.format(metric)},
              inplace=True)
    return sv


def pick_value(df, metric, beg_year, end_year):
    # Take average pick value
    df['Avg_Pk'] = df.groupby('Pk')['{}'.format(metric)].\
        transform(lambda x: x.mean())

    # Curve fitting seems to do the trick, although I'd like to keep pick #1 without the curve fit
    X = df['Pk']
    Y = df['Avg_Pk']
    z = np.polyfit(X, Y, 3)
    f = np.poly1d(z)

    df = df.sort_values('Pk')
    x = df['Pk']

    # Apply curve fit
    df['Curve_Fit_Pk'] = f[3] * (x ** 3) + f[2] * (x ** 2) + f[1] * x + f[0]
    # Replace #1 suggested by curve fir with simple average #1 pick
    df.loc[df.Pk == 1, ['Curve_Fit_Pk']] = df['Avg_Pk']

    df = df.sort_values('Pk')
    df = df.set_index('Year')

    # Create lists of variables and following
    lists = df['{}'.format(metric)].groupby(df.index).agg(lambda x: list(list(
        x)))
    cum_list = []
    for i in range(beg_year, end_year):
        mini_list = []
        for j in range(len(lists[i])):
            mini_list.append(lists[i][j:j + 7])
        cum_list.append(mini_list)
    cum_list = [item for sublist in cum_list for item in sublist]

    df = df.reset_index().sort_values(['Year', 'Pk'])

    # Sequence for weighing pick value. The selection is worth 1, the next pick is worth 0.75, etc.
    simple_transform = [
        [1, 0.75, 0.55, 0.40, 0.25, 0.15, 0.10, 0.07, 0.05, 0.04, 0.03, 0.02,
         0.01] for i in range(len(df))]

    # Apply sequence
    df['Weighted_Following'] = [sum(
        [y * simple_transform[xi][yi]/sum(simple_transform[0]) for yi, y in
         enumerate(x)]) for xi, x in enumerate(cum_list)]
    # Compare more to what's next at the top, avg at bottom with sliding scale
    # 50/50 Avg pick value vs. following pick actual value at #1 pick, 100% avg. pick value at bottom
    # (solves Isaiah Thomas/Kobe Bryant problem)
    df['Sliding Scale'] = (df['Pk'].astype(float) + 58) / 118
    df['Sliding Scale2'] = 1 - df['Sliding Scale']
    df['Pick Value'] = df['Curve_Fit_Pk'] * df['Sliding Scale'] + \
                       df['Weighted_Following'] * df['Sliding Scale2']

    # Metric is just the WS/48 minus the pick value
    df['Difference'] = df['{}'.format(metric)] - df['Pick Value']
    df['Difference'] = StandardScaler().fit_transform(df['Difference'])
    pv = pd.DataFrame(df.groupby('Tm').mean()['Difference'])
    pv.rename(columns={'Difference':'Pick Value {}'.format(metric)},
              inplace=True)
    return pv