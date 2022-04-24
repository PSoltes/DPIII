import pandas as pd
import os
import json
from datetime import datetime, timedelta
from numpy import nan, random, array

data_path = '../data'

def get_metadata(csv, unique_data_ids):
    data = pd.read_csv(csv)
    for data_id in unique_data_ids:
        row_index = data.index[data['dataid'] == str(data_id)].to_list()
        print(row_index)
        if not os.path.isdir(f'{data_path}/{data_id}'):
            os.mkdir(f'{data_path}/{data_id}')
        data.iloc[[row_index[0]], :].to_csv(
            f'{data_path}/{data_id}/meta.csv', index=False, sep=',', header=data.columns)


def split_and_transform_data(csv, unique_data_ids):
    for data_id in unique_data_ids:
        iter_csv = pd.read_csv(csv, iterator=True, chunksize=10000)
        df = pd.concat([chunk[chunk['dataid'] == data_id]
                        for chunk in iter_csv])
        df['localminute'] = pd.to_datetime(
            df['localminute'], format="%Y-%m-%d %H:%M:%S")
        df = df.sort_values(by=['localminute'])
        if not os.path.isdir(f'{data_path}/{data_id}'):
            os.mkdir(f'{data_path}/{data_id}')
        df.to_csv(f'{data_path}/{data_id}/data.csv', sep=',',
                  date_format="%Y-%m-%d %H:%M:%S", index=False)


def extract_aggs_from_data(unique_data_ids):  # solar/pv grid all_cols_sum
    time_series = pd.date_range(
        start="2018-01-01 00:00:00", end="2018-12-31 23:59:00", freq="min")
    for data_id in unique_data_ids:
        to_keep_cols = ['dataid', 'localminute',
                        'grid', 'solar']
        df = pd.read_csv(f'{data_path}/{data_id}/data.csv', index_col=False)
        meta = pd.read_csv(f'{data_path}/{data_id}/meta.csv')  # total_amount_of_pv
        total_amount_of_pv = meta['total_amount_of_pv'][0]
        df = df.drop(axis=1, labels=df.columns.difference(to_keep_cols))
        df['solar'] = df['solar'].fillna(0)
        df['solar_per_installed_pv'] = df['solar'] / total_amount_of_pv
        df['energy_consumption'] = df['grid'] + df['solar']
        df['localminute'] = df['localminute'].apply(
            lambda x: "".join(x.rsplit(':', 1)))
        df['localminute'] = pd.to_datetime(
            df['localminute'], format="%Y-%m-%d %H:%M:%S%z", utc=True)
        df['localminute'] = df['localminute'].dt.tz_convert(None)
        df['localminute'] = df['localminute'].apply(
            lambda x: x - timedelta(hours=6))
        time_df = pd.DataFrame()
        time_df['localminute'] = time_series
        df = pd.concat([df.set_index('localminute'), time_df.set_index(
            'localminute')], axis=1).reset_index()
        df.to_csv(f'{data_path}/{data_id}/aggs_data.csv', index=False)


def convert_prices_to_csv(path_to_prices):
    with open(path_to_prices, 'r') as file:
        prices_json = json.load(file)
        localtime = [data[0] for data in prices_json[0]['out_data']]
        localtime = pd.to_datetime(localtime, unit="ms")
        avg = [data[1] for data in prices_json[0]['out_data']]
        sum = [data[1] for data in prices_json[1]['out_data']]
        dataframe_data = {'localtime': localtime, 'avg': avg, 'sum': sum}
        df = pd.DataFrame(data=dataframe_data)
        df['localtime'] = df.localtime.dt.tz_localize(None)
        df['localtime'] = df['localtime'].apply(
            lambda x: x - timedelta(hours=6))
        df.to_csv(f'{data_path}/el_prices/spp.csv', index=False)


def assign_el_prices_to_1min_aggs(data_ids, path_to_lmp, path_to_spp):
    lmp = pd.read_csv(path_to_lmp, parse_dates=['localtime'])
    spp = pd.read_csv(path_to_spp, parse_dates=['localtime'])
    for id in data_ids:
        lmp_index = 0
        spp_index = 0
        data_df = pd.read_csv(f'{data_path}/{id}/aggs_data.csv',
                              parse_dates=['localminute'])
        data_df['lmp_avg'] = nan
        data_df['spp_avg'] = nan
        for index, row in data_df.iterrows():
            if row['localminute'] > lmp.at[lmp_index, 'localtime']:
                lmp_index += 1
            if row['localminute'] > spp.at[spp_index, 'localtime']:
                spp_index += 1
            data_df.at[index, 'lmp_avg'] = lmp.at[lmp_index, 'avg']
            data_df.at[index, 'spp_avg'] = spp.at[spp_index, 'avg']
        data_df.to_csv(f'{data_path}/{id}/aggs_data.csv', index=False)


def get_fuzzied_data(path, col, file_name):
    df = pd.read_csv(f'{path}/{file_name}')
    fuzzied_df = pd.DataFrame(columns=['localminute'] + list(range(0,192)))
    first = [df[col][0].tolist()]
    first_hour = array([random.normal(x, 0)
                        for x in df[col][1:4]]).tolist()
    first_4_hours = array([random.normal(x, 0)
                            for x in df[col][4:16]]).tolist()
    rest = array([random.normal(x, 0)
                    for x in df[col][16:192]]).tolist()
    row = [round(x, 3) for x in (first + first_hour + first_4_hours + rest)]
    indexed_row = [df['localminute'][0]] + row
    data_to_append = {}
    for i in range(len(fuzzied_df.columns)):
        data_to_append[fuzzied_df.columns[i]] = indexed_row[i]
    fuzzied_df = fuzzied_df.append(data_to_append, ignore_index=True)
    j = 1
    while j < len(df[col]) - 192:
        first = [df[col][j].tolist()]
        first_hour = first_hour[1:] + [random.normal(df[col][j + 3], abs(df[col][j + 3] * 0.15))]
        first_4_hours = first_4_hours[1:] + [random.normal(df[col][j + 15], abs(df[col][j + 15] * 0.25))]
        rest = rest[1:] + [random.normal(df[col][j + 191], abs(df[col][j + 191] * 0.5))]
        row = [round(x, 3) for x in (first + first_hour + first_4_hours + rest)]
        indexed_row = [df['localminute'][j]] + row
        data_to_append = {}
        for i in range(len(fuzzied_df.columns)):
            data_to_append[fuzzied_df.columns[i]] = indexed_row[i]
        fuzzied_df = fuzzied_df.append(data_to_append, ignore_index=True)
        j += 1
        if j % 100 == 0:
            print(j)
    fuzzied_df.to_csv(f'{path}/fuzzied_{col}_{file_name}', index=False)


def get_N_min_aggs_data(data_ids, N):
    for id in data_ids:
        df = pd.read_csv(f'{data_path}/{id}/aggs_data_fixedNAS.csv')
        meta = pd.read_csv(f'{data_path}/{id}/meta.csv')
        N_min_df = pd.DataFrame()
        N_min_df['localminute'] = df['localminute'].iloc[::N]
        N_min_df = N_min_df.reset_index()
        df = df.fillna(0)
        grouped_by_df = df.groupby(df.index // N).sum()
        N_min_df['grid'] = grouped_by_df['grid'] / N
        N_min_df['solar'] = grouped_by_df['solar'] / N
        N_min_df['solar_per_installed_pv'] = N_min_df['solar'] / \
            meta['total_amount_of_pv'][0]
        N_min_df['energy_consumption'] = N_min_df['grid'] + N_min_df['solar']
        N_min_df['lmp_avg'] = grouped_by_df['lmp_avg'] / N
        N_min_df['spp_avg'] = grouped_by_df['spp_avg'] / N
        N_min_df = N_min_df.round(3)
        N_min_df.to_csv(f'{data_path}/{id}/{N}min_aggs_data.csv', index=False)


def fixNA(data_ids):
    upper_limit_date = datetime.strptime(
        "2019-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    lower_limit_date = datetime.strptime(
        "2018-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    for id in data_ids:
        df = pd.read_csv(f'{data_path}/{id}/aggs_data.csv', parse_dates=["localminute"])
        nan_rows = df[df["grid"].isna()]
        for index, row in nan_rows.iterrows():
            dates = []
            for i in range(5):
                date_up = row['localminute'] + timedelta(days=i + 1)
                date_down = row['localminute'] - timedelta(days=i + 1)
                if date_up < upper_limit_date:
                    dates.append(date_up)
                if date_down >= lower_limit_date:
                    dates.append(date_down)
            aggs = df[df["localminute"].isin(dates)].sum()
            df.at[index, "solar"] = aggs["solar"] / len(dates)
            df.at[index, "grid"] = aggs["grid"] / len(dates)
            df.at[index, "energy_consumption"] = df.at[index,
                                                       "grid"] + df.at[index, "solar"]
        df.to_csv(f'{data_path}/{id}/aggs_data_fixedNAS.csv')

def getNA(data_ids):
    for id in data_ids:
        df = pd.read_csv(f'{data_path}/{id}/aggs_data_fixedNAS.csv', parse_dates=["localminute"])

if __name__ == "__main__":
    unique_data_ids = [661, 1642, 2335, 2818, 3039, 3456, 3538, 4031, 4373, 4767, 5746, 6139, 7536, 7719,
                       7800, 7901, 7951, 8565, 9019, 9278, 8156, 8386, 2361, 9922, 9160]
    for i in range(25):
        get_fuzzied_data(f'{data_path}/661/2_day_scenarios', 'solar', f'scenario_{i}.csv')

#'./el_prices/lmp.csv', './el_prices/spp.csv'
