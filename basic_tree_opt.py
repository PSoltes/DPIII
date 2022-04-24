import json
import pandas as pd
import numpy as np
# [charge_to_from_battery, charge_from_grid, gas_generator_state, gas_generator_produce]

data_path = '../data'
result_path = '../results'


def is_within_expensive_tarrif(current_datetime):
    if 6 <= current_datetime.hour < 11 or 18 <= current_datetime.hour < 23:
        return True
    return False


def calc_solar_deficit_extra(solar, load):
    return load - solar


def compute_extra_solar_strat(deficit, row, current_battery_state, battery_params):
    deficit = -deficit
    if is_within_expensive_tarrif(row['localminute']):
        return [row['localminute'], 0, -deficit]
    else:
        current_battery_usable_capacity = max(
            [battery_params['max_total_energy'] - current_battery_state, 0])
        battery = min([deficit, battery_params['max_discharge_rate'],
                       current_battery_usable_capacity])
        grid = deficit - battery
        return [row['localminute'], battery, -grid]


def compute_deficit_solar_strat(deficit, row, current_battery_state, battery_params):
    current_battery_usable_energy = max(
        [current_battery_state - battery_params['usable_energy_threshold'], 0])
    battery = min([current_battery_usable_energy,
                   battery_params['max_discharge_rate'], deficit])
    grid = deficit - battery

    return [row['localminute'], -battery, grid]


def compute_management_strat_for_t(row, current_battery_state, battery_params):
    deficit = calc_solar_deficit_extra(row['solar'], row['energy_consumption'])

    if deficit == 0:
        return [row['localminute'], 0, 0]

    if deficit < 0:
        return compute_extra_solar_strat(deficit, row, current_battery_state, battery_params)
    else:
        return compute_deficit_solar_strat(deficit, row, current_battery_state, battery_params)


if __name__ == "__main__":
    df = pd.read_csv(f'{data_path}/661/15min_aggs_data.csv', parse_dates=['localminute'])
    battery_params = {}
    current_battery_state = 2
    result_arr = []
    with open(f'{data_path}/tree_strat_batteries.json', 'r') as file:
        battery_params = json.load(file)
    for index, row in df.loc[0:191].iterrows():
        if pd.isna(row['grid']):
            result_arr.append([row['localminute'], np.nan,
                               np.nan, np.nan, np.nan, True])
        else:
            result = compute_management_strat_for_t(
                row, current_battery_state, battery_params['tesla_powerwall'])
            current_battery_state += result[1]
            result_arr.append([result[0], row['energy_consumption'],
                               current_battery_state, result[2], result[2] / 4*row['lmp_avg'], False])
    result_df = pd.DataFrame(data=result_arr, columns=[
                             'localminute', 'load', 'battery_state', 'grid', 'price', 'missing_data'])
    result_df.to_csv(f'{result_path}/tree_strat.csv', index=False)
    df = pd.read_csv(f'{result_path}/tree_strat.csv')
    df = df.fillna(0, axis=1)
    print(df['price'].sum()/(df['load'].sum()/4))
