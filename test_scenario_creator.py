import pandas as pd
from math import floor

data_path = '../data'
def create_test_scenarios_csvs(number_of_scenarios, scenario_length):
    df = pd.read_csv(f'{data_path}/661/15min_aggs_data.csv')
    scenario_length_15_intervals = scenario_length * 96 * 2
    scenario_intervals_in_year = min(365, max(1, floor(365 / number_of_scenarios)))
    start = 0
    end = scenario_length_15_intervals

    for scenario_number in range(number_of_scenarios):
        scenario_df = df.loc[start:end]
        scenario_df.drop('index', inplace=True, axis=1)
        scenario_df.to_csv(f'{data_path}/661/{scenario_length}_day_scenarios/scenario_{scenario_number}.csv', index=False)
        start += scenario_intervals_in_year * 96
        end += scenario_intervals_in_year * 96


if __name__ == "__main__":
    create_test_scenarios_csvs(25, 2)

