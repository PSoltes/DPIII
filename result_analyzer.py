import json
import pandas as pd
import numpy as np
from nsga import get_grid_loads

results_path = '../results'
data_path = '../data'

def get_price_for_optimized_battery_states(filename, starting_index = 0):
    prices_df = pd.read_csv(f'{data_path}/661/2_day_scenarios/scenario_0.csv',
                            index_col='localminute')
    with open(filename, "r") as file:
        arr = json.load(file)
        prices = prices_df['lmp_avg'][starting_index:starting_index + len(arr) -1].values
        solar = prices_df['solar'][starting_index:starting_index + len(arr) - 1].values
        loads = prices_df['energy_consumption'][starting_index:starting_index + len(arr) - 1].values

        grid_loads = get_grid_loads(arr[1:], arr[0], loads/4, solar/4, np.full(len(solar), True))
        price = sum(grid_loads * prices) / sum(loads/4)

        df = pd.DataFrame({'solar': solar / 4, 'energy_consumption': loads / 4, 'load': grid_loads, 'battery': arr[1:], 'prices': prices})
        df.to_csv(f'{results_path}/nsga.csv', index=False)


        print(price)


if __name__ == "__main__":
    get_price_for_optimized_battery_states(f'{results_path}/nsga-II-results/scenarios_non_outage_results_scenario_0.json', 0)