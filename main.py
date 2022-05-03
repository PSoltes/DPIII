from re import VERBOSE
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_crossover, get_termination
from pymoo.optimize import minimize
import json
from nsga import SingleCOE, MySampling, MyMutation, MyDuplicateElimination
from scenarios import create_scenarios

data_path = '../data'
results_path = '../results'

def optimize_scenario(scenario_number):
    prices = []
    battery_charges = [0]
    solar_df = pd.read_csv(
        f'{data_path}/661/2_day_scenarios/fuzzied_solar_scenario_{scenario_number}.csv', index_col='localminute')
    consumption_df = pd.read_csv(
        f'{data_path}/661/2_day_scenarios/fuzzied_energy_consumption_scenario_{scenario_number}.csv', index_col='localminute')
    prices_df = pd.read_csv(f'{data_path}/661/2_day_scenarios/scenario_{scenario_number}.csv',
                            index_col='localminute')
    prices_series = prices_df['lmp_avg']
    j = 0
    initial_battery_charge = 0
    for index, row in solar_df.iterrows():
        bat_params = {}
        timewindow_prices = prices_series[j:j + 192].values
        original_loads = consumption_df.loc[index].to_list()
        original_solar = row.to_list()
        loads_scenarios = []
        solar_scenarios = []
        grid_availability = []
        (loads_scenarios, solar_scenarios, grid_availability) = create_scenarios(
            9, original_loads, original_solar)
        loads_scenarios.append(original_loads)
        solar_scenarios.append(original_solar)
        grid_availability.append(np.full(len(original_loads), True).tolist())
        array_dfs = [] 
        for i in range(len(loads_scenarios)):
            array_dfs.append(pd.DataFrame({'lmp_avg': timewindow_prices, 'solar': solar_scenarios[i],
                                           'energy_consumption': loads_scenarios[i], 'grid_available': grid_availability[i]}))

        with open(f'{data_path}/batteries.json', 'r') as file:
            bat_params = json.load(file)

        problem = SingleCOE(array_dfs, bat_params["tesla_powerwall"]
                            ["max_total_energy"], 0, bat_params["tesla_powerwall"], initial_battery_charge)
        algorithm = NSGA2(
            pop_size=40,
            n_offsprings=40,
            sampling=MySampling(),
            crossover=get_crossover("real_ux"),
            mutation=MyMutation(),
            eliminate_duplicates=MyDuplicateElimination(),
        )

        termination = get_termination("time", "00:02:25")
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=i,
                       save_history=True,
                       verbose=False)
        res_F = res.F.tolist()
        res_F_prices = [x[0] for x in list(filter(lambda x: x[2] == 0, res_F))] if res.F.ndim == 2 else [res_F[0]]
        best_price_index = res_F_prices.index(max(res_F_prices))
        prices.append(res_F)
        if res.X is not None:
            result_list = res.X[best_price_index] if res.X.ndim == 2 else res.X
            initial_battery_charge = result_list[0]
            battery_charges.append(initial_battery_charge)
            # np.savetxt(f'./nsga-II-results/results_{i}.csv', result_list, delimiter=",")
        j += 1
        print(j)
        with open(f'{results_path}/nsga-II-results/scenarios_non_outage_results_scenario_{scenario_number}.json', "w") as file:
            json.dump(battery_charges, file)
        if j > 191:
            break
    with open(f'{results_path}/nsga-II-results/scenarios_non_outage_prices_scenario_{scenario_number}.json', "w") as file:
        json.dump(prices, file)
    with open(f'{results_path}/nsga-II-results/scenarios_non_outage_results_scenario_{scenario_number}.json', "w") as file:
        json.dump(battery_charges, file)


if __name__ == "__main__":
    optimize_scenario(17)
    
