from re import VERBOSE
import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population
from pymoo.core.duplicate import ElementwiseDuplicateElimination
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_crossover, get_termination
from pymoo.optimize import minimize
import json
from nsga import SingleCOE, MySampling, MyMutation, MyDuplicateElimination
from scenarios import create_scenarios

data_path = '../data'
results_path = '../results'

if __name__ == "__main__":
    prices = []
    battery_charges = [0]
    solar_df = pd.read_csv(
        f'{data_path}/661/fuzzied_solar_15min_aggs_data.csv', index_col='localminute')
    consumption_df = pd.read_csv(
        f'{data_path}/661/fuzzied_energy_consumption_15min_aggs_data.csv', index_col='localminute')
    prices_df = pd.read_csv(f'{data_path}/661/15min_aggs_data.csv',
                            index_col='localminute')
    prices_series = prices_df['lmp_avg']
    j = 0
    initial_battery_charge = 0
    for index, row in solar_df.iterrows():
        print(j)
        bat_params = {}
        timewindow_prices = prices_series[j:j + 192].values
        original_loads = consumption_df.loc[index].to_list()
        original_solar = row.to_list()
        (loads_scenarios, solar_scenarios) = create_scenarios(
            9, original_loads, original_solar)
        loads_scenarios.append(original_loads)
        solar_scenarios.append(original_solar)
        array_dfs = []
        for i in range(len(loads_scenarios)):
            array_dfs.append(pd.DataFrame({'lmp_avg': timewindow_prices, 'solar': solar_scenarios[i],
                                           'energy_consumption': loads_scenarios[i]}))

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

        termination = get_termination("time", "00:00:30")
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=i,
                       save_history=True,
                       verbose=False)
        prices.append(res.F.tolist())
        if res.X is not None:
            result_list = res.X[0] if res.X.ndim == 2 else res.X
            initial_battery_charge = result_list[0]
            battery_charges.append(initial_battery_charge)
            # np.savetxt(f'./nsga-II-results/results_{i}.csv', result_list, delimiter=",")
        j += 1
        print(j)
        if j > 10:
            break
    with open(f'{results_path}/nsga-II-results/pricess.json', "w") as file:
        json.dump(prices, file)
    with open(f'{results_path}/nsga-II-results/results.json', "w") as file:
        json.dump(battery_charges, file)
