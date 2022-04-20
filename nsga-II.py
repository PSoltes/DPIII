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
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.optimize import minimize
import json


class MyDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        return np.array_equal(a.X, b.X)


class MyMutation(Mutation):
    def __init__(self, probability=0.4):
        super().__init__()
        self.probability = probability

    def get_interval_intersection(self, intervalA, intervalB):
        if intervalA[1] < intervalB[0]:
            return (intervalA[0], intervalB[1])
        if intervalB[1] < intervalA[0]:
            return (intervalB[0], intervalA[1])

        return (max(intervalA[0], intervalB[0]), min(intervalA[1], intervalB[1]))

    def get_interval(self, number, size):
        return (max(0, number - size), number + size)

    def _do(self, problem, X, **kwargs):
        rng = np.random.default_rng()

        # for each individual
        for i in range(len(X)):
            r = np.random.rand()
            battery_discharge_limit = problem.battery_params['max_discharge_rate']
            if r < self.probability:
                newXi = np.empty(len(X[i]))
                prob = 1 / len(X[i])
                for j in range(len(X[i])):
                    if j == 0:
                        if np.random.rand() <= prob:
                            interval = self.get_interval_intersection(self.get_interval(
                                problem.initial_battery_charge, battery_discharge_limit), self.get_interval(X[i, j], battery_discharge_limit))
                            newXi[j] = rng.uniform(
                                low=interval[0], high=interval[1])
                        else:
                            newXi[j] = X[i, j]
                    else:
                        if np.random.rand() <= prob:
                            interval = self.get_interval_intersection(self.get_interval(
                                X[i, j - 1], battery_discharge_limit), self.get_interval(X[i, j], battery_discharge_limit))
                            newXi[j] = rng.uniform(
                                low=interval[0], high=interval[1])
                            if newXi[j] < 0:
                                print(newXi[j])
                        else:
                            newXi[j] = X[i, j]
                X[i] = newXi
            elif r < self.probability + 0.05:
                prob = 1 / len(X[i])
                X[i] = [bat_value if np.random.rand() <= prob else rng.uniform(
                    low=problem.xl[idx], high=problem.xu[idx]) for idx, bat_value in enumerate(X[i])]

        return X


class MyCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):

        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape

        # The output owith the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full_like(X, None, dtype=np.object)

        # for each mating provided
        for k in range(n_matings):

            a, b = X[0, k], X[1, k]
            off_a = np.full((n_var, len(problem.loads)), None)
            off_b = np.full((n_var, len(problem.loads)), None)
            for i in range(n_var):
                for j in range(len(problem.loads)):
                    if np.random.rand() < 0.5:
                        off_a[i][j] = a[i][j]
                        off_b[i][j] = b[i][j]
                    else:
                        off_a[i][j] = b[i][j]
                        off_b[i][j] = a[i][j]
            Y[0, k], Y[1, k] = off_a, off_b

        print(Y)

        return Y


class MySampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), None)
        rng = np.random.default_rng()
        charge_rate_bounds = (-problem.battery_params['max_discharge_rate'],
                              problem.battery_params['max_discharge_rate'])
        for i in range(n_samples):
            bat = []
            for j in range(problem.n_var):
                next_value = 0
                if j == 0:
                    next_value = problem.initial_battery_charge + rng.uniform(
                        low=charge_rate_bounds[0], high=charge_rate_bounds[1])
                else:
                    next_value = bat[j-1] + rng.uniform(
                        low=charge_rate_bounds[0], high=charge_rate_bounds[1])
                bat.append(
                    max(0, min(problem.battery_params['max_total_energy'], next_value)))
            X[i] = bat

        return X


class SingleCOE(Problem):

    def __init__(self, problem_data, xu, xl, battery_params, initial_battery_charge, conversion_rate_to_kwh=0.25):
        super().__init__(n_var=len(problem_data[0]['energy_consumption']),
                         n_obj=2,
                         n_constr=1,
                         xl=np.full(
                             (len(problem_data[0]['energy_consumption'])), xl),
                         xu=np.full((len(problem_data[0]['energy_consumption'])), xu))
        self.conversion_rate_to_kwh = conversion_rate_to_kwh
        self.loads = [np.array(df['energy_consumption'])
                      * conversion_rate_to_kwh for df in problem_data]
        self.prices = [np.array(df['lmp_avg']) for df in problem_data]
        self.solar = [np.array(df['solar']) *
                      conversion_rate_to_kwh for df in problem_data]
        self.total_load = [sum(x) for x in self.loads]
        self.battery_params = battery_params
        self.initial_battery_charge = initial_battery_charge

    def get_battery_delta(self, battery_states):
        battery_deltas = []
        battery_deltas.append(battery_states[0] - self.initial_battery_charge)
        for i in range(len(battery_states) - 1):
            battery_deltas.append(battery_states[i + 1] - battery_states[i])

        return battery_deltas

    def get_grid_loads(self, battery_states, loads, solar):
        grid_loads = []
        battery_deltas = self.get_battery_delta(battery_states)
        for i in range(len(battery_states)):
            grid_loads.append(
                loads[i] - (solar[i] - battery_deltas[i]))
        return grid_loads

    # X0 - power gotten from grid, buy -> +, sell -> -
    # X1 - power status of battery

    def _evaluate(self, X, out, *args, **kwargs):
        total_price = 0
        emissions = 0
        for i in range(len(self.loads)):
            grid_loads = [self.get_grid_loads(
                x, self.loads[i], self.solar[i]) for x in X]
            non_negative_grid_loads = [
                [max(0, xi) for xi in x] for x in grid_loads]
            price_of_energy = [sum(x) for x in np.multiply(
                grid_loads, self.prices[i])]
            # there is value for texas need to import it
            emissions_of_grid = [sum(x) for x in np.multiply(
                non_negative_grid_loads, 11)]
            # there is value for this, need to import
            emissions_of_solar = sum(np.multiply(self.solar[i], 5))
            # need to add battery emissions for this time window eg. battery_emissions / lifespan * timewindow
            total_emisions = emissions_of_grid + emissions_of_solar
            f_price = np.full((len(X)), 0) if self.total_load[i] == 0 else np.divide(
                price_of_energy, self.total_load[i])
            f_emissions = np.full((len(X)), 0) if self.total_load[i] == 0 else np.divide(
                total_emisions, self.total_load[i])
            maximal_charge_rate = np.array([self.maximal_charge_rate_condition(
                individual, self.battery_params['max_discharge_rate']) for individual in X])
            total_price += f_price
            emissions += f_emissions

        all_scenarios_price_f = total_price / len(self.loads)
        all_scenarios_emissions_f = emissions / len(self.loads)

        out["F"] = np.column_stack(
            [all_scenarios_price_f, all_scenarios_emissions_f])
        out["G"] = np.column_stack(
            [maximal_charge_rate])

    def maximal_charge_rate_condition(self, bat_status, charge_rate):
        total_sum = 0
        if abs(self.initial_battery_charge - bat_status[0]) > charge_rate:
            total_sum += abs(self.initial_battery_charge -
                             bat_status[0]) - charge_rate
        for i in range(len(bat_status) - 1):
            if abs(bat_status[i + 1] - bat_status[i]) > charge_rate:
                total_sum += abs(bat_status[i + 1] -
                                 bat_status[i]) - charge_rate
        return total_sum


if __name__ == "__main__":
    prices = []
    battery_charges = [0]
    solar_df = pd.read_csv(
        './661/fuzzied_solar_15min_aggs_data.csv', index_col='localminute')
    consumption_df = pd.read_csv(
        './661/fuzzied_energy_consumption_15min_aggs_data.csv', index_col='localminute')
    prices_df = pd.read_csv('./661/15min_aggs_data.csv',
                            index_col='localminute')
    prices_series = prices_df['lmp_avg']
    i = 0
    initial_battery_charge = 0
    for index, row in solar_df.iterrows():
        bat_params = {}
        timewindow_prices = prices_series[i:i + 192].values
        df = pd.DataFrame({'lmp_avg': timewindow_prices, 'solar': row,
                          'energy_consumption': consumption_df.loc[index]})
        array_dfs = [df.copy() for i in range(10)]
        with open('./batteries.json', 'r') as file:
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

        termination = get_termination("time", "00:00:05")
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
        i += 1
        print(i)
        if i > 10:
            break
    with open("./nsga-II-results/pricess.json", "w") as file:
        json.dump(prices, file)
    with open("./nsga-II-results/results.json", "w") as file:
        json.dump(battery_charges, file)
