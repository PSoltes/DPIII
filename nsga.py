import numpy as np
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from pymoo.core.duplicate import ElementwiseDuplicateElimination
import json
from decimal import Decimal, ROUND_HALF_EVEN

data_path = '../data'
results_path = '../results'


def column(matrix, i):
    return [row[i] for row in matrix]


def get_battery_delta(battery_states, initial_battery_charge):
    battery_deltas = []
    battery_deltas.append(battery_states[0] - initial_battery_charge)
    for i in range(len(battery_states) - 1):
        battery_deltas.append(battery_states[i + 1] - battery_states[i])

    return battery_deltas


def get_grid_loads(battery_states, initial_battery_charge, loads, solar, grid_availability):
    grid_loads = []
    battery_deltas = get_battery_delta(battery_states, initial_battery_charge)
    for i in range(len(battery_states)):
        grid_loads.append(
            loads[i] - (solar[i] - battery_deltas[i]) if grid_availability[i] == True else 0)
    return np.array(grid_loads)


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

    def get_mutation_if_grid_available(self, first_interval_middle, second_interval_middle, battery_discharge_limit, prob_of_happening, rng):
        if np.random.rand() <= prob_of_happening:
            interval = self.get_interval_intersection(self.get_interval(
                first_interval_middle, battery_discharge_limit), self.get_interval(second_interval_middle, battery_discharge_limit))
            return rng.uniform(
                low=interval[0], high=interval[1])
        else:
            return second_interval_middle

    def get_mutation_if_grid_unavailable(self, solar, first_interval_middle, second_interval_middle, battery_discharge_limit, prob_of_happening, rng):
        if np.random.rand() <= prob_of_happening:
            interval = self.get_interval_intersection(self.get_interval(
                first_interval_middle, min(battery_discharge_limit, solar)), self.get_interval(second_interval_middle, battery_discharge_limit))
            return rng.uniform(
                low=interval[0], high=interval[1])
        else:
            return second_interval_middle
        

    def _do(self, problem, X, **kwargs):
        rng = np.random.default_rng()

        # for each individual
        # n individuals in population
        # m battery states for each individual
        # X is n*m array
        for i in range(len(X)):
            r = np.random.rand()
            # in kWh
            battery_discharge_limit = problem.battery_params['max_discharge_rate']
            if r < self.probability:
                newXi = np.empty(len(X[i]))
                prob = 1 / len(X[i])
                for j in range(len(X[i])):
                    is_grid_available = all(
                        column(problem.grid_availability, j))
                    min_solar_value = max(0, min(column(problem.solar, j)))
                    first_interval_middle = 0
                    if j == 0:
                        first_interval_middle = problem.initial_battery_charge
                    else:
                        first_interval_middle = X[i, j - 1]
                    newXi[j] = self.get_mutation_if_grid_available(
                        first_interval_middle, X[i, j], battery_discharge_limit, prob, rng) if is_grid_available == True else self.get_mutation_if_grid_unavailable(min_solar_value, first_interval_middle, X[i, j], battery_discharge_limit, prob, rng)
                X[i] = newXi
            elif r < self.probability + 0.05:
                prob = 1 / len(X[i])
                X[i] = [bat_value if np.random.rand() <= prob else rng.uniform(
                    low=problem.xl[idx], high=problem.xu[idx]) for idx, bat_value in enumerate(X[i])]

        return X


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
                is_grid_available = all(
                        column(problem.grid_availability, j))
                min_solar_value = max(0, min(column(problem.solar, j)))
                min_load_value = min(column(problem.loads, j))
                high = charge_rate_bounds[1] if is_grid_available == True else min(min_solar_value, charge_rate_bounds[1])
                low = charge_rate_bounds[0] if is_grid_available == True else max(charge_rate_bounds[0], min_solar_value - min_load_value)
                if not Decimal(low).quantize(Decimal(1.000000), rounding=ROUND_HALF_EVEN).compare(Decimal(low).quantize(Decimal(1.000000), rounding=ROUND_HALF_EVEN)) == -1:
                    low = high
                if j == 0:
                    next_value = problem.initial_battery_charge + rng.uniform(
                        low=low, high=high)
                else:
                    next_value = bat[j-1] + rng.uniform(
                        low=low, high=high)
                bat.append(
                    max(0, min(problem.battery_params['max_total_energy'], next_value)))
            X[i] = bat

        return X


class SingleCOE(Problem):

    def __init__(self, problem_data, xu, xl, battery_params, initial_battery_charge, conversion_rate_to_kwh=0.25):
        super().__init__(n_var=len(problem_data[0]['energy_consumption']),
                         n_obj=3,
                         n_constr=2,
                         xl=np.full(
                             (len(problem_data[0]['energy_consumption'])), xl),
                         xu=np.full((len(problem_data[0]['energy_consumption'])), xu))
        self.conversion_rate_to_kwh = conversion_rate_to_kwh
        # loads for n scenarios -> n scenarios, each having 2 days worth of fuzzied loads for each interval
        self.loads = [np.array(df['energy_consumption'])
                      * conversion_rate_to_kwh for df in problem_data]
        self.grid_availability = [
            np.array(df['grid_available']) for df in problem_data]
        # prices for n scenarios
        self.prices = [np.array(df['lmp_avg']) for df in problem_data]
        # solar for n scenarios
        self.solar = [np.array(df['solar']) *
                      conversion_rate_to_kwh for df in problem_data]
        # sum of all loads in scenario eg. array of n (scenarios) numbers
        self.total_load = [sum(x) for x in self.loads]
        self.battery_params = battery_params
        self.initial_battery_charge = initial_battery_charge

    # X0 - power status of battery

    # pop of n individuals having battery states for 192 intervals in 2 days
    # solar, loads, prices m intervals long array
    def get_prices_for_pop_scenario(self, pop, grid_loads, prices, total_load):
        price_of_energy = [sum(x) for x in np.multiply(grid_loads, prices)]
        f_price = np.full((len(pop)), 0) if total_load == 0 else np.divide(
            price_of_energy, total_load)

        return f_price

    # pop of n individuals having battery states for 192 intervals in 2 days
    # solar, loads, prices m intervals long array
    def get_emissions_for_pop_scenario(self, pop, grid_loads, total_load):
        # array n individuals * m intervals
        non_negative_grid_loads = [[max(0, y) for y in x] for x in grid_loads]
        # need to add battery emissions for this time window eg. battery_emissions / lifespan * timewindow
        emissions_of_grid = [sum(x) for x in np.multiply(
            non_negative_grid_loads, 11)]     # there is value for texas need to import it
        f_emissions = np.full((len(pop)), 0) if total_load == 0 else np.divide(
            emissions_of_grid, total_load)

        return f_emissions

    def get_lpsp_for_pop_scenario(self, pop, grid_loads, loads, solar, total_load):
        lpsp_for_pop = [sum(loads - battery_states - grid_loads[index] - solar)
                        for index, battery_states in enumerate(pop)]
        normalized_lpsp_for_pop = [
            max(0, lpsp) / total_load for lpsp in lpsp_for_pop]
        return normalized_lpsp_for_pop

    def _evaluate(self, X, out, *args, **kwargs):
        price_sum_for_each_scenario = 0
        emissions_sum_for_each_scenario = 0
        lpsp_for_each_scenario = np.full(len(X), 0.0)
        maximal_charge_rate_for_each_scenario = np.full(len(X), 0.0)
        total_power_in_system_condition_for_each_scenario = np.full(
            len(X), 0.0)
        for i in range(len(self.loads)):
            # array n individuals * m intervals
            grid_loads = [get_grid_loads(
                x, self.initial_battery_charge, self.loads[i], self.solar[i], self.grid_availability[i]) for x in X]
            price_for_scenario = self.get_prices_for_pop_scenario(
                X, grid_loads, self.prices[i], self.total_load[i])
            emissions_for_scenario = self.get_emissions_for_pop_scenario(
                X, grid_loads, self.total_load[i])
            lpsp_for_scenario = self.get_lpsp_for_pop_scenario(
                X, grid_loads, self.loads[i], self.solar[i], self.total_load[i])
            maximal_charge_rate = np.array([self.maximal_charge_rate_condition(
                individual, self.battery_params['max_discharge_rate']) for individual in X])
            total_power_in_system_condition = np.array([self.total_power_in_system_condition(
                individual, self.solar[i], grid_loads[idx]) for idx, individual in enumerate(X)])
            price_sum_for_each_scenario += price_for_scenario
            emissions_sum_for_each_scenario += emissions_for_scenario
            lpsp_for_each_scenario += lpsp_for_scenario
            maximal_charge_rate_for_each_scenario += maximal_charge_rate
            total_power_in_system_condition_for_each_scenario += total_power_in_system_condition

        all_scenarios_price_f = price_sum_for_each_scenario / len(self.loads)
        all_scenarios_emissions_f = emissions_sum_for_each_scenario / \
            len(self.loads)
        all_scenarios_lpsp_f = lpsp_for_each_scenario / len(self.loads)

        out["F"] = np.column_stack(
            [all_scenarios_price_f, all_scenarios_emissions_f, all_scenarios_lpsp_f])
        out["G"] = np.column_stack(
            [maximal_charge_rate_for_each_scenario, total_power_in_system_condition_for_each_scenario])

    def total_power_in_system_condition(self, bat_status, solar, grid_loads):
        first = Decimal(solar[0] + grid_loads[0]).quantize(Decimal(1.000000), rounding=ROUND_HALF_EVEN)
        second = Decimal(bat_status[0] - self.initial_battery_charge).quantize(Decimal(1.000000), rounding=ROUND_HALF_EVEN)

        if first.compare(second) == -1:
            return 1
        for i in range(len(bat_status) - 1):
            first = Decimal(solar[i + 1] + grid_loads[i + 1]).quantize(Decimal(1.000000), rounding=ROUND_HALF_EVEN)
            second = Decimal(bat_status[i + 1] - bat_status[i]).quantize(Decimal(1.000000), rounding=ROUND_HALF_EVEN)
            if first.compare(second) == -1:
                # print(solar[i + 1], grid_loads[i + 1], bat_status[i], bat_status[i + 1])
                return 1
        return 0

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
    with open('../results/nsga-II-results/non_scenarios_prices_scenario_3x.json') as file:
        results = json.load(file)
        for result_arr in results:
            for result in result_arr:
                print(result[2] == 0)
