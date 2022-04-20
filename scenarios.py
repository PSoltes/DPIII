import numpy as np
import random
from sklearn.cluster import KMeans

def create_scenarios_for_array(number_of_scenarios, array, varaiance_koef):
    result = []
    for i in range(number_of_scenarios):
        result.append([])
    for load in array:
        scenario_loads = np.random.normal(load, abs(load*varaiance_koef), number_of_scenarios)
        for i in range(number_of_scenarios):
            result[i].append(scenario_loads[i])


    return result

def assign_scenario_to_cluster(n_of_clusters, scenarios, cluster_index):
    res = {}
    for i in range(n_of_clusters):
        res[i] = []
    for i in range(len(scenarios)):
        res[cluster_index[i]].append(scenarios[i])
    return res

def pick_random_scenarios_from_cluster(dict_of_clusters):
    scenarios = []
    for cluster in dict_of_clusters.values():
        scenarios.append(random.choice(cluster))
    return scenarios
        



def create_scenarios(number_of_scenarios, loads_forecast, solar_forecast):
    loads_scenarios = create_scenarios_for_array(number_of_scenarios*10, loads_forecast, 0.05)
    solar_scenarios = create_scenarios_for_array(number_of_scenarios*10, solar_forecast, 0.05)
    loads_kmeans = KMeans(n_clusters = number_of_scenarios, random_state=25).fit_predict(loads_scenarios)
    loads_assigned_to_clusters = assign_scenario_to_cluster(number_of_scenarios, loads_scenarios, loads_kmeans)
    solar_kmeans = KMeans(n_clusters = number_of_scenarios, random_state=25).fit_predict(solar_scenarios)
    solar_assigned_to_clusters = assign_scenario_to_cluster(number_of_scenarios, solar_scenarios, solar_kmeans)


    return (pick_random_scenarios_from_cluster(loads_assigned_to_clusters), pick_random_scenarios_from_cluster(solar_assigned_to_clusters))

if __name__ == "__main__":
    print(create_scenarios(10, [25, 28, 29], [25, 28, 29]))