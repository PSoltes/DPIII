import json
def get_prices_aggs(filename):
    with open(filename, "r") as file:
        arr = json.load(file)
    flattened = []
    sum = 0
    for element in arr:
        el = element[0][0] if type(element[0]) is list else element[0]
        flattened.append(el)
        sum += el
    print(min(flattened))
    print(max(flattened))
    print(sum/len(flattened))

if __name__ == "__main__":
    get_prices_aggs("./nsga-II-results/prices.json")