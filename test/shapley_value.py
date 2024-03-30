
from math import factorial
from numpy import average


def get_marginal_contribution(coalitionValues:list[tuple[set,int]], player, players):
    coalitions, values = zip(*coalitionValues)
    contributions = []
    for i1, c in enumerate(coalitions):
        if player in c:
            newCoalition = c.copy()
            newCoalition.remove(player)
            i2 = coalitions.index(newCoalition)
            l = len(newCoalition)
            p = len(players)
            weight = (factorial(l)*factorial(p-l-1))/factorial(p)
            # print((values[i1]-values[i2]), weight)
            contributions.append((values[i1]-values[i2])*weight)
    return sum(contributions)

if __name__ == "__main__":
    cv1 = [
        ({1,2}, 10000),
        ({1}, 7500),
        ({2}, 5000),
        (set(), 0)
    ]

    cv2 = [
        ({1,2,3}, 10000),
        (set(), 0),
        ({1,2}, 7500),
        ({1,3}, 7500),
        ({2,3}, 5000),
        ({1}, 5000),
        ({2}, 5000),
        ({3}, 0),
    ]

    players = {1,2,3}
    k1 = get_marginal_contribution(cv2, 1, players)
    k2 = get_marginal_contribution(cv2, 2, players)
    k3 = get_marginal_contribution(cv2, 3, players)
    print(k1,k2,k3)
    # a = [0.3333333333333333, 0.3333333333333333, 0.16666666666666666, 0.16666666666666666]
    # print(sum(a))