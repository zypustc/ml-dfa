from benchmark_datasets import sets
from benchmark import reaction_calculate
import sys

param = sys.argv[1]

for set in sets:
    print(set)
    print(type(sets[set]))

    reaction_calculate(sets[set], set, str(param), 40)

