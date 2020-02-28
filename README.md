# ML Assignment 2

## Prerequisites
Python 3.7 is required together with the latest versions of the following packages (all available from PIP):
- `numpy`
- `pandas`
- `matplotlib`
- `scikitlearn`
- `mlrose_hiive`


## Optimization problems
1. Navigate to the directory of the optimization problem:
```
cd ./{continuous_peaks|for_peaks|knapsack}
```

2. Run the optimization for all four algorithms on the problem:
```
python optimization.py
```
This will output several stats in form of CSV files inside the *stats* sub-folder.

3. Run the analysis for all four algorithms on the problem:
```
python analysis.py
```
This will output several plots in form of PNG files inside the *plots* sub-folder.


## Neural Network
1. Navigate to the directory of neural network optimization problem:
```
cd ./mlp
```

2. Run the optimization of the neural network's weights using all different algorithms:
```
python mlp_training.py
```
This will output several stats in form of CSV files inside the *stats* sub-folder.

3. Run the analysis of the results:
```
python analysis.py
```
This will output several plots in form of PNG files inside the *plots* sub-folder.