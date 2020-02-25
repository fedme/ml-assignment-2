import pandas as pd
from OLD import problems


def read_stats_csv(problem_name):
    stats = pd.read_csv(f'{problems.RAW_OUTPUT_DIRECTORY}/{problem_name}/rhc__{problem_name}__run_stats_df.csv')
    stats = stats.drop(stats.columns[0], axis=1)  # drop index col from csv
    return stats


def rhc_iteration_vs_fitness(problem_name):
    stats = read_stats_csv(problem_name)
    iterations = stats['Iteration'].unique()
    iteration_vs_fitness = pd.DataFrame(index=iterations)

    # Loop over all max restarts
    max_restarts = stats['Restarts'].unique()
    for max_restart in max_restarts:
        df = stats[stats['Restarts'] == max_restart]

        # Find the restart that scored the best fitness value
        df2 = df[df['Iteration'] == df['Iteration'].max()]
        best_restart = df2[df2['Fitness'] == df2['Fitness'].min()]['current_restart'].iloc[0]  # TODO: should be MAX
        df3 = df[df['current_restart'] == best_restart]
        df3 = df3.set_index('Iteration')

        iteration_vs_fitness[f'Fitness_{max_restart}_Restarts'] = df3['Fitness']

    return iteration_vs_fitness


if __name__ == '__main__':
    # rhc_iteration_vs_fitness(problems.QUEENS_NAME)
    rhc_iteration_vs_fitness(problems.MAX_K_COLOR_NAME)
