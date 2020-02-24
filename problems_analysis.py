import pandas as pd
import problems


def read_stats_csv(problem_name):
    stats = pd.read_csv(f'{problems.RAW_OUTPUT_DIRECTORY}/{problem_name}/rhc__{problem_name}__run_stats_df.csv')
    stats = stats.drop(stats.columns[0], axis=1)  # drop index col from csv
    return stats


def run_rhc_analysis(problem_name):
    stats = read_stats_csv(problem_name)
    restart_values = stats['Restarts'].unique()
    for restart_value in restart_values:
        df = stats[stats['Restarts'] == restart_value]
        print(f'{problem_name} analysis run.')


if __name__ == '__main__':
    run_rhc_analysis(problems.QUEENS_NAME)  # TODO hardcoded to queens
