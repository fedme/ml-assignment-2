import matplotlib.pyplot as plt
from OLD import problems, problems_analysis

PLOT_FOLDER = 'problems_plots'


def rhc_iteration_vs_fitness(problem_name, xlim=None):
    df = problems_analysis.rhc_iteration_vs_fitness(problem_name)
    df.plot()
    plt.title(f'{problem_name.capitalize()} - RHC - Iterations vs Fitness')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    if xlim is not None:
        plt.xlim(0, xlim)
    plt.savefig(f'{PLOT_FOLDER}/rhc__{problem_name}__iteration_vs_fitness.png')


if __name__ == '__main__':
    rhc_iteration_vs_fitness(problems.MAX_K_COLOR_NAME)
