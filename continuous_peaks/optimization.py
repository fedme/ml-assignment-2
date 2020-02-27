from timeit import default_timer as timer
import pandas
import mlrose_hiive

PROBLEM_NAME = 'continuous_peaks'
STATS_FOLDER = 'stats'
SEED = 12367
MAX_ITERS = 1000
eval_count = 0
orig_fitness_func = None
SIZE_VALUES = [5, 10, 20, 50, 100]


def fitness_func(state):
    global eval_count
    eval_count += 1
    return orig_fitness_func.evaluate(state)


def get_problem(size=100, t_pct=0.06):
    global orig_fitness_func
    orig_fitness_func = mlrose_hiive.ContinuousPeaks(t_pct)

    fitness = mlrose_hiive.CustomFitness(fitness_func)
    problem = mlrose_hiive.DiscreteOpt(length=size, fitness_fn=fitness, maximize=True)

    return problem


def rhc_optimization(size):
    algo = 'rhc'
    problem = get_problem(size)

    # Gridsearch params
    restarts_values = [10, 30, 50]
    max_attempts_values = [5]
    n_runs = len(restarts_values) * len(max_attempts_values)

    # Best vals
    restarts, max_attempts = None, None
    fitness, curves, n_invocations, time = float('-inf'), [], 0, 0

    # Gridsearch
    global eval_count
    run_counter = 0
    for run_restarts in restarts_values:
        for run_max_attempts in max_attempts_values:
            # Print status
            run_counter += 1
            print(
                f'RUN {run_counter} of {n_runs} [restarts: {run_restarts}] [max_attempts: {run_max_attempts}]')

            # Run problem
            eval_count = 0
            start = timer()
            run_state, run_fitness, run_curves = mlrose_hiive.random_hill_climb(problem,
                                                                                max_iters=MAX_ITERS,
                                                                                random_state=SEED,
                                                                                curve=True)
            end = timer()

            # Save curves and params
            if run_fitness > fitness:
                restarts = run_restarts
                max_attempts = run_max_attempts
                fitness = run_fitness
                curves = run_curves
                n_invocations = eval_count
                time = end - start

    df = pandas.DataFrame(curves, columns=['fitness'])
    df['restarts'] = restarts
    df['max_attempts'] = max_attempts
    df['max_iters'] = MAX_ITERS
    df['n_invocations'] = n_invocations
    df['time'] = time
    df.to_csv(f'{STATS_FOLDER}/{algo}_{size}_stats.csv', index=False)

    print(f'{algo}_{size} run.')


def sa_optimization(size):
    algo = 'sa'
    problem = get_problem(size)

    # Gridsearch params

    init_temp = 1
    decay = 0.9
    min_temp = 0.1
    schedule_values = [
        mlrose_hiive.GeomDecay(init_temp=init_temp, decay=decay, min_temp=min_temp),
        mlrose_hiive.ArithDecay(init_temp=init_temp, decay=decay, min_temp=min_temp),
        mlrose_hiive.ExpDecay(init_temp=init_temp, min_temp=min_temp)
    ]
    max_attempts_values = [10, 50, 100, 200, 1000, 5000]
    n_runs = len(schedule_values) * len(max_attempts_values)

    # Best vals
    schedule, max_attempts = None, None
    fitness, curves, n_invocations, time = float('-inf'), [], 0, 0

    # Gridsearch
    global eval_count
    run_counter = 0
    for run_schedule in schedule_values:
        for run_max_attempts in max_attempts_values:
            # Print status
            run_counter += 1
            print(f'RUN {run_counter} of {n_runs} [schedule: {run_schedule.__class__.__name__}] [max_attempts: {run_max_attempts}]')

            # Run problem
            eval_count = 0
            start = timer()
            run_state, run_fitness, run_curves = mlrose_hiive.simulated_annealing(problem,
                                                                                  max_iters=MAX_ITERS,
                                                                                  random_state=SEED,
                                                                                  curve=True)
            end = timer()

            # Save curves and params
            if run_fitness > fitness:
                schedule = run_schedule.__class__.__name__
                max_attempts = run_max_attempts
                fitness = run_fitness
                curves = run_curves
                n_invocations = eval_count
                time = end - start

    df = pandas.DataFrame(curves, columns=['fitness'])
    df['schedule'] = schedule
    df['max_attempts'] = max_attempts
    df['max_iters'] = MAX_ITERS
    df['n_invocations'] = n_invocations
    df['time'] = time
    df.to_csv(f'{STATS_FOLDER}/{algo}_{size}_stats.csv', index=False)

    print(f'{algo}_{size} run.')


def ga_optimization(size):
    algo = 'ga'
    problem = get_problem(size)

    # Gridsearch params
    max_attempts_values = [5]
    pop_size_values = [50, 100, 150, 200]
    pop_breed_percent_values = [0.5, 0.75]
    mutation_prob_values = [0.1, 0.3]
    n_runs = len(max_attempts_values) * len(pop_size_values) * len(pop_breed_percent_values) * len(mutation_prob_values)

    # Best vals
    pop_size, pop_breed_percent, mutation_prob, max_attempts = None, None, None, None
    fitness, curves, n_invocations, time = float('-inf'), [], 0, 0

    # Gridsearch
    global eval_count
    run_counter = 0
    for run_pop_size in pop_size_values:
        for run_pop_breed_percent in pop_breed_percent_values:
            for run_mutation_prob in mutation_prob_values:
                for run_max_attempts in max_attempts_values:
                    # Print status
                    run_counter += 1
                    print(f'RUN {run_counter} of {n_runs} [pop_size: {run_pop_size}] [pop_breed_percent: {run_pop_breed_percent}] [mutation_prob: {run_mutation_prob}] [max_attempts: {run_max_attempts}]')

                    # Run problem
                    eval_count = 0
                    start = timer()
                    run_state, run_fitness, run_curves = mlrose_hiive.genetic_alg(problem,
                                                                                  max_iters=MAX_ITERS,
                                                                                  random_state=SEED,
                                                                                  curve=True)
                    end = timer()

                    # Save curves and params
                    if run_fitness > fitness:
                        pop_size = run_pop_size
                        pop_breed_percent = run_pop_breed_percent
                        mutation_prob = run_mutation_prob
                        max_attempts = run_max_attempts
                        fitness = run_fitness
                        curves = run_curves
                        n_invocations = eval_count
                        time = end - start

    df = pandas.DataFrame(curves, columns=['fitness'])
    df['pop_size'] = pop_size
    df['pop_breed_percent'] = pop_breed_percent
    df['mutation_prob'] = mutation_prob
    df['max_attempts'] = max_attempts
    df['max_iters'] = MAX_ITERS
    df['n_invocations'] = n_invocations
    df['time'] = time
    df.to_csv(f'{STATS_FOLDER}/{algo}_{size}_stats.csv', index=False)

    print(f'{algo}_{size} run.')


def mimic_optimization(size):
    algo = 'mimic'
    problem = get_problem(size)
    problem.set_mimic_fast_mode(True)

    # Gridsearch params
    pop_size_values = [50, 100, 150, 200]
    keep_pct_values = [0.2, 0.3]
    max_attempts_values = [5]
    n_runs = len(max_attempts_values) * len(pop_size_values) * len(keep_pct_values)

    # Best vals
    pop_size, keep_pct,  max_attempts = None, None, None
    fitness, curves, n_invocations, time = float('-inf'), [], 0, 0

    # Gridsearch
    global eval_count
    run_counter = 0
    for run_pop_size in pop_size_values:
        for run_keep_pct in keep_pct_values:
            for run_max_attempts in max_attempts_values:
                # Print status
                run_counter += 1
                print(f'RUN {run_counter} of {n_runs} [pop_size: {run_pop_size}] [run_keep_pct: {run_keep_pct}] [max_attempts: {run_max_attempts}]')

                # Run problem
                eval_count = 0
                start = timer()
                run_state, run_fitness, run_curves = mlrose_hiive.mimic(problem,
                                                                        max_iters=MAX_ITERS,
                                                                        random_state=SEED,
                                                                        curve=True)
                end = timer()

                # Save curves and params
                if run_fitness > fitness:
                    pop_size = run_pop_size
                    keep_pct = run_keep_pct
                    max_attempts = run_max_attempts
                    fitness = run_fitness
                    curves = run_curves
                    n_invocations = eval_count
                    time = end - start

    df = pandas.DataFrame(curves, columns=['fitness'])
    df['pop_size'] = pop_size
    df['keep_pct'] = keep_pct
    df['max_attempts'] = max_attempts
    df['max_iters'] = MAX_ITERS
    df['n_invocations'] = n_invocations
    df['time'] = time
    df.to_csv(f'{STATS_FOLDER}/{algo}_{size}_stats.csv', index=False)

    print(f'{algo}_{size} run.')


def run_problems():
    for size in SIZE_VALUES:
        rhc_optimization(size)
        sa_optimization(size)
        ga_optimization(size)
        mimic_optimization(size)


if __name__ == '__main__':
    run_problems()
