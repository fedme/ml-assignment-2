import numpy as np
import pandas
import mlrose_hiive

PROBLEM_NAME = 'knapsack'
eval_count = 0
orig_fitness_func = None


def fitness_func(state):
    global eval_count
    eval_count += 1
    return orig_fitness_func.evaluate(state)


def get_problem():
    global orig_fitness_func

    seed = 42
    number_of_items_types = 100
    max_weight_per_item = 25
    max_value_per_item = 10
    max_weight_pct = 0.35
    max_item_count = 10
    multiply_by_max_item_count = True
    np.random.seed(seed)
    weights = 1 + np.random.randint(max_weight_per_item, size=number_of_items_types)
    values = 1 + np.random.randint(max_value_per_item, size=number_of_items_types)
    orig_fitness_func = mlrose_hiive.Knapsack(weights, values, max_weight_pct=max_weight_pct, max_item_count=max_item_count, multiply_by_max_item_count=multiply_by_max_item_count)

    fitness = mlrose_hiive.CustomFitness(fitness_func)
    problem = mlrose_hiive.DiscreteOpt(length=number_of_items_types, fitness_fn=fitness, maximize=True)
    return problem


def rhc_optimization():
    algo = 'rhc'
    problem = get_problem()

    # Gridsearch params
    max_iters = 500
    restarts_values = [10, 30, 50]
    max_attempts_values = [10, 50, 100, 200]
    n_runs = len(restarts_values) * len(max_attempts_values)

    # Best vals
    restarts, max_attempts = None, None
    fitness, curves, n_invocations = float('-inf'), [], 0

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
            run_state, run_fitness, run_curves = mlrose_hiive.random_hill_climb(problem,
                                                                                restarts=run_restarts,
                                                                                max_attempts=run_max_attempts,
                                                                                max_iters=max_iters,
                                                                                random_state=42,
                                                                                curve=True)
            # Save curves and params
            if run_fitness > fitness:
                restarts = run_restarts
                max_attempts = run_max_attempts
                fitness = run_fitness
                curves = run_curves
                n_invocations = eval_count

    df = pandas.DataFrame(curves, columns=['fitness'])
    df['restarts'] = restarts
    df['max_attempts'] = max_attempts
    df['max_iters'] = max_iters
    df['n_invocations'] = n_invocations
    df.to_csv(f'{algo}_stats.csv', index=False)

    print(f'{algo} run.')


def sa_optimization():
    algo = 'sa'
    problem = get_problem()

    # Gridsearch params
    max_iters = 500
    schedule_values = [mlrose_hiive.GeomDecay(), mlrose_hiive.ArithDecay(), mlrose_hiive.algorithms.decay.ExpDecay()]
    max_attempts_values = [10, 50, 100, 200]
    n_runs = len(schedule_values) * len(max_attempts_values)

    # Best vals
    schedule, max_attempts = None, None
    fitness, curves, n_invocations = float('-inf'), [], 0

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
            run_state, run_fitness, run_curves = mlrose_hiive.simulated_annealing(problem,
                                                                                  schedule=run_schedule,
                                                                                  max_attempts=run_max_attempts,
                                                                                  max_iters=max_iters,
                                                                                  random_state=42,
                                                                                  curve=True)
            # Save curves and params
            if run_fitness > fitness:
                schedule = run_schedule.__class__.__name__
                max_attempts = run_max_attempts
                fitness = run_fitness
                curves = run_curves
                n_invocations = eval_count

    df = pandas.DataFrame(curves, columns=['fitness'])
    df['schedule'] = schedule
    df['max_attempts'] = max_attempts
    df['max_iters'] = max_iters
    df['n_invocations'] = n_invocations
    df.to_csv(f'{algo}_stats.csv', index=False)

    print(f'{algo} run.')


def ga_optimization():
    algo = 'ga'
    problem = get_problem()

    # Gridsearch params
    max_iters = 500
    max_attempts_values = [10, 50, 100, 200]
    pop_size_values = [50, 100, 150, 200]
    pop_breed_percent_values = [0.5, 0.75]
    mutation_prob_values = [0.1, 0.3]
    n_runs = len(max_attempts_values) * len(pop_size_values) * len(pop_breed_percent_values) * len(mutation_prob_values)

    # Best vals
    pop_size, pop_breed_percent, mutation_prob, max_attempts = None, None, None, None
    fitness, curves, n_invocations = float('-inf'), [], 0

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
                    run_state, run_fitness, run_curves = mlrose_hiive.genetic_alg(problem,
                                                                                  pop_size=run_pop_size,
                                                                                  pop_breed_percent=run_pop_breed_percent,
                                                                                  mutation_prob=run_mutation_prob,
                                                                                  max_attempts=run_max_attempts,
                                                                                  max_iters=max_iters,
                                                                                  random_state=42,
                                                                                  curve=True)
                    # Save curves and params
                    if run_fitness > fitness:
                        pop_size = run_pop_size
                        pop_breed_percent = run_pop_breed_percent
                        mutation_prob = run_mutation_prob
                        max_attempts = run_max_attempts
                        fitness = run_fitness
                        curves = run_curves
                        n_invocations = eval_count

    df = pandas.DataFrame(curves, columns=['fitness'])
    df['pop_size'] = pop_size
    df['pop_breed_percent'] = pop_breed_percent
    df['mutation_prob'] = mutation_prob
    df['max_attempts'] = max_attempts
    df['max_iters'] = max_iters
    df['n_invocations'] = n_invocations
    df.to_csv(f'{algo}_stats.csv', index=False)

    print(f'{algo} run.')


def mimic_optimization():
    algo = 'mimic'
    problem = get_problem()
    problem.set_mimic_fast_mode(True)

    # Gridsearch params
    max_iters = 500
    pop_size_values = [50, 100, 150, 200]
    keep_pct_values = [0.2, 0.3]
    max_attempts_values = [10, 50, 100, 200]
    n_runs = len(max_attempts_values) * len(pop_size_values) * len(keep_pct_values)

    # Best vals
    pop_size, keep_pct,  max_attempts = None, None, None
    fitness, curves, n_invocations = float('-inf'), [], 0

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
                run_state, run_fitness, run_curves = mlrose_hiive.mimic(problem,
                                                                        pop_size=run_pop_size,
                                                                        keep_pct=run_keep_pct,
                                                                        max_attempts=run_max_attempts,
                                                                        max_iters=max_iters,
                                                                        random_state=42,
                                                                        curve=True)
                # Save curves and params
                if run_fitness > fitness:
                    pop_size = run_pop_size
                    keep_pct = run_keep_pct
                    max_attempts = run_max_attempts
                    fitness = run_fitness
                    curves = run_curves
                    n_invocations = eval_count

    df = pandas.DataFrame(curves, columns=['fitness'])
    df['pop_size'] = pop_size
    df['keep_pct'] = keep_pct
    df['max_attempts'] = max_attempts
    df['max_iters'] = max_iters
    df['n_invocations'] = n_invocations
    df.to_csv(f'{algo}_stats.csv', index=False)

    print(f'{algo} run.')


if __name__ == '__main__':
    rhc_optimization()
    sa_optimization()
    ga_optimization()
    mimic_optimization()
