import os
import pickle
import gurobipy as gp
from gurobipy import GRB
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import io
from contextlib import redirect_stderr

def generate_feasible_solutions(A, E, Q, variables_info, b_vector, d_vector, num_objectives=3000):
    """
    Generates feasible solutions by solving quadratic programs with random positive semi-definite
    cost matrices. Then computes the cost x^T Q x for these solutions using the provided Q.

    Args:
        A (np.ndarray): Inequality constraint matrix (m x n).
        E (np.ndarray): Equality constraint matrix (p x n).
        Q (np.ndarray): Cost matrix (n x n).
        variables_info (list): List of variable names ('x0', 'b1', etc.).
        b_vector (np.ndarray): RHS vector for inequalities (length m).
        d_vector (np.ndarray): RHS vector for equalities (length p).
        num_objectives (int): Number of random objective functions to set.

    Returns:
        tuple: (feasible_solutions, cost_values)
            feasible_solutions: List of feasible solutions as numpy arrays.
            cost_values: List of corresponding cost values (x^T Q x).
    """
    def solve_random_qp(args):
        A, E, Q, variables_info, b_vector, d_vector, idx = args
        n = A.shape[1]
        try:
            # Create Gurobi environment inside the function
            with io.StringIO() as f, redirect_stderr(f):
                env = gp.Env(empty=True)
                env.setParam('LogToConsole', 0)
                env.setParam('OutputFlag', 0)    # Disable all Gurobi output
                env.start()

                model = gp.Model(f"FeasibleSolutionGenerator_{idx}", env=env)

                # Create variables
                variables = []
                for i, var_name in enumerate(variables_info):
                    if var_name[0] == 'x':
                        # Continuous variable between lower and upper bounds (adjust as needed)
                        var = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=var_name)
                    else:
                        # Binary variable
                        var = model.addVar(vtype=GRB.BINARY, name=var_name)
                    variables.append(var)

                model.update()

                # Add inequality constraints
                m = A.shape[0]
                for i in range(m):
                    expr = gp.LinExpr()
                    for j in range(n):
                        coeff = A[i, j]
                        if coeff != 0:
                            expr += coeff * variables[j]
                    model.addConstr(expr <= b_vector[i], name=f"ineq_c_{i}")

                # Add equality constraints
                p = E.shape[0]
                for i in range(p):
                    expr = gp.LinExpr()
                    for j in range(n):
                        coeff = E[i, j]
                        if coeff != 0:
                            expr += coeff * variables[j]
                    model.addConstr(expr == d_vector[i], name=f"eq_c_{i}")

                model.update()

                # Generate a random symmetric positive semi-definite matrix Q_random
                random_matrix = np.random.randn(n, n)
                Q_random = np.dot(random_matrix.T, random_matrix)  # Ensures positive semi-definite
                Q_random = 0.5 * (Q_random + Q_random.T)  # Ensure symmetry

                # Set the quadratic objective function using Q_random
                obj = gp.QuadExpr()
                for i in range(n):
                    for j in range(n):
                        coeff = Q_random[i, j]
                        if coeff != 0:
                            obj += coeff * variables[i] * variables[j]
                model.setObjective(obj, GRB.MINIMIZE)

                # Optimize the model
                model.optimize()

                if model.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
                    solution = model.getAttr('X', variables)
                    solution_array = np.array(solution)
                    # Compute the cost using the provided cost matrix Q
                    cost = solution_array.T @ Q @ solution_array

                    return (solution_array, cost)
                else:
                    return None
        except Exception as e:
            # You can log the exception if needed
            return None

    # Prepare arguments for parallel execution
    args_list = [(A, E, Q, variables_info, b_vector, d_vector, idx) for idx in range(num_objectives)]

    # Use joblib Parallel to run solve_random_qp in parallel
    results = Parallel(n_jobs=-1)(
        delayed(solve_random_qp)(args) for args in tqdm(args_list, desc="Generating Feasible Solutions")
    )

    # Filter out None results and extract solutions and costs
    feasible_solutions_and_costs = [res for res in results if res is not None]

    # Remove duplicate solutions
    solutions_set = set()
    unique_feasible_solutions_and_costs = []
    for solution, cost in feasible_solutions_and_costs:
        solution_tuple = tuple(np.round(solution, decimals=6))
        if solution_tuple not in solutions_set:
            solutions_set.add(solution_tuple)
            unique_feasible_solutions_and_costs.append((solution, cost))

    # Unpack the solutions and costs
    feasible_solutions = [item[0] for item in unique_feasible_solutions_and_costs]
    cost_values = [item[1] for item in unique_feasible_solutions_and_costs]

    return feasible_solutions, cost_values

def generate_infeasible_solutions(A, E, variables_info, b_vector, d_vector, Q, num_infeasible_samples, feasible_solutions):
    """
    Generates a set of infeasible solutions by perturbing feasible solutions to slightly violate constraints.
    
    Args:
        A (np.ndarray): Inequality constraint matrix (m x n).
        E (np.ndarray): Equality constraint matrix (p x n).
        variables_info (list): List of variable names (e.g., ['x0', 'b1', ...]).
        b_vector (np.ndarray): RHS vector for inequalities (length m).
        d_vector (np.ndarray): RHS vector for equalities (length p).
        Q (np.ndarray): Cost matrix (n x n).
        num_infeasible_samples (int): Number of infeasible samples to generate.
        feasible_solutions (list): List of feasible solutions to avoid duplicates.
    
    Returns:
        tuple: (infeasible_solutions, infeasible_costs)
            infeasible_solutions: List of infeasible variable assignments (numpy arrays).
            infeasible_costs: List of corresponding cost values (x^T Q x).
    """
    def generate_single_infeasible_sample(args):
        A, E, variables_info, b_vector, d_vector, Q, feasible_array = args
        n = A.shape[1]
        attempts = 0
        max_attempts_per_process = 10
        while attempts < max_attempts_per_process:
            # Start with a random feasible solution
            idx = np.random.randint(len(feasible_array))
            feasible_sample = feasible_array[idx].copy()

            # Perturb the feasible solution to make it infeasible
            perturbation = np.random.normal(loc=0, scale=0.1, size=n)  # Small perturbation
            variable_values = feasible_sample + perturbation

            # Ensure variable values are within feasible variable ranges
            for i, var_name in enumerate(variables_info):
                if var_name[0] == 'x':
                    # Continuous variable between 0 and 1 (adjust bounds as needed)
                    variable_values[i] = np.clip(variable_values[i], 0, 1)
                else:
                    # Binary variable
                    variable_values[i] = np.round(variable_values[i])
                    variable_values[i] = np.clip(variable_values[i], 0, 1)

            # Check feasibility
            Ax = A @ variable_values
            Ex = E @ variable_values
            is_feasible = np.all(Ax <= b_vector) and np.allclose(Ex, d_vector, atol=1e-4)

            if not is_feasible:
                # Compute the cost using Q
                cost = variable_values.T @ Q @ variable_values
                return (variable_values, cost)
            attempts += 1
        return None

    from joblib import Parallel, delayed

    feasible_array = np.array(feasible_solutions)
    n = A.shape[1]
    # We may need to generate more samples than needed due to duplicates
    oversampling_factor = 1
    num_attempts = num_infeasible_samples * oversampling_factor

    args_list = [(A, E, variables_info, b_vector, d_vector, Q, feasible_array) for _ in range(num_attempts)]

    print("Generating infeasible samples...")
    results = Parallel(n_jobs=-1)(
        delayed(generate_single_infeasible_sample)(args) for args in tqdm(args_list, desc="Infeasible Samples")
    )

    # Filter out None results
    infeasible_solutions_and_costs = [res for res in results if res is not None]

    # Remove duplicates
    infeasible_solutions_set = set()
    unique_infeasible_solutions_and_costs = []
    for solution, cost in infeasible_solutions_and_costs:
        solution_tuple = tuple(np.round(solution, decimals=5))
        if solution_tuple not in infeasible_solutions_set:
            infeasible_solutions_set.add(solution_tuple)
            unique_infeasible_solutions_and_costs.append((solution, cost))
            if len(unique_infeasible_solutions_and_costs) >= num_infeasible_samples:
                break

    # Unpack solutions and costs
    infeasible_solutions = [item[0] for item in unique_infeasible_solutions_and_costs]
    infeasible_costs = [item[1] for item in unique_infeasible_solutions_and_costs]

    return infeasible_solutions, infeasible_costs


def generate_infeasible_nonintegral_solutions(A, E, variables_info, b_vector, d_vector, Q, num_infeasible_nonintegral_samples):
    """
    Generates solutions that are both infeasible and non-integral with a twist:
    We first randomly decide how many binary variables remain binary (0 or 1) and how many are treated as continuous.
    For those treated as binary, we sample them as 0 or 1.
    For those treated as continuous, we sample from [0,1].
    Continuous variables (x-prefix) are always from [0,1].

    Non-integral means at least one binary variable designated as "non-binary" is not close to 0 or 1.
    """

    # Identify binary variables
    binary_indices = [i for i, var_name in enumerate(variables_info) if var_name[0] != 'x']
    n_binary = len(binary_indices)
    n = A.shape[1]

    def is_integral_for_binary(val, epsilon=1e-3):
        return (abs(val - 0) < epsilon) or (abs(val - 1) < epsilon)

    def generate_sample(args):
        A, E, variables_info, b_vector, d_vector, Q = args
        # If there are no binary variables, the concept of integral vs. non-integral doesn't apply
        if n_binary == 0:
            return None

        # Randomly decide how many binary variables remain truly binary
        # Range is [0, n_binary-1]
        k = np.random.randint(0, n_binary) if n_binary > 1 else 0

        # Randomly pick which binary variables will remain binary
        binary_vars_to_remain_binary = np.random.choice(binary_indices, size=k, replace=False)
        binary_vars_to_remain_binary_set = set(binary_vars_to_remain_binary)

        variable_values = np.zeros(n)
        for i, var_name in enumerate(variables_info):
            if var_name[0] == 'x':
                # Continuous variable in [0,1]
                variable_values[i] = np.random.uniform(0, 1)
            else:
                # Binary variable
                if i in binary_vars_to_remain_binary_set:
                    # Remain binary: set exactly 0 or 1
                    variable_values[i] = np.random.choice([0, 1])
                else:
                    # Treat as continuous from [0,1]
                    variable_values[i] = np.random.uniform(0, 1)

        # Check non-integrality
        # Non-integral means that at least one binary variable NOT in binary_vars_to_remain_binary_set is not integral
        # i.e., it's sampled continuously and not close to 0 or 1.
        non_binary_vars = set(binary_indices) - binary_vars_to_remain_binary_set
        if non_binary_vars:
            # Check if at least one non-binary variable is truly non-integral
            non_integral_found = False
            for idx_b in non_binary_vars:
                val = variable_values[idx_b]
                if not is_integral_for_binary(val):
                    non_integral_found = True
                    break
            if not non_integral_found:
                # All "non-binary" vars ended up close to integer values by chance -> discard
                return None
        else:
            # If somehow k == n_binary (all are binary), then we have no "non-binary" vars
            # Without non-binary vars, the solution can't be non-integral by this definition.
            return None

        # Check feasibility
        Ax = A @ variable_values
        Ex = E @ variable_values
        feas_tolerance = 1e-4
        feasible_ineq = np.all(Ax <= b_vector + feas_tolerance)
        feasible_eq = np.allclose(Ex, d_vector, atol=feas_tolerance)

        if feasible_ineq and feasible_eq:
            # It's feasible, we don't want that
            return None

        cost = variable_values.T @ Q @ variable_values
        return (variable_values, cost)

    oversampling_factor = 1
    attempts = num_infeasible_nonintegral_samples * oversampling_factor
    args_list = [(A, E, variables_info, b_vector, d_vector, Q) for _ in range(attempts)]

    print("Generating infeasible, non-integral samples with mixed binary settings...")
    results = Parallel(n_jobs=-1)(
        delayed(generate_sample)(args) for args in tqdm(args_list, desc="Infeasible Non-Integral Mixed")
    )

    # Filter out None results
    infeasible_nonintegral_solutions_and_costs = [res for res in results if res is not None]

    # Remove duplicates
    infeasible_nonintegral_solutions_set = set()
    unique_infeasible_nonintegral_solutions_and_costs = []
    for solution, cost in infeasible_nonintegral_solutions_and_costs:
        solution_tuple = tuple(np.round(solution, decimals=5))
        if solution_tuple not in infeasible_nonintegral_solutions_set:
            infeasible_nonintegral_solutions_set.add(solution_tuple)
            unique_infeasible_nonintegral_solutions_and_costs.append((solution, cost))
            if len(unique_infeasible_nonintegral_solutions_and_costs) >= num_infeasible_nonintegral_samples:
                break

    # Unpack solutions and costs
    infeasible_nonintegral_solutions = [item[0] for item in unique_infeasible_nonintegral_solutions_and_costs]
    infeasible_nonintegral_costs = [item[1] for item in unique_infeasible_nonintegral_solutions_and_costs]

    return infeasible_nonintegral_solutions, infeasible_nonintegral_costs

def load_or_generate_solutions(
    generate_new,
    feasible_data_file,
    infeasible_data_file,
    infeasible_nonintegral_data_file,
    A, E, Q, variables_info, b_vector, d_vector,
    generate_feasible_solutions, generate_infeasible_solutions, generate_infeasible_nonintegral_solutions
):
    # Load or generate feasible solutions
    if not generate_new and os.path.exists(feasible_data_file):
        # Load feasible data
        with open(feasible_data_file, 'rb') as f:
            feasible_data = pickle.load(f)
        feasible_solutions = feasible_data['solutions']
        feasible_costs = feasible_data['costs']
        print("Loaded existing feasible solutions from file.")
    else:
        # Adjust the number as needed. For example, let's say we want to generate
        # as many feasible solutions as the dimension of Q or a fixed number.
        # If feasible_solutions is unknown, pick a number or base it on something else:
        num_objectives = Q.shape[0]  # or a fixed number
        feasible_solutions, feasible_costs = generate_feasible_solutions(
            A, E, Q, variables_info, b_vector, d_vector, num_objectives
        )
        # Save the generated data for future use
        feasible_data = {'solutions': feasible_solutions, 'costs': feasible_costs}
        with open(feasible_data_file, 'wb') as f:
            pickle.dump(feasible_data, f)
        print("Generated and saved feasible solutions.")

    # Load or generate infeasible solutions
    if not generate_new and os.path.exists(infeasible_data_file):
        # Load infeasible data
        with open(infeasible_data_file, 'rb') as f:
            infeasible_data = pickle.load(f)
        infeasible_solutions = infeasible_data['solutions']
        infeasible_costs = infeasible_data['costs']
        print("Loaded existing infeasible solutions from file.")
    else:
        num_infeasible_samples = len(feasible_solutions)  # Adjust as needed
        infeasible_solutions, infeasible_costs = generate_infeasible_solutions(
            A, E, variables_info, b_vector, d_vector, Q, num_infeasible_samples, feasible_solutions
        )
        # Save the generated data for future use
        infeasible_data = {'solutions': infeasible_solutions, 'costs': infeasible_costs}
        with open(infeasible_data_file, 'wb') as f:
            pickle.dump(infeasible_data, f)
        print("Generated and saved infeasible solutions.")

    # Load or generate infeasible non-integral solutions
    if not generate_new and os.path.exists(infeasible_nonintegral_data_file):
        # Load infeasible non-integral data
        with open(infeasible_nonintegral_data_file, 'rb') as f:
            infeasible_nonintegral_data = pickle.load(f)
        infeasible_nonintegral_solutions = infeasible_nonintegral_data['solutions']
        infeasible_nonintegral_costs = infeasible_nonintegral_data['costs']
        print("Loaded existing infeasible non-integral solutions from file.")
    else:
        num_infeasible_nonintegral_samples = len(feasible_solutions)  # Adjust as needed
        infeasible_nonintegral_solutions, infeasible_nonintegral_costs = generate_infeasible_nonintegral_solutions(
            A, E, variables_info, b_vector, d_vector, Q, num_infeasible_nonintegral_samples
        )
        # Save the generated data for future use
        infeasible_nonintegral_data = {
            'solutions': infeasible_nonintegral_solutions,
            'costs': infeasible_nonintegral_costs
        }
        with open(infeasible_nonintegral_data_file, 'wb') as f:
            pickle.dump(infeasible_nonintegral_data, f)
        print("Generated and saved infeasible non-integral solutions.")

    return (feasible_solutions, feasible_costs,
            infeasible_solutions, infeasible_costs,
            infeasible_nonintegral_solutions, infeasible_nonintegral_costs)

