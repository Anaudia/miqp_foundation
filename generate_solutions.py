import gurobipy as gp
from gurobipy import GRB
import numpy as np
from tqdm import tqdm
import io
from contextlib import redirect_stderr

def generate_feasible_solutions(A, E, Q, variables_info, b_vector, d, num_objectives=3000):
    """
    Generates feasible solutions by solving quadratic programs with random positive semi-definite
    cost matrices. Then computes the cost x^T Q x for these solutions.

    Args:
        A (np.ndarray): Inequality constraint matrix (m x n).
        E (np.ndarray): Equality constraint matrix (p x n).
        Q (np.ndarray): Cost matrix (n x n).
        variables_info (list): List of variable names ('x0', 'b1', etc.).
        b_vector (np.ndarray): RHS vector for inequalities (length m).
        d (np.ndarray): RHS vector for equalities (length p).
        num_objectives (int): Number of random objective functions to set.

    Returns:
        tuple: (feasible_solutions, cost_values)
            feasible_solutions: List of unique feasible solutions as numpy arrays.
            cost_values: List of corresponding cost values (x^T Q x).
    """
    m, n = A.shape
    p = E.shape[0]
    feasible_solutions = set()
    cost_values = []

    import io
    from contextlib import redirect_stderr

    f = io.StringIO()
    with redirect_stderr(f):
        env = gp.Env(empty=True)
        env.setParam('LogToConsole', 0)
        env.setParam('OutputFlag', 0)    # Disable all Gurobi output
        env.start()

    for obj_idx in tqdm(range(num_objectives), desc="Generating Feasible Solutions"):
        try:
            model = gp.Model(f"FeasibleSolutionGenerator_{obj_idx}", env=env)

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
            for i in range(m):
                expr = gp.LinExpr()
                for j in range(n):
                    coeff = A[i, j]
                    if coeff != 0:
                        expr += coeff * variables[j]
                model.addConstr(expr <= b_vector[i], name=f"ineq_c_{i}")

            # Add equality constraints
            for i in range(p):
                expr = gp.LinExpr()
                for j in range(n):
                    coeff = E[i, j]
                    if coeff != 0:
                        expr += coeff * variables[j]
                model.addConstr(expr == d[i], name=f"eq_c_{i}")

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
                solution_tuple = tuple(np.round(solution, decimals=6))  # Round to avoid floating point issues

                if solution_tuple not in feasible_solutions:
                    feasible_solutions.add(solution_tuple)

                    # Compute the cost using the provided cost matrix Q
                    x_vec = np.array(solution)
                    cost = x_vec.T @ Q @ x_vec
                    cost_values.append(cost)

        except Exception as e:
            print(f"An error occurred for objective {obj_idx}: {e}")
            continue

    # Convert set of tuples back to list of numpy arrays
    feasible_solutions = [np.array(sol) for sol in feasible_solutions]
    return feasible_solutions, cost_values

def generate_infeasible_solutions(A, E, variables_info, b_vector, d, Q, num_infeasible_samples, feasible_solutions):
    """
    Generates a set of infeasible solutions by perturbing feasible solutions to slightly violate constraints.
    
    Args:
        A (np.ndarray): Inequality constraint matrix (m x n).
        E (np.ndarray): Equality constraint matrix (p x n).
        variables_info (list): List of variable names (e.g., ['x0', 'b1', ...]).
        b_vector (np.ndarray): RHS vector for inequalities (length m).
        d (np.ndarray): RHS vector for equalities (length p).
        Q (np.ndarray): Cost matrix (n x n).
        num_infeasible_samples (int): Number of infeasible samples to generate.
        feasible_solutions (list): List of feasible solutions to avoid duplicates.
    
    Returns:
        tuple: (infeasible_solutions, infeasible_costs)
            infeasible_solutions: List of infeasible variable assignments (numpy arrays).
            infeasible_costs: List of corresponding cost values (x^T Q x).
    """
    m, n = A.shape
    feasible_set = set([tuple(np.round(sol, decimals=5)) for sol in feasible_solutions])
    infeasible_solutions = set()
    infeasible_costs = []

    feasible_array = np.array(feasible_solutions)

    print("Generating infeasible samples...")
    with tqdm(total=int(num_infeasible_samples), desc="Infeasible Samples") as pbar:
        attempts = 0
        max_attempts = num_infeasible_samples * 10  # Prevent infinite loops
        while len(infeasible_solutions) < num_infeasible_samples and attempts < max_attempts:
            # Start with a random feasible solution
            idx = np.random.randint(len(feasible_solutions))
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
            is_feasible = np.all(Ax <= b_vector) and np.allclose(Ex, d, atol=1e-4)

            if not is_feasible:
                solution_tuple = tuple(np.round(variable_values, decimals=5))
                if solution_tuple not in feasible_set and solution_tuple not in infeasible_solutions:
                    infeasible_solutions.add(solution_tuple)
                    # Compute the cost using Q
                    cost = variable_values.T @ Q @ variable_values
                    infeasible_costs.append(cost)
                    pbar.update(1)
            attempts += 1

        if attempts == max_attempts:
            print("Reached maximum attempts while generating infeasible samples.")

    # Convert set to list of numpy arrays
    infeasible_solutions = [np.array(sol) for sol in infeasible_solutions]

    return infeasible_solutions, infeasible_costs


# def generate_infeasible_solutions(A, E, variables_info, b_vector, d, Q, num_infeasible_samples, feasible_solutions):
#     """
#     Generates a set of infeasible solutions by randomly assigning variable values
#     that violate the given constraints.

#     Args:
#         A (np.ndarray): Inequality constraint matrix (m x n).
#         E (np.ndarray): Equality constraint matrix (p x n).
#         variables_info (list): List of variable names (e.g., ['x0', 'b1', ...]).
#         b_vector (np.ndarray): RHS vector for inequalities (length m).
#         d (np.ndarray): RHS vector for equalities (length p).
#         Q (np.ndarray): Cost matrix (n x n).
#         num_infeasible_samples (int): Number of infeasible samples to generate.
#         feasible_solutions (list): List of feasible solutions to avoid duplicates.

#     Returns:
#         tuple: (infeasible_solutions, infeasible_costs)
#             infeasible_solutions: List of infeasible variable assignments (numpy arrays).
#             infeasible_costs: List of corresponding cost values (x^T Q x).
#     """
#     m, n = A.shape
#     feasible_set = set([tuple(np.round(sol, decimals=5)) for sol in feasible_solutions])
#     infeasible_solutions = set()
#     infeasible_costs = []

#     def generate_infeasible_sample():
#         """
#         Generates an infeasible sample by randomly assigning variable values and ensuring constraints are violated.

#         Returns:
#             np.ndarray: Infeasible variable assignments.
#         """
#         is_feasible = True
#         attempt = 0
#         max_attempts = 1000  # Prevent infinite loops
#         while is_feasible and attempt < max_attempts:
#             variable_values = np.zeros(n)
#             for i, var_name in enumerate(variables_info):
#                 if var_name[0] == 'x':
#                     # Continuous variable within feasible range
#                     variable_values[i] = np.random.uniform(0, 1)
#                 else:
#                     # Binary variable (0 or 1)
#                     variable_values[i] = np.random.randint(0, 2)

#             Ax = A @ variable_values
#             Ex = E @ variable_values
#             is_feasible = np.all(Ax <= b_vector) and np.allclose(Ex, d, atol=1e-4)
#             if is_feasible:
#                 # Adjust variables involved in the equality constraint to violate it
#                 indices_in_E = np.where(E[0] != 0)[0]
#                 variable_values[indices_in_E] += np.random.uniform(-0.5, 0.5, size=len(indices_in_E))
#                 # Recalculate Ax and Ex after adjustment
#                 Ax = A @ variable_values
#                 Ex = E @ variable_values
#                 is_feasible = np.all(Ax <= b_vector) and np.allclose(Ex, d, atol=1e-4)
#             attempt += 1

#         if attempt == max_attempts:
#             raise ValueError("Failed to generate an infeasible sample after maximum attempts.")

#         return variable_values

#     print("Generating infeasible samples...")
#     with tqdm(total=num_infeasible_samples, desc="Infeasible Samples") as pbar:
#         while len(infeasible_solutions) < num_infeasible_samples:
#             x_combined = generate_infeasible_sample()
#             solution_tuple = tuple(np.round(x_combined, decimals=5))
#             if solution_tuple not in feasible_set and solution_tuple not in infeasible_solutions:
#                 infeasible_solutions.add(solution_tuple)
#                 # Compute the cost using Q
#                 cost = x_combined.T @ Q @ x_combined
#                 infeasible_costs.append(cost)
#                 pbar.update(1)

#     # Convert set to list of numpy arrays
#     infeasible_solutions = [np.array(sol) for sol in infeasible_solutions]

#     return infeasible_solutions, infeasible_costs