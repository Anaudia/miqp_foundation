import re
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def parse_qplib_file(number):
    """
    Parses a QPLIB problem file and extracts matrices and vectors for optimization.

    Args:
        number (str): The file number or filename (without extension) to parse.

    Returns:
        dict: A dictionary containing the following keys:
            - A (np.ndarray): Inequality constraint matrix (Ax ≤ b).
            - b_vector (np.ndarray): RHS vector for inequalities.
            - E (np.ndarray): Equality constraint matrix (Ex = d).
            - d (np.ndarray): RHS vector for equalities.
            - Q (np.ndarray): Cost matrix for quadratic objective.
            - variables_info (list): List of variable names (e.g., ['x0', 'b1', ...]).
            - binary_indices (list): Indices of binary variables in variables_info.
            - variable_indices (dict): Mapping from variable names to their indices.
    """
    # Initialize variables
    objective_section = False
    constraints_section = False
    binary_section = False

    objective_lines = []
    constraint_lines = []
    binary_lines = []

    # Read the file
    with open(f'QPLIB_{number}.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Minimize'):
                objective_section = True
                constraints_section = False
                binary_section = False
                continue
            elif line.startswith('Subject To'):
                objective_section = False
                constraints_section = True
                binary_section = False
                continue
            elif line.startswith('Binary'):
                objective_section = False
                constraints_section = False
                binary_section = True
                continue
            elif line.startswith('End'):
                objective_section = False
                constraints_section = False
                binary_section = False
                continue

            if objective_section:
                objective_lines.append(line)
            elif constraints_section:
                constraint_lines.append(line)
            elif binary_section:
                binary_lines.append(line)

    # Collect variables
    variables_set = set()

    # From the 'Binary' section
    binary_text = ' '.join(binary_lines)
    binary_vars = re.findall(r'\b[b,x]\d+\b', binary_text)
    variables_set.update(binary_vars)

    # From the 'Subject To' section
    constraint_text = ' '.join(constraint_lines)
    vars_in_constraints = re.findall(r'\b[b,x]\d+\b', constraint_text)
    variables_set.update(vars_in_constraints)

    # From the objective function
    objective_text = ' '.join(objective_lines)
    vars_in_objective = re.findall(r'\b[b,x]\d+\b', objective_text)
    variables_set.update(vars_in_objective)

    # Function to extract variable type and number
    def extract_var_num(var_name):
        m = re.match(r'([bx])(\d+)', var_name)
        if m:
            var_type = m.group(1)
            var_num = int(m.group(2))
            return var_type, var_num
        else:
            return None, None

    # Create list of variables with type and number
    variables_info = []
    for var in variables_set:
        var_type, var_num = extract_var_num(var)
        if var_type is not None:
            variables_info.append((var_type, var_num, var))

    # Sort variables: 'x' variables first, then 'b' variables, both sorted by number
    variables_info.sort(key=lambda x: (0 if x[0] == 'x' else 1, x[1]))

    # Create variable_indices mapping
    variable_indices = {var_info[2]: idx for idx, var_info in enumerate(variables_info)}
    n = len(variable_indices)

    # Now, parse the objective function
    # Extract the quadratic terms inside the square brackets
    quadratic_match = re.search(r'\[(.*)\]/2', objective_text)
    if quadratic_match:
        quadratic_text = quadratic_match.group(1)
    else:
        print("No quadratic terms found in the objective function.")
        quadratic_text = ''

    # Replace '-' with '+ -' to split terms correctly
    quadratic_text = quadratic_text.replace('-', '+ -')
    terms = quadratic_text.split('+')

    Q = np.zeros((n, n))

    # Parse quadratic terms and build Q matrix
    for term in terms:
        term = term.strip()
        if term == '':
            continue
        # Match terms like '52.8828 x2^2' or '-63.7552 x2 * x3'
        # First, split the term to separate the coefficient
        match = re.match(r'([+-]?\s*\d*\.?\d+(?:[eE][+-]?\d+)?)\s*(.*)', term)
        if match:
            coeff_str = match.group(1).replace(' ', '')
            coeff = float(coeff_str)
            vars_part = match.group(2)
            vars_part = vars_part.strip()
            # Check for squared terms
            squared_match = re.match(r'([bx]\d+)\s*\^\s*2', vars_part)
            if squared_match:
                var_name = squared_match.group(1)
                if var_name in variable_indices:
                    i = variable_indices[var_name]
                    # Update Q[i, i] with the full coefficient
                    Q[i, i] += coeff
                else:
                    print(f"Variable {var_name} not found.")
            else:
                # Check for cross terms
                cross_match = re.match(r'([bx]\d+)\s*\*\s*([bx]\d+)', vars_part)
                if not cross_match:
                    # Try matching 'x2*x3' without spaces
                    cross_match = re.match(r'([bx]\d+)\*(?:\s*)([bx]\d+)', vars_part)
                if cross_match:
                    var1_name = cross_match.group(1)
                    var2_name = cross_match.group(2)
                    if var1_name in variable_indices and var2_name in variable_indices:
                        i = variable_indices[var1_name]
                        j = variable_indices[var2_name]
                        # Divide the coefficient by 2 for cross terms
                        adjusted_coeff = coeff / 2.0
                        # Update Q[i, j] and Q[j, i]
                        Q[i, j] += adjusted_coeff
                        Q[j, i] += adjusted_coeff
                    else:
                        print(f"Variables {var1_name} or {var2_name} not found.")
                else:
                    print(f"Could not parse term: {term}")
        else:
            print(f"Could not parse term: {term}")

    # Since the entire quadratic term is divided by 2 in the objective, we adjust Q accordingly
    Q = (1/2) * Q

    # ==============================
    # Process the constraints
    # ==============================

    # First, combine multi-line constraints into full constraints
    full_constraints = []
    i = 0
    while i < len(constraint_lines):
        line = constraint_lines[i].strip()
        # Skip empty lines or comments
        if not line or line.startswith('\\'):
            i += 1
            continue
        # Handle constraints that may span multiple lines
        while not re.search(r'(=|<=|>=)\s*[\d\.]+$', line) and (i + 1) < len(constraint_lines):
            i += 1
            next_line = constraint_lines[i].strip()
            line += ' ' + next_line
        full_constraints.append(line)
        i += 1

    # Now process the full_constraints without modifying the list
    constraints_eq = []
    constraints_leq = []
    constraints_geq = []

    for line in full_constraints:
        # Remove constraint name if any
        if ':' in line:
            _, rest = line.split(':', 1)
            rest = rest.strip()
        else:
            rest = line
        # Split on operators
        operator_match = re.match(r'(.+?)(<=|>=|=)(.+)', rest)
        if operator_match:
            lhs = operator_match.group(1).strip()
            operator = operator_match.group(2)
            rhs = operator_match.group(3).strip()
        else:
            print(f"Could not parse constraint line: {line}")
            continue
        # Parse the LHS to extract variables
        # Replace '-' with '+ -' to split terms correctly
        lhs = lhs.replace('-', '+ -')
        terms = lhs.split('+')
        coeffs = {}
        for term in terms:
            term = term.strip()
            if term == '':
                continue
            # Match coefficient and variable
            term_match = re.match(r'([+-]?\s*\d*\.?\d*)\s*([bx]\d+)', term)
            if term_match:
                coeff_str = term_match.group(1).replace(' ', '')
                if coeff_str in ['', '+']:
                    coeff = 1.0
                elif coeff_str == '-':
                    coeff = -1.0
                else:
                    coeff = float(coeff_str)
                var_name = term_match.group(2)
            else:
                # Assume coefficient of 1
                coeff = 1.0
                var_name = term
                if var_name.startswith('-'):
                    coeff = -1.0
                    var_name = var_name[1:].strip()
            if var_name in variable_indices:
                idx = variable_indices[var_name]
                coeffs[idx] = coeffs.get(idx, 0) + coeff
            else:
                print(f"Variable {var_name} not found in constraints.")
        # Get RHS value
        rhs_value = float(rhs)
        # Store the coefficients and RHS value
        if operator == '=':
            constraints_eq.append((coeffs, rhs_value))
        elif operator == '<=':
            constraints_leq.append((coeffs, rhs_value))
        elif operator == '>=':
            constraints_geq.append((coeffs, rhs_value))
        else:
            print(f"Unknown operator {operator} in constraint.")
            continue

    # Build E matrix and d vector for equality constraints
    m_eq = len(constraints_eq)
    E = np.zeros((m_eq, n))
    d = np.zeros(m_eq)

    for row_idx, (coeffs, rhs_value) in enumerate(constraints_eq):
        for idx, coeff in coeffs.items():
            E[row_idx, idx] = coeff
        d[row_idx] = rhs_value

    # Build A matrix and b vector for inequality constraints (<=)
    m_leq = len(constraints_leq)
    A_leq = np.zeros((m_leq, n))
    b_leq = np.zeros(m_leq)

    for row_idx, (coeffs, rhs_value) in enumerate(constraints_leq):
        for idx, coeff in coeffs.items():
            A_leq[row_idx, idx] = coeff
        b_leq[row_idx] = rhs_value

    # Build G matrix and h vector for inequality constraints (>=), converting them to -A x ≤ -b
    m_geq = len(constraints_geq)
    A_geq = np.zeros((m_geq, n))
    b_geq = np.zeros(m_geq)

    for row_idx, (coeffs, rhs_value) in enumerate(constraints_geq):
        for idx, coeff in coeffs.items():
            A_geq[row_idx, idx] = -coeff
        b_geq[row_idx] = -rhs_value

    # Combine inequality constraints
    A = np.vstack([A_leq, A_geq])
    b_vector = np.hstack([b_leq, b_geq])

    # Extract binary variables
    binary_vars_in_text = re.findall(r'\b[b,x]\d+\b', binary_text)
    binary_indices = []
    for var in binary_vars_in_text:
        if var in variable_indices:
            idx = variable_indices[var]
            binary_indices.append(idx)
        else:
            print(f"Binary variable {var} not found.")

    # Prepare variables_info list (variable names only)
    variables_info_list = [var_info[2] for var_info in variables_info]

    # Return all relevant data structures in a dictionary
    return {
        'A': A,
        'b_vector': b_vector,
        'E': E,
        'd': d,
        'Q': Q,
        'variables_info': variables_info_list,
        'binary_indices': binary_indices,
        'variable_indices': variable_indices
    }


def my_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        # Feasible integer solution found
        try:
            solution = model.cbGetSolution(model.getVars())
            # print("Feasible solution found:", solution)
            model._feasible_solutions.append(solution)
        except gp.GurobiError as e:
            print(f"Error retrieving feasible solution: {e}")
    
    elif where == GRB.Callback.MIPNODE:
        try:
            # Check node status
            node_status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
            
            if node_status == GRB.Status.OPTIMAL:
                # Optimal relaxation available
                relaxation_solution = model.cbGetNodeRel(model.getVars())
                #is_feasible = test_relaxation_feasibility(model, relaxation_solution)
                # print(f"Optimal relaxation solution is {is_feasible} and the solution is: {relaxation_solution}")
                # model._relaxation_solutions.append({"status": "Optimal",  "feasiable": is_feasible, "solution": relaxation_solution})
                
                # print(f"Optimal relaxation solution is: {relaxation_solution}")
                model._relaxation_solutions.append({"status": "Optimal", "solution": relaxation_solution})
            
            elif node_status == GRB.Status.INFEASIBLE:
                # Node relaxation is infeasible
                # print("Infeasible relaxation at node.")
                model._relaxation_solutions.append({"status": "Infeasible", "solution": None})
            
            elif node_status == GRB.Status.UNBOUNDED:
                # Node relaxation is unbounded
                # print("Unbounded relaxation at node.")
                model._relaxation_solutions.append({"status": "Unbounded", "solution": None})
            
            elif node_status == GRB.Status.INF_OR_UNBD:
                # Node relaxation is infeasible or unbounded
                # print("Infeasible or unbounded relaxation at node.")
                model._relaxation_solutions.append({"status": "InfOrUnbd", "solution": None})
            
            else:
                # print(f"Unhandled node status: {node_status}")
                model._relaxation_solutions.append({"status": "Unknown", "solution": None})
        
        except gp.GurobiError as e:
            pass
            # print(f"Error retrieving relaxation solution: {e}")

