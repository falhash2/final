import random
import time

def generate_random_3x3_code_goal_prioritized(goal_state):
    return [goal_state[i:i+3] for i in range(0, 9, 3)]

def generate_random_3x3_code():
    code = generate_random_3x3_code_goal_prioritized([1, 2, 3, 4, 5, 6, 7, 8, 0])
    while not is_valid_assignment(code):
        code = generate_random_3x3_code_goal_prioritized([1, 2, 3, 4, 5, 6, 7, 8, 0])
    return code

def is_valid_assignment(code):
    flat_code = [number for row in code for number in row]
    return len(set(flat_code)) == len(flat_code)

def is_goal_state(code, goal_state):
    return code == goal_state


def backtracking_search_util(code, goal_state, explored_states):
    if is_goal_state(code, goal_state):
        return code

    zero_position = find_zero_position(code)

    for value in set(goal_state[zero_position[0]]) - set(code[zero_position[0]]):
        new_code = [row[:] for row in code]
        new_code[zero_position[0]][zero_position[1]] = value

        if tuple(map(tuple, new_code)) not in explored_states:
            explored_states.add(tuple(map(tuple, new_code)))
            result = backtracking_search_util(new_code, goal_state, explored_states)

            if result is not None:
                return result

    return None

def backtracking_search(initial_state, goal_state):
    if not is_valid_assignment(initial_state):
        raise ValueError("Invalid initial state")

    result = None
    max_attempts = 10
    success_count = 0
    failure_count = 0

    while max_attempts > 0:
        result = backtracking_search_util(initial_state, goal_state, set())
        if result is not None:
            success_count += 1
            # animate_solution(animation_list)  # Commented out animation
            print("Backtracking Search: Solution found -", result)
            return result
        else:
            failure_count += 1
            print(f"No progress made, generating a new initial state. Attempts left: {max_attempts - 1}")
            initial_state = generate_random_3x3_code()
            max_attempts -= 1

    print("Backtracking Search: No solution found after multiple attempts.")
    print(f"Success Rate: {success_count / (success_count + failure_count) * 100:.2f}%")
    print(f"Failure Rate: {failure_count / (success_count + failure_count) * 100:.2f}%")
    return None

def forward_checking_util(code, goal_state, remaining_values):
    if is_valid_assignment(code) and is_goal_state(code, goal_state):
        return code

    zero_position = find_zero_position(code)

    for value in remaining_values:
        new_code = [row[:] for row in code]
        new_code[zero_position[0]][zero_position[1]] = value

        updated_remaining_values = remaining_values - {value}

        result = forward_checking_util(new_code, goal_state, updated_remaining_values)

        if result is not None:
            return result

    return None

def forward_checking(initial_state, goal_state):
    if not is_valid_assignment(initial_state):
        raise ValueError("Invalid initial state")

    remaining_values = set(goal_state[0]) - set(initial_state[0])
    result = None

    while result is None:
        result = forward_checking_util(initial_state, goal_state, remaining_values)

    print("Forward Checking: Solution found -", result)
    # animate_solution(animation_list)  # Commented out animation
    return result

def arc_consistency_ac3(code, goal_state):
    queue = initialize_queue(code)

    while queue:
        x, y = queue.pop(0)

        if revise(code, goal_state, x, y):
            if not code[x[0]][x[1]]:
                return None  # Inconsistent assignment

            neighbors = get_neighbors(x)
            for neighbor in neighbors:
                if neighbor != y:
                    queue.append((neighbor, x))

    if is_goal_state(code, goal_state):
        print("Arc-Consistency (AC-3): Solution found -", code)
        return code

    return None


def initialize_queue(code):
    return [((i, j), k) for i in range(3) for j in range(3) for k in range(3) if i != k]

def revise(code, goal_state, x, y):
    revised = False

    if not isinstance(code[x[0]][x[1]], list):
        return revised

    for value_x in code[x[0]][x[1]][:]:
        if not any(value_x == goal_state[y[0]][y[1]] and value_y == code[y[0]][y[1]][0] for value_y in code[y[0]][y[1]]):
            code[x[0]][x[1]].remove(value_x)
            revised = True

    return revised

    return revised

def get_neighbors(x):
    return [i for i in range(3) if i != x]

def find_zero_position(code):
    for i in range(3):
        for j in range(3):
            if code[i][j] == 0:
                return i, j

def test_algorithm(algorithm, instances):
    total_time = 0
    success_count = 0
    failure_count = 0

    for instance in instances:
        start_time = time.time()
        solution = algorithm(*instance)
        end_time = time.time()
        total_time += end_time - start_time

        if solution is not None:
            success_count += 1
            print(f"Instance: {instance[0]}, Solution: {solution}, Elapsed Time: {end_time - start_time:.5f} seconds - Success")
        else:
            failure_count += 1
            print(f"Instance: {instance[0]}, Solution: None, Elapsed Time: {end_time - start_time:.5f} seconds - Failure")

    print("\nSuccess Rate: {:.2f}%".format(success_count / len(instances) * 100))
    print("Failure Rate: {:.2f}%".format(failure_count / len(instances) * 100))
    print("Average Time: {:.5f} seconds\n".format(total_time / len(instances)))

# Number of instances to test
num_instances = 100

instances = [(generate_random_3x3_code(), [[1, 2, 3], [4, 5, 6], [7, 8, 0]]) for _ in range(num_instances)]

# Test Backtracking Search
print("Backtracking Search:")
backtracking_success_count = 0
backtracking_failure_count = 0
backtracking_total_time = 0

for _ in range(num_instances):
    initial_state = generate_random_3x3_code()
    goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    
    start_time = time.time()
    result = backtracking_search(initial_state, goal_state)
    end_time = time.time()
    
    if result is not None:
        backtracking_success_count += 1
    else:
        backtracking_failure_count += 1
    
    backtracking_total_time += end_time - start_time

# Print Backtracking Search results
print("\nBacktracking Search:")
print(f"Success Rate: {backtracking_success_count / num_instances * 100:.2f}%")
print(f"Failure Rate: {backtracking_failure_count / num_instances * 100:.2f}%")
print(f"Total Time: {backtracking_total_time:.5f} seconds")

# Test Forward Checking
print("\nForward Checking:")
forward_checking_success_count = 0
forward_checking_failure_count = 0
forward_checking_total_time = 0

for _ in range(num_instances):
    start_time = time.time()
    result = forward_checking(initial_state, goal_state)
    end_time = time.time()

    if result is not None:
        forward_checking_success_count += 1
    else:
        forward_checking_failure_count += 1

    forward_checking_total_time += end_time - start_time

# Print Forward Checking results
print("\nForward Checking:")
print(f"Success Rate: {forward_checking_success_count / num_instances * 100:.2f}%")
print(f"Failure Rate: {forward_checking_failure_count / num_instances * 100:.2f}%")
print(f"Total Time: {forward_checking_total_time:.5f} seconds")

# Test Arc-Consistency (AC-3)
print("\nArc-Consistency (AC-3):")
arc_consistency_success_count = 0
arc_consistency_failure_count = 0
arc_consistency_total_time = 0

for _ in range(num_instances):
    start_time = time.time()
    result = arc_consistency_ac3(initial_state, goal_state)
    end_time = time.time()

    if result is not None:
        arc_consistency_success_count += 1
    else:
        arc_consistency_failure_count += 1

    arc_consistency_total_time += end_time - start_time

# Print Arc-Consistency results
print("\nArc-Consistency (AC-3):")
print(f"Success Rate: {arc_consistency_success_count / num_instances * 100:.2f}%")
print(f"Failure Rate: {arc_consistency_failure_count / num_instances * 100:.2f}%")
print(f"Total Time: {arc_consistency_total_time:.5f} seconds")

