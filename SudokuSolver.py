import random
import time
import multiprocessing
from functools import partial

class TimeoutError(Exception):
    pass

def run_with_timeout(func, *args, seconds=10):
    p = multiprocessing.Process(target=func, args=args)
    p.start()
    p.join(seconds)

    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError("Timeout")
    else:
        return p.exitcode

def generate_random_sudoku_goal_prioritized():
    base = 3
    side = base * base
    digits = random.sample(range(1, side + 1), side)
    goal_state = [[(digits[(base * (r % base) + r // base + c) % side]) for c in range(side)] for r in range(side)]
    return goal_state

def generate_random_sudoku():
    goal_state = generate_random_sudoku_goal_prioritized()
    rows = [g[:] for g in goal_state]
    cols = [list(row) for row in zip(*goal_state)]
    random.shuffle(rows)
    random.shuffle(cols)

    puzzle = [list(row) for row in zip(*cols)]

    while not is_valid_assignment_sudoku(puzzle):
        random.shuffle(rows)
        random.shuffle(cols)
        puzzle = [list(row) for row in zip(*cols)]

    return puzzle

def is_valid_assignment_sudoku(puzzle):
    size = len(puzzle)
    subsquare_size = int(size**0.5)

    def is_valid_block(block):
        return set(block) == set(range(1, size + 1))

    for i in range(size):
        row = puzzle[i]
        col = [puzzle[j][i] for j in range(size)]
        if not (is_valid_block(row) and is_valid_block(col)):
            return False

    for i in range(0, size, subsquare_size):
        for j in range(0, size, subsquare_size):
            block = [puzzle[x][y] for x in range(i, i + subsquare_size) for y in range(j, j + subsquare_size)]
            if not is_valid_block(block):
                return False

    return True

def find_zero_position_sudoku(puzzle):
    for i in range(len(puzzle)):
        for j in range(len(puzzle[0])):
            if puzzle[i][j] == 0:
                return i, j
    return -1, -1

def backtracking_search_util_sudoku(puzzle, goal_state, explored_states):
    if puzzle == goal_state:
        return puzzle

    zero_position = find_zero_position_sudoku(puzzle)

    for value in set(range(1, len(puzzle) + 1)) - set(puzzle[zero_position[0]]):
        new_puzzle = [row[:] for row in puzzle]
        new_puzzle[zero_position[0]][zero_position[1]] = value

        if tuple(map(tuple, new_puzzle)) not in explored_states:
            explored_states.add(tuple(map(tuple, new_puzzle)))
            result = backtracking_search_util_sudoku(new_puzzle, goal_state, explored_states)

            if result is not None:
                return result

    return None

def backtracking_search_sudoku(initial_state, goal_state):
    if not is_valid_assignment_sudoku(initial_state):
        raise ValueError("Invalid initial state")

    result = None
    max_attempts = 1000
    success_count = 0
    failure_count = 0

    while max_attempts > 0:
        result = backtracking_search_util_sudoku(initial_state, goal_state, set())
        if result is not None:
            success_count += 1
            return result
        else:
            failure_count += 1
            print(f"No progress made, generating a new initial state. Attempts left: {max_attempts - 1}")
            initial_state = generate_random_sudoku()
            max_attempts -= 1

    print("Backtracking Search (Sudoku): No solution found after multiple attempts.")
    total_attempts = success_count + failure_count
    if total_attempts > 0:
        success_rate = success_count / total_attempts * 100
        failure_rate = failure_count / total_attempts * 100
        print(f"Success Rate: {success_rate:.2f}%")
        print(f"Failure Rate: {failure_rate:.2f}%")
    else:
        print("No attempts made.")

def forward_checking_util_sudoku(puzzle, goal_state, remaining_values):
    if is_valid_assignment_sudoku(puzzle) and puzzle == goal_state:
        return puzzle

    zero_position = find_zero_position_sudoku(puzzle)

    for value in remaining_values[zero_position[0]][zero_position[1]]:
        new_puzzle = [row[:] for row in puzzle]
        new_puzzle[zero_position[0]][zero_position[1]] = value

        new_remaining_values = [[set(values) for values in row] for row in remaining_values]

        if update_remaining_values_sudoku(new_remaining_values, zero_position, value):
            result = forward_checking_util_sudoku(new_puzzle, goal_state, new_remaining_values)

            if result is not None:
                return result

    return None

def forward_checking_sudoku(initial_state, goal_state):
    if not is_valid_assignment_sudoku(initial_state):
        raise ValueError("Invalid initial state")

    remaining_values = [[set(range(1, len(initial_state) + 1)) if initial_state[i][j] == 0 else set() for j in range(len(initial_state[0]))] for i in range(len(initial_state))]

    result = forward_checking_util_sudoku(initial_state, goal_state, remaining_values)
    if result is not None:
        return result
    else:
        print("Forward Checking (Sudoku): No solution found.")
        return None

def arc_consistency_ac3_sudoku(initial_state, goal_state):
    if not is_valid_assignment_sudoku(initial_state):
        raise ValueError("Invalid initial state")

    queue = []

    remaining_values = [[set(range(1, len(initial_state) + 1)) if initial_state[i][j] == 0 else set() for j in range(len(initial_state[0]))] for i in range(len(initial_state))]

    for i in range(len(initial_state)):
        for j in range(len(initial_state[0])):
            if initial_state[i][j] != 0:
                queue.append(((i, j), (i, j)))

    while queue:
        (i, j), (x, y) = queue.pop(0)

        if arc_reduce_sudoku(remaining_values, (i, j), (x, y)):
            if not remaining_values[i][j]:
                return None

            for k in range(len(initial_state)):
                for l in range(len(initial_state[0])):
                    if (k, l) != (i, j) and (k, l) != (x, y):
                        if k == i or l == j or (k // 3 == i // 3 and l // 3 == j // 3):
                            queue.append(((k, l), (i, j)))

    result = forward_checking_util_sudoku(initial_state, goal_state, remaining_values)
    if result is not None:
        return result
    else:
        print("Arc-Consistency (AC-3) (Sudoku): No solution found.")
        return None

def arc_reduce_sudoku(remaining_values, position1, position2):
    reduced = False
    values1 = remaining_values[position1[0]][position1[1]]
    values2 = remaining_values[position2[0]][position2[1]]

    for value in values1:
        if len(values2) == 1 and value in values2:
            remaining_values[position1[0]][position1[1]].remove(value)
            reduced = True

    return reduced

def update_remaining_values_sudoku(remaining_values, position, value):
    for i in range(len(remaining_values)):
        if i != position[0]:
            remaining_values[i][position[1]].discard(value)

    for j in range(len(remaining_values[0])):
        if j != position[1]:
            remaining_values[position[0]][j].discard(value)

    subsquare_size = int(len(remaining_values) ** 0.5)
    subsquare_row = position[0] // subsquare_size
    subsquare_col = position[1] // subsquare_size

    for i in range(subsquare_size * subsquare_row, subsquare_size * (subsquare_row + 1)):
        for j in range(subsquare_size * subsquare_col, subsquare_size * (subsquare_col + 1)):
            if (i, j) != position:
                remaining_values[i][j].discard(value)

    return True

def test_algorithm_sudoku(algorithm_func, instances, algorithm_name, num_runs=1):
    total_time = 0
    success_count = 0
    failure_count = 0

    for instance in instances * num_runs:
        try:
            partial_func = partial(algorithm_func, *instance)
            result = run_with_timeout(partial(partial_func), seconds=2)
            if result is not None:
                success_count += 1
            else:
                failure_count += 1
        except TimeoutError:
            failure_count += 1

    print(f"\n{algorithm_name} (Sudoku):")
    print(f"Success Rate: {success_count / (success_count + failure_count) * 100:.2f}%")
    print(f"Failure Rate: {failure_count / (success_count + failure_count) * 100:.2f}%")
    print(f"Total Time: {total_time:.5f} seconds\n")

# Number of instances to test

def run_tests():
    num_instances_sudoku = 5
    instances_sudoku = [(generate_random_sudoku(), generate_random_sudoku_goal_prioritized()) for _ in range(num_instances_sudoku)]

    # Test Backtracking Search for Sudoku
    test_algorithm_sudoku(backtracking_search_sudoku, instances_sudoku, "Backtracking Search", num_runs=5)

    # Test Forward Checking for Sudoku
    test_algorithm_sudoku(forward_checking_sudoku, instances_sudoku, "Forward Checking", num_runs=5)

    # Test Arc-Consistency (AC-3) for Sudoku
    test_algorithm_sudoku(arc_consistency_ac3_sudoku, instances_sudoku, "Arc-Consistency (AC-3)", num_runs=5)

if __name__ == '__main__':
    run_tests()