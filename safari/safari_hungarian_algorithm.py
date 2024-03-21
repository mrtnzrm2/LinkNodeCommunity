import numpy as np

def hungarian_algorithm(cost_matrix):
    cost_matrix = np.array(cost_matrix)
    num_rows, num_cols = cost_matrix.shape
    num_workers = num_rows
    num_jobs = num_cols

    if num_workers > num_jobs:
        cost_matrix = np.pad(cost_matrix, ((0, 0), (0, num_workers - num_jobs)), mode='constant')
    elif num_jobs > num_workers:
        cost_matrix = np.pad(cost_matrix, ((0, num_jobs - num_workers), (0, 0)), mode='constant')

    num_rows, num_cols = cost_matrix.shape

    marked_zeros = np.zeros((num_rows, num_cols))
    row_covered = np.zeros(num_rows, dtype=bool)
    col_covered = np.zeros(num_cols, dtype=bool)
    num_covered = 0

    while num_covered < num_workers:
        marked_zeros.fill(0)
        for i in range(num_rows):
            for j in range(num_cols):
                if cost_matrix[i, j] == 0 and not row_covered[i] and not col_covered[j]:
                    marked_zeros[i, j] = 1

        if np.sum(marked_zeros) == 0:
            min_val = np.min(cost_matrix[~row_covered, :][:, ~col_covered])
            cost_matrix[~row_covered, :][:, ~col_covered] -= min_val
            row_min = np.min(cost_matrix[~row_covered, :], axis=1)
            cost_matrix[~row_covered, :] -= row_min[:, np.newaxis]
            col_min = np.min(cost_matrix[:, ~col_covered], axis=0)
            cost_matrix[:, ~col_covered] -= col_min
        else:
            for i in range(num_rows):
                for j in range(num_cols):
                    if marked_zeros[i, j]:
                        if not row_covered[i] and not col_covered[j]:
                            marked_zeros[i, j] = 2
                            row_covered[i] = True
                            col_covered[j] = True
                            num_covered += 1
                            break

    assignment = []
    for i in range(num_workers):
        for j in range(num_jobs):
            if marked_zeros[i, j] == 2:
                assignment.append((i, j))

    return assignment

# Example usage
cost_matrix = [[3, 1, 3], [3, 2, 3], [2, 4, 5]]
assignment = hungarian_algorithm(cost_matrix)
print("Assignments:", assignment)
