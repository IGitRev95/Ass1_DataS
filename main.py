import numpy as np


def main():
    ex1()
    ex3()


def ex1():
    # Eq.(1)
    mat_a = np.asmatrix([[2, 1, 2],
                         [1, -2, 1],
                         [1, 2, 3],
                         [1, 1, 1]])
    vec_b = np.asarray([[6],
                        [1],
                        [5],
                        [2]])
    # ex1.a
    ex1a(mat_a, vec_b)
    print()
    # ex1.c
    ex1c(mat_a, vec_b)
    print()
    # ex1.d
    ex1d(mat_a, vec_b)
    print()


def ex1a(matrix_a: np.matrix, vector_b: np.asarray):
    print("regular least squares - Results - ex1.a")
    least_squares(matrix_a, vector_b)


def least_squares(matrix_a: np.matrix, vector_b: np.asarray):
    ans_vector = np.linalg.inv(matrix_a.T @ matrix_a) @ matrix_a.T @ vector_b
    print("x = \n {}".format(ans_vector))
    print("r = Ax - b = \n {}".format(((matrix_a @ ans_vector) - vector_b)))


def ex1c(matrix_a: np.matrix, vector_b: np.asarray):
    print("weighted least squares - Results - ex1.c")
    weights_matrix = np.eye(4)
    weight_0 = 1
    r_0 = 1
    r = 0
    while np.abs(r_0) > (10 ** (-3)):
        weights_matrix[0][0] = weight_0
        result_weighted_ls = weighted_least_squares(matrix_a, vector_b, weights_matrix)
        r = matrix_a @ result_weighted_ls - vector_b
        r_0 = r[0][0]
        weight_0 = weight_0 + 1

    print("x = \n{} \nr = Ax - b = \n{} \nr_0 = {} \nweight_0 = {}".format(result_weighted_ls, r, r_0, (weight_0 - 1)))


def weighted_least_squares(matrix_a: np.matrix, vector_b: np.asarray, weights_matrix: np.matrix):
    return np.linalg.inv(matrix_a.T @ weights_matrix @ matrix_a) @ matrix_a.T @ weights_matrix @ vector_b


def ex1d(matrix_a: np.matrix, vector_b: np.asarray):
    print("Tikhonov regularization least squares - Results - ex1.d")
    tikhonov_least_squares(matrix_a, vector_b, 0.1)


def tikhonov_least_squares(matrix_a: np.matrix, vector_b: np.asarray, lambda_val: float):
    matrix_lambda = np.eye(3) * lambda_val
    ans_vector = np.linalg.inv((matrix_a.T @ matrix_a) + matrix_lambda) @ matrix_a.T @ vector_b
    print("x = \n{}\nr = Ax - b =\n{}".format(ans_vector, ((matrix_a @ ans_vector) - vector_b)))


def ex3():
    print("Results - ex3.c - The D Matrix:")
    mat_a = np.asarray([[5, 6, 7, 8],
                        [1, 3, 5, 4],
                        [1, 0.5, 4, 2],
                        [3, 4, 3, 1]])
    mat_b = np.asarray([[0.57, 0.56, 0.8, 1],
                        [1.5, 4, 6.7, 4.9],
                        [0.2, 0.1, 1, 0.6],
                        [11, 30, 26, 10]])

    diagonal_ally = []
    for i in range(len(mat_a)):
        diagonal_ally.append(__d_of_row__(mat_a, mat_b, i))
    mat_d = np.diag(diagonal_ally)
    print(mat_d)


def __d_of_row__(a, b, i):
    return ((1 / (a[i].T @ a[i])) * a[i].T) @ b[i]  # implementation of the equation: D_i=(a_i.T@a_i)^(-1)@a_i.T@b_i


if __name__ == '__main__':
    main()
