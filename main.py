import numpy as np
import matplotlib.pyplot as plt


def main():
    ex1()


def ex1():
    mat_a = np.asmatrix([[2, 1, 2],
                         [1, -2, 1],
                         [1, 2, 3],
                         [1, 1, 1]])
    vec_b = np.asarray([[6],
                        [1],
                        [5],
                        [2]])
    ex1a(mat_a, vec_b)
    ex1c(mat_a, vec_b)


def ex1a(matrix_a: np.matrix, vector_b: np.asarray):
    print("regular least squares - ex1.a")
    least_squares(matrix_a, vector_b)


def least_squares(matrix_a: np.matrix, vector_b: np.asarray):
    try:
        ans_vector = np.linalg.inv(matrix_a.T @ matrix_a) @ matrix_a.T @ vector_b
    except:
        print("(A.T @ A) is a Singular matrix therefore not invertible")

    print("x = \n {}".format(ans_vector))
    print("r = Ax - b = \n {}".format(((matrix_a @ ans_vector) - vector_b)))


def ex1c(matrix_a: np.matrix, vector_b: np.asarray):
    print("weighted least squares - ex1.c")
    weights_matrix = np.eye(4)
    weights_matrix[0][0] = 100
    weighted_least_squares(matrix_a, vector_b, weights_matrix)


def weighted_least_squares(matrix_a: np.matrix, vector_b: np.asarray, weights_matrix: np.matrix):
    try:
        ans_vector = np.linalg.inv(matrix_a.T @ weights_matrix @ matrix_a) @ matrix_a.T @ weights_matrix @ vector_b
    except:
        print("(A.T @ W @ A) is a Singular matrix therefore not invertible")

    print("x = \n {}".format(ans_vector))
    print("r = Ax - b = \n {}".format(((matrix_a @ ans_vector) - vector_b)))


if __name__ == '__main__':
    main()
