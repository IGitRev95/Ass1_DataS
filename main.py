import numpy as np
import matplotlib.pyplot as plt


def main():
    ex1p1()


def ex1p1():
    mat_a = np.asmatrix([[2, 1, 2],
                         [1, -2, 1],
                         [1, 2, 3],
                         [1, 1, 1]])
    vec_b = np.asarray([[6],
                        [1],
                        [5],
                        [2]])
    least_squares(mat_a, vec_b)


def least_squares(matrix_a: np.matrix, vector_b: np.asarray):
    try:
        ans_vector = np.linalg.inv(matrix_a.T @ matrix_a) @ matrix_a.T @ vector_b
    except:
        print("(A.T @ A) is a Singular matrix therefore not invertible")

    print("x = \n {}".format(ans_vector))
    print("r = Ax - b = \n {}".format(((matrix_a @ ans_vector) - vector_b)))


if __name__ == '__main__':
    main()
