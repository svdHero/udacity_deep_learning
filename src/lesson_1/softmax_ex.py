import numpy as np
import matplotlib.pyplot as plt


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)  # broadcasting does the magic


def old_school_softmax(z):
    z = np.array(z)
    if z.ndim == 1:
        sm = np.exp(z) / np.sum(np.exp(z))
        return sm
    else:
        sm = np.zeros_like(z)
        num_cols = sm.shape[1]
        for col_index in range(num_cols):
            sm[:, col_index] = softmax(z[:, col_index])
        return sm


def test_softmax(factor=1.0):
    scores = [3.0, 1.0, 0.2]
    scores = [s*factor for s in scores]
    print(softmax(scores))

    # Plot softmax curves
    x = np.arange(-2.0, 6.0, 0.1)
    scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])*factor
    y = softmax(scores)

    # plt.figure()
    # plt.plot(x, y.T, linewidth=2)
    # plt.show()

    plt.figure()
    plt.title("Softmax values for factor {0}".format(factor))
    bottoms = np.cumsum(y, axis=0)
    plt.bar(x, y[0, :], label="y[{0!s}]".format(0))
    for i in range(1, y.shape[0]):
        plt.bar(x, y[i, :], bottom=bottoms[i-1], label="y[{0!s}]".format(i))
    plt.ylabel('Softmax')
    # plt.minorticks_on()
    # plt.grid(which='major')
    # plt.grid(which='minor')
    plt.legend()
    plt.show()


def main():
    test_softmax()
    test_softmax(10)
    test_softmax(1/10)


if __name__ == "__main__":
    main()
