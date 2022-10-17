def parity(x):
    k = 0
    d = x
    while d != 0:
        k = k + 1
        d = d & (d - 1)
    return k % 2
