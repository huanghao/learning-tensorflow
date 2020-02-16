import numpy as np

n = 5
x = np.array(range(1, n+1))

row = x.reshape([1, n])
col = x.reshape([n, 1])

mult = row * col
notdiag = row != col
eles = mult[notdiag]


def p(s, a):
    print(s, a.shape, '\n', a)

p('x', x)
p('mult', mult)
p('notdiag', notdiag)
p('mult[notdiag]', eles)

"""
x (5,)
 [1 2 3 4 5]
mult (5, 5)
 [[ 1  2  3  4  5]
 [ 2  4  6  8 10]
 [ 3  6  9 12 15]
 [ 4  8 12 16 20]
 [ 5 10 15 20 25]]
notdiag (5, 5)
 [[False  True  True  True  True]
 [ True False  True  True  True]
 [ True  True False  True  True]
 [ True  True  True False  True]
 [ True  True  True  True False]]
mult[notdiag] (20,)
 [ 2  3  4  5  2  6  8 10  3  6 12 15  4  8 12 20  5 10 15 20]
"""
