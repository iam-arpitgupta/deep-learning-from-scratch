
#Broadcasting role 
matrix = [[1,2,3,5],
           [4,7,9,3],
           [9,7,6,5]]


# sum 
np.sum(matrix , axis = 0) #all rows , dim =3 
np.sum(matrix , axis = 1) #all cols , dim =3 


# with dim 
np.sum(matrix , axis = 0 , keep_dim = True) #all rows , dim = 1 X 3 2D array
np.sum(matrix , axis = 1 , keep_dim = True) #all col , dim =  3  X 1 2D array



# broadcasting rules 
import numpy as np 
A = [[1, 2, 3], [4, 5, 6], [7, 8,9]]
print(np.sum(A))

print(np.sum(A, axis = 0))
print(np.sum(A, axis = 0).shape)

print(np.sum(A, axis = 1))
print(np.sum(A, axis = 1).shape)

print(np.sum(A, axis = 0,keepdims = True))
print(np.sum(A, axis = 0,keepdims = True).shape)

print(np.sum(A, axis = 1,keepdims = True))
print(np.sum(A, axis = 1,keepdims = True).shape)

print(np.max(A, axis = 0))
print(np.max(A, axis = 1))




