import numpy as np
import time as tm


# arr1 = np.zeros((4,67200), dtype='bool')
arr1 = np.random.randint(0,2,(4,67200), dtype='bool')

t_0 = tm.time() # for timing

arr2 = np.copy(arr1)

arr2 = arr2.reshape((4,20,20,21,8))
arr2[1,:,:,15,:] = False
arr2 = arr2.reshape((4,67200))

arr1[1,::] = False

elapsed = tm.time() - t_0
print('Elapsed time: %f ms' % (elapsed*1000))

if np.all(arr1[:,:] == arr2[:,:]):
    print('Equal with reshape')
    


pass