Code to reproduce:
```
import numpy as np

dir = "C:/DL4J/Git/dl4j-test-resources/src/main/resources/numpy_arrays/"
dtypes = [np.float16, np.float32, np.double, np.int8, np.int16, np.int32, np.int64, np.uint8]
for d in dtypes:
    array = np.arange(12, dtype=d).reshape(3,4)
    print(array)
    print(array.dtype)
    path = dir + "arange_3,4_" + str(array.dtype) + ".npy"
    print(path)
    np.save(path, array)
```


## Code to reproduce .npz arrays

```
import numpy as np

dir = "C:/DL4J/Git/dl4j-test-resources/src/main/resources/numpy_arrays/npz/"
dtypes = [np.float16, np.float32, np.double, np.int8, np.int16, np.int32, np.int64, np.uint8]
for d in dtypes:
    array = np.arange(12, dtype=d).reshape(3,4)
    array2 = np.linspace(start=0, stop=20, num=3, dtype=np.float32)
    print(array)
    print(array.dtype)
    path = dir + str(array.dtype) + ".npz"
    print(path)
    np.savez(path, firstArr=array, secondArr=array2)
```


## Code to reproduce txt format (tab delimited)

```
import numpy as np

dir = "C:/DL4J/Git/dl4j-test-resources/src/main/resources/numpy_arrays/txt/"
array = np.arange(12, dtype=np.float32).reshape(3,4)
#print(array)
#print(array.dtype)
path = dir + "arange_3,4_" + str(array.dtype) + ".txt"
print(path)
np.savetxt(path, array)

dir = "C:/DL4J/Git/dl4j-test-resources/src/main/resources/numpy_arrays/txt_tab/"
path = dir + "arange_3,4_" + str(array.dtype) + ".txt"
print(path)
np.savetxt(path, array, delimiter="\t")
```


## Code to reproduce scalars:

```
import numpy as np

dir = "C:/DL4J/Git/dl4j-test-resources/src/main/resources/numpy_arrays/scalar/"
dtypes = [np.float16, np.float32, np.double, np.int8, np.int16, np.int32, np.int64, np.uint8]
for d in dtypes:
    array = np.ones([], dtype=d) #.arange(12, dtype=d).reshape(3,4)
    print(array)
    print(array.dtype)
    print(array.shape)
    path = dir + "scalar_" + str(array.dtype) + ".npy"
    print(path)
    np.save(path, array)
```
