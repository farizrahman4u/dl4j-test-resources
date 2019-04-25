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
