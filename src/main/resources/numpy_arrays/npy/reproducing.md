```
import numpy as np

for dt in [np.float32, np.float64, np.float16, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
    arr = np.arange(12, dtype=dt).reshape(3,4)
    filename = "arr_3,4_" + str(arr.dtype) + ".npy"
    path = "C:/Temp/npy/" + filename
    print(path)
    np.save(path, arr)
```


```
import numpy as np

for dt in [np.float32, np.float64, np.float16, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
    arr = np.zeros([0, 3], dtype=dt)
    filename = "arr_0,3_" + str(arr.dtype) + ".npy"
    path = "C:/Temp/npy_empty/" + filename
    print(path)
    np.save(path, arr)
```
