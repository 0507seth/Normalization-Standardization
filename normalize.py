from sklearn import preprocessing
import numpy as np

arr = np.random.randint(100,size=(15))
print('Original Array = ', arr)

arr_norm = preprocessing.normalize([arr])
print('Normalized Array = ', arr_norm )

"""
## Output
arr =  [71  3 80 55 27 66 13 41 81 20 46 30  1 81 70]
Normalized Array = [0.34299517, 0.01449275, 0.38647343, 0.26570048, 0.13043478,
                      0.31884058, 0.06280193, 0.19806763, 0.39130435, 0.09661836,
                      0.22222222, 0.14492754, 0.00483092, 0.39130435, 0.33816425]]
 """
