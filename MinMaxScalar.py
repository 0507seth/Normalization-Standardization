from sklearn import preprocessing
import pandas as pd
Insurance_Data = pd.read_csv('insurance2.csv')
scaler = preprocessing.MinMaxScaler(feature_range=(0, 2))
norm = scaler.fit_transform(Insurance_Data)
norm_df = pd.DataFrame(norm,columns=[Insurance_Data.columns])
print('Original Data \n',Insurance_Data.head(10))
print('Normalized Data by MinMaxScaler() \n',norm_df.head(10))

"""
##Output
Original Data 
    age  sex     bmi  children  smoker  region      charges  insuranceclaim
0   19    0  27.900         0       1       3  16884.92400               1
1   18    1  33.770         1       0       2   1725.55230               1
2   28    1  33.000         3       0       2   4449.46200               0
3   33    1  22.705         0       0       1  21984.47061               0
4   32    1  28.880         0       0       1   3866.85520               1
5   31    0  25.740         0       0       2   3756.62160               0
6   46    0  33.440         1       0       2   8240.58960               1
7   37    0  27.740         3       0       1   7281.50560               0
8   37    1  29.830         2       0       0   6406.41070               0
9   60    0  25.840         0       0       1  28923.13692               0
Normalized Data by MinMaxScaler()
         age  sex       bmi children smoker    region   charges insuranceclaim
0  0.043478  0.0  0.642454      0.0    2.0  2.000000  0.503222            2.0
1  0.000000  2.0  0.958300      0.4    0.0  1.333333  0.019272            2.0
2  0.434783  2.0  0.916868      1.2    0.0  1.333333  0.106230            0.0
3  0.652174  2.0  0.362927      0.0    0.0  0.666667  0.666020            0.0
4  0.608696  2.0  0.695184      0.0    0.0  0.666667  0.087631            2.0
5  0.565217  0.0  0.526231      0.0    0.0  1.333333  0.084112            0.0
6  1.217391  0.0  0.940543      0.4    0.0  1.333333  0.227259            2.0
7  0.826087  0.0  0.633844      1.2    0.0  0.666667  0.196641            0.0
8  0.826087  2.0  0.746301      0.8    0.0  0.000000  0.168704            0.0
9  1.826087  0.0  0.531612      0.0    0.0  0.666667  0.887531            0.0
"""
