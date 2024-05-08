import pandas as pd
from io import StringIO


# ## es
# data = """
# SEED,OUTDIST,l1_param_1,l1_param_2,l1_param_3,l1_param_4,l1_param_5,l5_param_1,l5_param_2,l5_param_3,l5_param_4,l5_param_5
# 0,original,0,0.3125,0.5208333,0.5208333,0.625,0.7291667,0.3125,0.2083333,0.3125,0.5208333
# 1,original,0,0.625,0.4166667,0.1041667,0.3125,0.4166667,0.625,0.625,0.4166667,0.5208333
# 2,original,0.849,0.777,0.825,0.801,0.777,0.953,0.881,0.953,1.058,1.033
# """

# df = pd.read_csv(StringIO(data))
# print(df)

# # seed 3개 평균
# mean_values = df.select_dtypes(include=[float]).mean()

# average_row = pd.DataFrame([mean_values], index=['Average'])  # 평균 행을 데이터프레임으로 만들기
# average_row.insert(0, 'OUTDIST', 'Average')
# average_row.insert(0, 'SEED', 'ALL') 

# # 맨 하단에 행 추가
# df = pd.concat([df, average_row], ignore_index=True)

# print("\nDataFrame with Averages:")
# print(df)


# ## imagenet
# data = """
# SEED,OUTDIST,gaussian_noise, shot_noise,impulse_noise, defocus_blur, glass_blur, motion_blur,  zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression
# 0,original,54.696,         53.4,       54.373,         55.765,         54.833,     43.948,         33.696,     38.347, 39.867, 26.098, 11.239,     54.301,     28.773,             23.042,     29.355
# 1,original,53.339,         51.674,     52.87,          54.469,         53.409,     42.802,         32.546,     36.809, 38.424, 24.847, 10.114,     53.646,     27.736,             22.353,     28.084
# 2,original,52.584,         51.298,     52.599,         53.934,         52.749,     42.116,         32.104,     36.564, 38.026, 24.26,  10.041,     52.742,     27.091,             21.646,     28.182
# """
data = """
SEED,OUTDIST,gaussian_noise, shot_noise,impulse_noise, defocus_blur, glass_blur, motion_blur,  zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression
0,original,75.04006,76.53245,66.69671,88.35136,67.26763,87.61018,88.82212,84.4351,83.63381,86.4383,91.46635,84.23478,78.17508,83.13301,77.07332
1,original,75.04006,76.53245,66.69671,88.35136,67.26763,87.61018,88.82212,84.4351,83.63381,86.4383,91.46635,84.23478,78.17508,83.13301,77.07332
2,original,73.16707,77.30369,67.57812,88.04087,68.39944,87.95072,88.02083,84.03446,82.27163,87.8105,90.3746,85.56691,78.35537,82.77244,75.3105
"""
df = pd.read_csv(StringIO(data))

# seed 3개 평균
mean_values = df.select_dtypes(include=[float]).mean()

average_row = pd.DataFrame([mean_values.values], columns=df.select_dtypes([float]).columns)  # 평균값
average_row['SEED'] = 'All' 
average_row['OUTDIST'] = 'Average'  

# 맨 하단에 행 추가
df = pd.concat([df, average_row], ignore_index=True)

print("DataFrame with Averages:")
print(df)
