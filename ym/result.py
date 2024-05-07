import pandas as pd
from io import StringIO


## es
data = """
SEED,OUTDIST,l1_param_1,l1_param_2,l1_param_3,l1_param_4,l1_param_5,l5_param_1,l5_param_2,l5_param_3,l5_param_4,l5_param_5
0,original,0.754,0.754,0.5,0.754,0.538,0.575,0.65,0.642,0.612,0.575
1,original,0.553,0.553,0.603,0.603,0.654,0.603,0.654,0.553,0.654,0.654
2,original,0.849,0.777,0.825,0.801,0.777,0.953,0.881,0.953,1.058,1.033
"""

df = pd.read_csv(StringIO(data))
print(df)

# seed 3개 평균
mean_values = df.select_dtypes(include=[float]).mean()

average_row = pd.DataFrame([mean_values], index=['Average'])  # 평균 행을 데이터프레임으로 만들기
average_row.insert(0, 'OUTDIST', 'Average')
average_row.insert(0, 'SEED', 'ALL') 

# 맨 하단에 행 추가
df = pd.concat([df, average_row], ignore_index=True)

print("\nDataFrame with Averages:")
print(df)


# ## imagenet
# data = """
# SEED,OUTDIST,gaussian_noise, shot_noise,impulse_noise, defocus_blur, glass_blur, motion_blur,  zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression
# 0,original,54.696,         53.4,       54.373,         55.765,         54.833,     43.948,         33.696,     38.347, 39.867, 26.098, 11.239,     54.301,     28.773,             23.042,     29.355
# 1,original,53.339,         51.674,     52.87,          54.469,         53.409,     42.802,         32.546,     36.809, 38.424, 24.847, 10.114,     53.646,     27.736,             22.353,     28.084
# 2,original,52.584,         51.298,     52.599,         53.934,         52.749,     42.116,         32.104,     36.564, 38.026, 24.26,  10.041,     52.742,     27.091,             21.646,     28.182
# """

# df = pd.read_csv(StringIO(data))

# # seed 3개 평균
# mean_values = df.select_dtypes(include=[float]).mean()

# average_row = pd.DataFrame([mean_values.values], columns=df.select_dtypes([float]).columns)  # 평균값
# average_row['SEED'] = 'All' 
# average_row['OUTDIST'] = 'Average'  

# # 맨 하단에 행 추가
# df = pd.concat([df, average_row], ignore_index=True)

# print("DataFrame with Averages:")
# print(df)
