# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.arange(5)
# years = ['0.1', '0.3', '0.5',"0.7","0.9"]
# values = [0.39, 0.43, 0.49,0.45,0.40]
#
# plt.bar(x, values)
# plt.xticks(x, years)
#
# plt.show()
import pandas as pd

raf = [0.39, 0.43, 0.49,0.45,0.40,0.84]
ck = [0.35, 0.33, 0.52,0.51,0.63,0.90]
fer = [0.0, 0.24, 0.49,0.47,0.43,0.72]
xlabel = ['0.1', '0.3', '0.5',"0.7","0.9","top-3"]

df = pd.DataFrame({'RAF-DB' : raf, 'CK+' : ck, 'FER2013' : fer}, index = xlabel)

import matplotlib.pyplot as plt
import numpy as np

# 그림 사이즈, 바 굵기 조정
fig, ax = plt.subplots(figsize=(12,8))
bar_width = 0.25

index = np.arange(6)

# 각 연도별로 3개 샵의 bar를 순서대로 나타내는 과정, 각 그래프는 0.25의 간격을 두고 그려짐
b1 = plt.bar(index, df['RAF-DB'], bar_width, alpha=0.4, color='red', label='RAF-DB')

b2 = plt.bar(index + bar_width, df['CK+'], bar_width, alpha=0.4, color='blue', label='CK+')

b3 = plt.bar(index + 2 * bar_width, df['FER2013'], bar_width, alpha=0.4, color='green', label='FER2013')

# x축 위치를 정 가운데로 조정하고 x축의 텍스트를 year 정보와 매칭
plt.xticks(np.arange(bar_width, 6 + bar_width, 1), xlabel,fontsize = 20)
plt.yticks(fontsize=20)

plt.xlabel('scale factor', size = 20)
plt.ylabel('accuracy', size = 20)
plt.legend(fontsize = 20)
plt.show()