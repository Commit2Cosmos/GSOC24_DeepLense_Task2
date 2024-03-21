import matplotlib.pyplot as plt
import pandas as pd
import os


df = pd.read_csv(os.path.join("./data/easy_test.csv"))
df.set_index('ID', inplace=True)


plt.hist(df["numb_pix_lensed_image"], bins=50, density=True, alpha=0.75, color='b')
plt.show()