# %%
import pandas as pd
import numpy as np

# %%
data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print(data)

# %%
result = data.sum()
print(result)