import pandas as pd
import numpy as np

import os

root_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(root_path, 'data')
labels_path = os.path.join(data_path, 'labels')

src = r'Index_summary.csv'
df = pd.read_csv(os.path.join(data_path, src), encoding='utf-8')

df.to_csv(os.path.join(data_path, 'Index_summary.csv'), index=False, encoding='utf-8-sig')
