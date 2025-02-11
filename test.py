from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

enhanced_Interaction_Customer_Transaction_Login = pd.read_csv("enhanced_customer_data.csv")
from ctgan import CTGAN
import pandas as pd
from tqdm import tqdm  # 引入进度条
import time

from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import networkx as nx
from collections import defaultdict
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# ==============================
# 1️⃣  加载原始数据
# ==============================
data = enhanced_Interaction_Customer_Transaction_Login.copy()

# ==============================
# 2️⃣  选择需要增强的数据（可选择 ChurnStatus = 1 的数据）
# ==============================
data_to_augment = data[data['ChurnStatus'] == 1]  # 只增强流失用户数据
num_samples = len(data_to_augment) * 5  # 生成五倍的流失数据

# ==============================
# 3️⃣  训练 CTGAN 模型（添加进度条）
# ==============================
epochs = 500  # 训练 300 轮

# 创建 CTGAN 模型
ctgan = CTGAN(epochs=epochs, batch_size=500, verbose=False)  

# 使用 tqdm 包装训练过程
print("Training CTGAN...")
for epoch in tqdm(range(epochs), desc="CTGAN Training Progress"):
    
    ctgan.fit(data_to_augment)  

print("CTGAN Training Complete!")

# ==============================
# 4️⃣  生成新的合成数据
# ==============================
print("🔄 Generating synthetic data...")
new_data = ctgan.sample(num_samples)

# 确保数据类型匹配
for col in data_to_augment.columns:
    if col in new_data.columns:
        new_data[col] = new_data[col].astype(data_to_augment[col].dtype)

print("✅ Synthetic Data Generated!")

# ==============================
# 5️⃣  合并合成数据和原始数据
# ==============================
augmented_data = pd.concat([data, new_data], ignore_index=True)

augmented_data.to_csv("augmented_data.csv", index=False)
