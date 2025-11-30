import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# --- 1. 數據生成與標註（模擬人類道德傾向數據集） ---

np.random.seed(42)
N = 10000  # 數據點數量

# P1: 潛在受害者數量 (1-10)
P1 = np.random.randint(1, 11, N)
# P2: 乘客生命權重 (0: 一般, 1: 高權重)
P2 = np.random.randint(0, 2, N)
# P3: 行為意圖 (0: 主動干預/轉向, 1: 被動不作為/直行)
P3 = np.random.randint(0, 2, N)
# P4: 法律風險 (0-10, 浮點數)
P4 = np.random.rand(N) * 10

X = np.stack([P1, P2, P3, P4], axis=1) # 特徵矩陣

# 道德傾向標註規則 (Y): 0: 功利主義, 1: 義務論
# 假設規則:
# 1. 受害者數量 P1 越多，越傾向功利主義 (Y=0)。
# 2. 乘客權重 P2 越高，越傾向義務論 (Y=1) (保護乘客)。
# 3. 主動干預 P3=0，越傾向義務論 (Y=1) (避免主動傷害/近因效應)。
# 4. 法律風險 P4 越高，越傾向義務論 (Y=1) (避免法律風險)。

# 結合規則生成傾向（模擬複雜的道德權衡）
# 傾向功利 (Y=0) 的基礎分數: (P1 * 0.4) - (P2 * 1.5) - (P3 * 0.8)
# 傾向義務 (Y=1) 的基礎分數: (P2 * 1.5) + (P3 * 0.8) + (P4 * 0.2)

# 功利分數 (Utility Score): P1 越大，分數越高 (傾向 Y=0)
score_util = (P1 * 0.4) - (P2 * 1.5) - (P3 * 0.8) + np.random.randn(N) * 1.5

# 義務分數 (Deontology Score): P2/P3/P4 越高，分數越高 (傾向 Y=1)
score_deon = (P2 * 1.5) + (P3 * 0.8) + (P4 * 0.2) + np.random.randn(N) * 1.5

# Y=0 (功利) 如果功利分數 > 義務分數
Y = (score_util > score_deon).astype(int)
# 這裡將標籤反轉，以符合報告中的定義：Y=0 (功利主義), Y=1 (義務論)
Y = 1 - Y

# --- 2. 數據預處理與劃分 ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)

# --- 3. 模型訓練 ---
model = LogisticRegression()
model.fit(X_train, Y_train)

# --- 4. 評估與結果分析 ---
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

print("--- 簡化模型問題：道德傾向預測模型 ---")
print(f"訓練數據集數量: {len(X_train)} 條")
print(f"測試數據集數量: {len(X_test)} 條")
print("-" * 40)
print(f"分類準確度 (Accuracy): {accuracy * 100:.2f}%")
print("-" * 40)

# 模型權重分析 (特徵重要性)
feature_names = ['P1: 潛在受害者數量', 'P2: 乘客生命權重', 'P3: 行為意圖(不作為)', 'P4: 法律風險']
print("模型權重分析 (Log Odds 係數):")
# 權重為正數表示該特徵增加 Y=1 (義務論) 的傾向
# 權重為負數表示該特徵增加 Y=0 (功利主義) 的傾向
for name, coef in zip(feature_names, model.coef_[0]):
    print(f"- {name}: {coef:.3f}")

# --- 5. 範例預測 ---
print("\n--- 範例預測 ---")

# 範例 1: 傾向功利主義
# P1=10 (大量受害者), P2=0 (一般乘客), P3=0 (主動干預), P4=1 (低風險)
example_utilitarian = scaler.transform(np.array([[10, 0, 0, 1]]))
pred_util = model.predict(example_utilitarian)[0]
prob_util = model.predict_proba(example_utilitarian)[0]
print(f"情境 1 (大量受害者): 預測傾向 = {'義務論' if pred_util == 1 else '功利主義'} (機率: {prob_util[0]*100:.1f}% 功利)")

# 範例 2: 傾向義務論
# P1=1 (少量受害者), P2=1 (高權重乘客), P3=1 (被動不作為), P4=9 (高風險)
example_deontological = scaler.transform(np.array([[1, 1, 1, 9]]))
pred_deon = model.predict(example_deontological)[0]
prob_deon = model.predict_proba(example_deontological)[0]
print(f"情境 2 (高權重乘客/被動): 預測傾向 = {'義務論' if pred_deon == 1 else '功利主義'} (機率: {prob_deon[1]*100:.1f}% 義務)")

# 範例 3: 模糊情境
# P1=5 (中等受害者), P2=0 (一般乘客), P3=0 (主動干預), P4=5 (中等風險)
example_mixed = scaler.transform(np.array([[5, 0, 0, 5]]))
pred_mixed = model.predict(example_mixed)[0]
prob_mixed = model.predict_proba(example_mixed)[0]
print(f"情境 3 (模糊/中等): 預測傾向 = {'義務論' if pred_mixed == 1 else '功利主義'} (機率: {np.max(prob_mixed)*100:.1f}% 最高機率)")

# 備註：在數據生成中，我們透過隨機性引入了複雜性，模擬人類決策的非確定性，
# 這使得準確度無法達到 100%，也更貼近現實中道德判斷的模糊性。