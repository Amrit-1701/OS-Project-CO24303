import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Create Dataset
# -------------------------------
data = pd.DataFrame({
    'cpu_usage': [10, 20, 30, 40, 50, 60, 70, 80, 90],
    'workload': ['low', 'low', 'low', 'medium', 'medium', 'medium', 'high', 'high', 'high']
})

# -------------------------------
# Step 2: Train ML Model
# -------------------------------
X = data[['cpu_usage']]
y = data['workload']

model = DecisionTreeClassifier()
model.fit(X, y)

# -------------------------------
# Step 3: DVFS Function
# -------------------------------
def dvfs_control(cpu):
    pred = model.predict([[cpu]])[0]
    
    if pred == 'low':
        freq = 1.2
    elif pred == 'medium':
        freq = 2.0
    else:
        freq = 3.0
    
    return pred, freq

# -------------------------------
# Step 4: Simulation
# -------------------------------
cpu_values = []
freq_values = []

print("\n--- DVFS Simulation ---\n")

for i in range(10):
    cpu = random.randint(10, 95)
    workload, freq = dvfs_control(cpu)
    
    cpu_values.append(cpu)
    freq_values.append(freq)
    
    print(f"CPU Usage: {cpu}% → Workload: {workload} → Frequency: {freq} GHz")

# -------------------------------
# Step 5: Graph (IMPORTANT)
# -------------------------------
plt.plot(cpu_values, freq_values, marker='o')
plt.xlabel("CPU Usage (%)")
plt.ylabel("CPU Frequency (GHz)")
plt.title("DVFS: CPU Usage vs Frequency")
plt.grid()

plt.show()