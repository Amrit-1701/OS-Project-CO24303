import pandas as pd
import random
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
# Step 4: Real-Time Graph Setup
# -------------------------------
cpu_values = []
freq_values = []
x_values = []

fig, ax = plt.subplots()

def update(frame):
    cpu = random.randint(10, 95)
    workload, freq = dvfs_control(cpu)

    x_values.append(frame)
    cpu_values.append(cpu)
    freq_values.append(freq)

    print(f"CPU Usage: {cpu}% -> Workload: {workload} -> Frequency: {freq} GHz")

    ax.clear()

    ax.plot(x_values, cpu_values, marker='o', label='CPU Usage (%)')
    ax.plot(x_values, freq_values, marker='s', label='Frequency (GHz)')

    ax.set_title("Real-Time DVFS Monitoring")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)

# -------------------------------
# Step 5: Animation
# -------------------------------
ani = FuncAnimation(fig, update, interval=1000)

plt.show()
