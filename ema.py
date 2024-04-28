import matplotlib.pyplot as plt
import numpy as np

# Set parameters for the Brownian motion
N = 1000  # number of points
T = 1  # time duration
dt = T / N  # time step
beta = 0.99  # decay factor for EMA
k = 16  # interval for skipped-EMA

# Generate Brownian motion
increments = np.random.normal(0, np.sqrt(dt), N) + 0.001
positions = np.cumsum(increments)

# Calculate the EMA
ema1 = np.zeros(N)
ema1[0] = positions[0]
for i in range(1, N):
    beta = (1 - 1 / (i + 1)) ** (1 + 4)
    ema1[i] = beta * ema1[i - 1] + (1 - beta) * positions[i]

# Calculate skipped-EMA1
skipped_ema1 = np.zeros(N)
skipped_ema1[0] = positions[0]

for i in range(1, N):
    beta = (1 - 1 / (i + 1)) ** (1 + 4)
    betaK = 1 - (1 - beta) * k
    betaK = max(0, betaK)

    if i % k == 0:
        # print(betaK)
        skipped_ema1[i] = betaK * skipped_ema1[i - k] + (1 - betaK) * positions[i]
    else:
        skipped_ema1[i] = skipped_ema1[i - 1]


ema2 = np.zeros(N)
ema2[0] = positions[0]
for i in range(1, N):
    beta = (1 - 1 / (i + 1)) ** (1 + 16)
    ema2[i] = beta * ema2[i - 1] + (1 - beta) * positions[i]

# Calculate skipped-EMA2
skipped_ema2 = np.zeros(N)
skipped_ema2[0] = positions[0]

for i in range(1, N):
    beta = (1 - 1 / (i + 1)) ** (1 + 16)
    betaK = 1 - (1 - beta) * k
    betaK = max(0, betaK)

    if i % k == 0:
        # print(betaK)
        skipped_ema2[i] = betaK * skipped_ema2[i - k] + (1 - betaK) * positions[i]
    else:
        skipped_ema2[i] = skipped_ema2[i - 1]


# Plotting the results
time = np.linspace(0, T, N)
plt.figure(figsize=(10, 6))
plt.plot(time, positions, label="Brownian Motion")
plt.plot(time, ema1, label="Karras EMA1 with gamma = 4")
plt.plot(time, skipped_ema1, label=f"Skipped-Karras EMA (every {k} steps), gamma = 4")

plt.plot(time, ema2, label="Karras EMA2 with gamma = 16")
plt.plot(time, skipped_ema2, label=f"Skipped-Karras EMA (every {k} steps), gamma = 16")


plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Brownian Motion with EMA and Skipped-EMA")
plt.legend()
plt.grid(True)
plt.show()

# save
plt.savefig("ema.png")
