from cynn import TinnNetwork, seed
import random
import time

# Seed random number generators
seed(int(time.time()))
random.seed(int(time.time()))

# XOR training data
xor_data = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
]

# Create network
net = TinnNetwork(2, 4, 1)

# Train with constant learning rate
rate = 0.5

for epoch in range(3000):
    random.shuffle(xor_data)
    total_error = 0.0

    for inputs, targets in xor_data:
        error = net.train(inputs, targets, rate)
        total_error += error

    avg_error = total_error / len(xor_data)

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: avg error = {avg_error:.6f}")

# Test predictions
for inputs, expected in xor_data:
    pred = net.predict(inputs)
    result = "✓" if abs(pred[0] - expected[0]) < 0.3 else "✗"
    print(f"{result} {inputs} -> {pred[0]:.4f} (expected {expected[0]})")
