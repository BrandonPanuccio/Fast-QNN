import math

x = -9000
out = x * x  # Step 1: Square the input (ensures positive values)
out = out + 1e-6  # Step 2: Add a small constant for numerical stability
out = math.sqrt(out)  # Step 3: Take the square root to restore the magnitude
out = (x + out) / 2  # Step 4: Combine the original and adjusted values, then scale
print(out)
