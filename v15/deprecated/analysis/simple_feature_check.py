#!/usr/bin/env python3
"""Just check the features, no data loading."""

import pickle

# Load samples
print("Loading samples...")
with open('v15/cache/production_samples.pkl', 'rb') as f:
    samples = pickle.load(f)

sample = samples[109]
print(f"\nSample 109:")
print(f"  Timestamp: {sample.timestamp}")
print(f"  Best window: {sample.best_window}")

# Check SPY 5min w10 features
window = 10
tf = '5min'

slope_key = f"{tf}_w{window}_spy_channel_slope"
intercept_key = f"{tf}_w{window}_spy_channel_intercept"
std_dev_ratio_key = f"{tf}_w{window}_spy_std_dev_ratio"

print(f"\nSPY {tf} w{window} features:")

if slope_key in sample.tf_features:
    print(f"  {slope_key}: {sample.tf_features[slope_key]:.6f}")
else:
    print(f"  {slope_key}: NOT FOUND")

if intercept_key in sample.tf_features:
    print(f"  {intercept_key}: {sample.tf_features[intercept_key]:.6f}")
else:
    print(f"  {intercept_key}: NOT FOUND")

if std_dev_ratio_key in sample.tf_features:
    print(f"  {std_dev_ratio_key}: {sample.tf_features[std_dev_ratio_key]:.6f}")
else:
    print(f"  {std_dev_ratio_key}: NOT FOUND")

# List all SPY 5min w10 features
print(f"\nAll SPY {tf} w{window} features:")
count = 0
for key in sorted(sample.tf_features.keys()):
    if f"{tf}_w{window}_spy" in key:
        print(f"  {key}: {sample.tf_features[key]:.6f}")
        count += 1

if count == 0:
    print("  None found!")

print(f"\nTotal SPY {tf} w{window} features: {count}")
