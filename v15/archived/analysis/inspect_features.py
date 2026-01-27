import pickle

with open('/Users/frank/Desktop/CodingProjects/x14/samples_small.pkl', 'rb') as f:
    samples = pickle.load(f)

sample = samples[0] if isinstance(samples, list) else samples['samples'][0]
features = sample.tf_features

tf_5min = set(k.replace('5min_', '') for k in features.keys() if k.startswith('5min_'))
tf_daily = set(k.replace('daily_', '') for k in features.keys() if k.startswith('daily_'))

print("Features in 5min but NOT in daily:")
for f in sorted(tf_5min - tf_daily):
    print(f"  {f}")

print("\nFeatures in daily but NOT in 5min:")
for f in sorted(tf_daily - tf_5min):
    print(f"  {f}")
