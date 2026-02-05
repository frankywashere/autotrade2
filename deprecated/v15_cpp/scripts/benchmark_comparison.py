#!/usr/bin/env python3
"""
Benchmark Comparison Visualization
Generates performance comparison charts for C++ vs Python scanner
"""

import matplotlib.pyplot as plt
import numpy as np

# Performance data
implementations = ['Python\n(1 worker)', 'C++\n(1 worker)', 'C++\n(8 workers)']
total_times = [7723.9, 25.6, 6.7]  # seconds
samples_per_sec = [0.13, 39.12, 148.30]
speedups = [1.0, 301.7, 1152.8]

# Phase breakdown data
phases = ['Pass 1\n(channels)', 'Pass 2\n(labels)', 'Pass 3\n(samples)']
python_times = [70.7, 153.2, 7500.0]
cpp_1w_times = [1.2, 5.7, 17.1]
cpp_8w_times = [1.0, 0.5, 5.0]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('V15 Scanner: C++ vs Python Performance Benchmark', fontsize=16, fontweight='bold')

# 1. Total execution time comparison (log scale)
ax1 = axes[0, 0]
colors = ['#e74c3c', '#3498db', '#2ecc71']
bars1 = ax1.bar(implementations, total_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Total Time (seconds, log scale)', fontsize=11, fontweight='bold')
ax1.set_title('Total Execution Time\n(1000 samples)', fontsize=12, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (bar, time) in enumerate(zip(bars1, total_times)):
    height = bar.get_height()
    if time > 100:
        label = f'{time:.0f}s'
    else:
        label = f'{time:.1f}s'
    ax1.text(bar.get_x() + bar.get_width()/2., height*1.2,
             label, ha='center', va='bottom', fontsize=10, fontweight='bold')

# 2. Speedup comparison
ax2 = axes[0, 1]
bars2 = ax2.bar(implementations, speedups, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Speedup (x times faster)', fontsize=11, fontweight='bold')
ax2.set_title('Speedup vs Python Baseline', fontsize=12, fontweight='bold')
ax2.axhline(y=10, color='red', linestyle='--', linewidth=2, label='10x Target')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.legend(fontsize=10)

# Add value labels on bars
for bar, speedup in zip(bars2, speedups):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height*1.05,
             f'{speedup:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3. Phase breakdown comparison (grouped bar chart)
ax3 = axes[1, 0]
x = np.arange(len(phases))
width = 0.25

bars3_1 = ax3.bar(x - width, python_times, width, label='Python (1w)',
                  color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)
bars3_2 = ax3.bar(x, cpp_1w_times, width, label='C++ (1w)',
                  color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
bars3_3 = ax3.bar(x + width, cpp_8w_times, width, label='C++ (8w)',
                  color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1)

ax3.set_ylabel('Time (seconds, log scale)', fontsize=11, fontweight='bold')
ax3.set_title('Phase-by-Phase Breakdown', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(phases)
ax3.set_yscale('log')
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# 4. Throughput comparison
ax4 = axes[1, 1]
bars4 = ax4.bar(implementations, samples_per_sec, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Samples per Second', fontsize=11, fontweight='bold')
ax4.set_title('Processing Throughput', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bar, sps in zip(bars4, samples_per_sec):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height*1.05,
             f'{sps:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add summary text box
summary_text = (
    "KEY RESULTS:\n"
    "• Single-threaded: 301.7x faster\n"
    "• Multi-threaded: 1152.8x faster\n"
    "• Target (10x): EXCEEDED by 30-115x\n"
    "• Status: Production Ready"
)
fig.text(0.5, 0.01, summary_text, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=[0, 0.06, 1, 0.96])
plt.savefig('/Users/frank/Desktop/CodingProjects/x14/v15_cpp/benchmark_results.png',
            dpi=300, bbox_inches='tight')
print("Benchmark visualization saved to: benchmark_results.png")
plt.show()
