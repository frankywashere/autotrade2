# AutoTrade2 v3.5 - Quick Start Guide


# Create virtual environment if you want.
myenv\Scripts\activate 



pip install -r requirements.txt
python train_hierarchical.py --interactive


Settings for average M1-M5 Mac
Device: MPS 
(num_workers=0)
Precision: FP32
Memory profiling - optional
Training Period: 2015-2025
Feature acceleration GPU: NO
Chunking: Yes
Native Timeframe gen: Yes Streaming
Continuation prediction mode: adaptive
adaptive_labels (horizon 20-40 bars) (default)
Model Capacity: 192
Dataset split configuration: 3-way split: 85% train, 10% validation, 5% test (recommended)
Final aggregation method: Physics-Only - Weighted average (interpretable)
Epochs: 100
Batch size: 8 (test up to 512)
Learning Rate: 0.001
Dataset split configuration: 3-way split: 85% train, 10% validation, 5% test (recommended) 
Fusion Head: no
