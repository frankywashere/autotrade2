# Training on Google Compute Engine (GCE) - Autotrade

This guide helps you move the Autotrade Hierarchical LNN training to a powerful Google Cloud GPU instance.

## 1. Create the Instance
Go to [GCP Console](https://console.cloud.google.com/) > **Compute Engine** > **VM Instances**.

**Recommended Specs:**
*   **Instance Name:** `autotrade-gpu-server`
*   **Region:** `us-central1`
*   **Machine Type:** `n1-highmem-16` (104GB RAM to handle the 90GB dataset)
*   **GPU:** Nvidia T4 (Cheap/Free tier compatible)
*   **Boot Disk:** 
    *   **Image:** `Deep Learning VM with CUDA 12.1 M124` (Debian 11)
    *   **Size:** 100GB+ SSD (Crucial for `mmap` speed)

## 2. Upload Your Project
From your local machine (inside `/Users/frank/Desktop/CodingProjects/exp`):
```bash
gcloud compute scp --recurse $(ls -d * | grep -v 'data\|myenv\|.git') autotrade-gpu-server:~/autotrade --zone us-central1-a
```

## 3. Run Setup
On the GCE instance:
```bash
cd ~/autotrade
chmod +x gce_setup/setup.sh
./gce_setup/setup.sh
```

## 4. Transfer Training Data
Your `data/` folder is very large (~90GB). 
1.  Upload `data/` to a Google Cloud Storage Bucket.
2.  On the VM: `gsutil cp -r gs://your-bucket-name/data .`

## 5. Start Training
```bash
screen -S training
source training_env/bin/activate
python3 train_hierarchical.py
```