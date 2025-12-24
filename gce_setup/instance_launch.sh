#!/bin/bash

# instance_launch.sh - Helper to launch the instance via CLI
# Customize these variables:
INSTANCE_NAME="training-gpu-1"
ZONE="us-central1-a"
MACHINE_TYPE="n1-highmem-16"
GPU_TYPE="nvidia-tesla-t4"

echo "Launching $INSTANCE_NAME in $ZONE..."

gcloud compute instances create $INSTANCE_NAME \
    --project=$(gcloud config get-value project) \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator=count=1,type=$GPU_TYPE \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --metadata="install-nvidia-driver=true"

echo "Once created, use: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"

