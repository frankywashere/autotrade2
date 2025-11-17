"""
GPU Monitoring Module for Real-Time Performance Tracking

Provides real-time GPU utilization, VRAM usage, temperature, and power metrics
to help diagnose training bottlenecks.

Usage:
    monitor = GPUMonitor(device)
    metrics = monitor.get_compact_metrics()  # For tqdm postfix
    detailed = monitor.format_detailed()      # For epoch summaries
"""

import torch
from typing import Dict, Optional


class GPUMonitor:
    """
    Real-time GPU monitoring for bottleneck detection.

    Collects GPU utilization %, VRAM usage, temperature, and power draw.
    Uses nvidia-ml-py if available, falls back to torch.cuda for basic metrics.
    """

    def __init__(self, device):
        """
        Initialize GPU monitor.

        Args:
            device: torch.device object
        """
        self.device = device
        self.pynvml_available = False
        self.nvml_handle = None

        # Only works for CUDA devices
        if device.type != 'cuda':
            return

        # Try to initialize NVIDIA Management Library
        try:
            import pynvml
            pynvml.nvmlInit()

            # Get handle for current device
            device_index = device.index if device.index is not None else 0
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.pynvml_available = True
            self.pynvml = pynvml

        except (ImportError, Exception):
            # Fall back to torch.cuda functions (limited metrics)
            self.pynvml_available = False

    def get_metrics(self) -> Optional[Dict]:
        """
        Get current GPU metrics.

        Returns:
            Dict with keys: 'gpu_util', 'mem_used_gb', 'mem_total_gb',
                           'mem_percent', 'temperature', 'power_watts'
            Returns None if not a CUDA device or metrics unavailable
        """
        if self.device.type != 'cuda':
            return None

        metrics = {}

        if self.pynvml_available and self.nvml_handle:
            try:
                # Get GPU utilization
                util = self.pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                metrics['gpu_util'] = util.gpu  # Percentage

                # Get memory info
                mem_info = self.pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                metrics['mem_used_gb'] = mem_info.used / (1024**3)
                metrics['mem_total_gb'] = mem_info.total / (1024**3)
                metrics['mem_percent'] = (mem_info.used / mem_info.total) * 100

                # Get temperature
                temp = self.pynvml.nvmlDeviceGetTemperature(
                    self.nvml_handle,
                    self.pynvml.NVML_TEMPERATURE_GPU
                )
                metrics['temperature'] = temp

                # Get power draw
                power = self.pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle)
                metrics['power_watts'] = power / 1000  # Convert mW to W

            except Exception as e:
                # If NVML fails, fall back to torch
                pass

        # Fallback to torch.cuda (only memory available)
        if not metrics:
            try:
                device_index = self.device.index if self.device.index is not None else 0
                mem_allocated = torch.cuda.memory_allocated(device_index) / (1024**3)
                mem_reserved = torch.cuda.memory_reserved(device_index) / (1024**3)
                mem_total = torch.cuda.get_device_properties(device_index).total_memory / (1024**3)

                metrics['mem_used_gb'] = mem_reserved  # Use reserved as proxy for used
                metrics['mem_total_gb'] = mem_total
                metrics['mem_percent'] = (mem_reserved / mem_total) * 100
                metrics['gpu_util'] = None  # Not available via torch
                metrics['temperature'] = None
                metrics['power_watts'] = None
            except Exception:
                return None

        return metrics

    def get_compact_metrics(self) -> Dict[str, str]:
        """
        Get metrics formatted for tqdm postfix (compact).

        Returns:
            Dict with 'gpu' and 'vram' keys for tqdm postfix
            Returns empty dict if metrics unavailable
        """
        metrics = self.get_metrics()
        if not metrics:
            return {}

        result = {}

        # GPU utilization
        if metrics.get('gpu_util') is not None:
            result['gpu'] = f"{metrics['gpu_util']:.0f}%"

        # VRAM usage
        if metrics.get('mem_used_gb') is not None and metrics.get('mem_total_gb') is not None:
            result['vram'] = f"{metrics['mem_used_gb']:.1f}/{metrics['mem_total_gb']:.0f}GB"

        return result

    def format_detailed(self) -> str:
        """
        Get metrics formatted for epoch summary (detailed).

        Returns:
            Formatted string with all available metrics
            Returns empty string if metrics unavailable
        """
        metrics = self.get_metrics()
        if not metrics:
            return ""

        parts = []

        # GPU utilization
        if metrics.get('gpu_util') is not None:
            parts.append(f"{metrics['gpu_util']:.0f}% util")

        # VRAM usage
        if metrics.get('mem_used_gb') is not None and metrics.get('mem_total_gb') is not None:
            parts.append(
                f"{metrics['mem_used_gb']:.1f}/{metrics['mem_total_gb']:.0f}GB VRAM "
                f"({metrics['mem_percent']:.0f}%)"
            )

        # Temperature
        if metrics.get('temperature') is not None:
            parts.append(f"{metrics['temperature']}°C")

        # Power
        if metrics.get('power_watts') is not None:
            parts.append(f"{metrics['power_watts']:.0f}W")

        if parts:
            return "GPU: " + ", ".join(parts)
        return ""

    def __del__(self):
        """Cleanup NVML on destruction."""
        if self.pynvml_available:
            try:
                self.pynvml.nvmlShutdown()
            except:
                pass
