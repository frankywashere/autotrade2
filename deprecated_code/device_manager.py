"""
Device management for PyTorch training across CPU, CUDA, and MPS
Supports M1-M5 Apple Silicon, Google Colab (CUDA), and CPU fallback
Comprehensive error reporting and automatic fallback handling
"""

import torch
import platform
import subprocess
import os
import sys
from typing import Optional, Tuple, Dict, Any
from packaging import version

class DeviceManager:
    """Manages device selection and allocation for training with detailed error reporting"""

    @staticmethod
    def detect_hardware() -> Dict[str, Any]:
        """Detect available hardware and capabilities"""
        info = {
            'platform': platform.system(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            'cpu_count': torch.get_num_threads(),
        }

        # Check PyTorch version for MPS support
        if info['mps_available']:
            if version.parse(torch.__version__.split('+')[0]) < version.parse('1.12.0'):
                info['mps_available'] = False
                info['mps_error'] = "PyTorch 1.12+ required for MPS (current: {})".format(torch.__version__)

        # CUDA details
        if info['cuda_available']:
            try:
                info['cuda_device_count'] = torch.cuda.device_count()
                info['cuda_device_name'] = torch.cuda.get_device_name(0)
                info['cuda_capability'] = torch.cuda.get_device_capability(0)
                info['cuda_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            except Exception as e:
                info['cuda_error'] = str(e)
                info['cuda_available'] = False

        # Apple Silicon details
        if info['platform'] == 'Darwin':
            try:
                # Get macOS version
                mac_ver = platform.mac_ver()[0]
                info['macos_version'] = mac_ver

                # Check if macOS 13.0+ for full MPS support
                if mac_ver and info['mps_available']:
                    major, minor = map(int, mac_ver.split('.')[:2])
                    if major < 13:
                        info['mps_warning'] = f"macOS 13.0+ recommended for MPS (current: {mac_ver})"

                # Detect Apple chip type
                chip_info = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
                info['apple_chip'] = chip_info

                # Check if it's Apple Silicon
                if 'Apple' in chip_info:
                    info['is_apple_silicon'] = True
                    # Parse chip generation (M1, M2, M3, M4, M5)
                    if 'M1' in chip_info:
                        info['chip_generation'] = 'M1'
                    elif 'M2' in chip_info:
                        info['chip_generation'] = 'M2'
                    elif 'M3' in chip_info:
                        info['chip_generation'] = 'M3'
                    elif 'M4' in chip_info:
                        info['chip_generation'] = 'M4'
                    elif 'M5' in chip_info:
                        info['chip_generation'] = 'M5'
                    else:
                        info['chip_generation'] = 'Unknown Apple Silicon'
                else:
                    info['is_apple_silicon'] = False
                    if info['mps_available']:
                        info['mps_available'] = False
                        info['mps_error'] = "MPS requires Apple Silicon Mac"

            except Exception as e:
                info['platform_error'] = str(e)

        return info

    @staticmethod
    def test_device_operations(device_name: str) -> Dict[str, Any]:
        """Test specific operations on a device to check compatibility"""
        results = {
            'device': device_name,
            'available': False,
            'tensor_creation': False,
            'forward_pass': False,
            'backward_pass': False,
            'memory_available': None,
            'errors': []
        }

        try:
            device = torch.device(device_name)

            # Test 1: Tensor creation
            try:
                test_tensor = torch.randn(100, 100).to(device)
                results['tensor_creation'] = True
                results['available'] = True
            except Exception as e:
                results['errors'].append(f"Tensor creation failed: {str(e)}")
                return results

            # Test 2: Basic operations
            try:
                result = test_tensor @ test_tensor.T
                result = torch.nn.functional.relu(result)
                results['forward_pass'] = True
            except Exception as e:
                results['errors'].append(f"Forward operations failed: {str(e)}")

            # Test 3: Backward pass
            try:
                test_tensor.requires_grad = True
                output = (test_tensor ** 2).sum()
                output.backward()
                results['backward_pass'] = True
            except Exception as e:
                results['errors'].append(f"Backward pass failed: {str(e)}")

            # Test 4: Memory check
            if device_name == 'cuda':
                results['memory_available'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            elif device_name == 'mps':
                # MPS uses unified memory, report system memory
                import psutil
                results['memory_available'] = psutil.virtual_memory().total / 1024**3
            else:
                import psutil
                results['memory_available'] = psutil.virtual_memory().available / 1024**3

        except Exception as e:
            results['errors'].append(f"Device initialization failed: {str(e)}")

        return results

    @staticmethod
    def get_available_devices() -> Dict[str, Dict[str, Any]]:
        """Get list of available devices with detailed status"""
        devices = {}

        # Always test CPU
        devices['cpu'] = DeviceManager.test_device_operations('cpu')

        # Test CUDA if potentially available
        if torch.cuda.is_available():
            devices['cuda'] = DeviceManager.test_device_operations('cuda')

        # Test MPS if potentially available
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices['mps'] = DeviceManager.test_device_operations('mps')

        return devices

    @staticmethod
    def select_device_interactive() -> torch.device:
        """Interactive device selection with hardware detection and error reporting"""
        hardware_info = DeviceManager.detect_hardware()
        devices = DeviceManager.get_available_devices()

        print("\n" + "=" * 70)
        print("🖥️  HARDWARE DETECTION")
        print("=" * 70)
        print(f"Platform: {hardware_info['platform']}")
        print(f"Processor: {hardware_info['processor']}")
        print(f"PyTorch: {hardware_info['pytorch_version']}")
        print(f"CPU Threads: {hardware_info['cpu_count']}")

        if hardware_info.get('is_apple_silicon'):
            print(f"Apple Silicon: {hardware_info.get('chip_generation', 'Unknown')}")
            if hardware_info.get('macos_version'):
                print(f"macOS: {hardware_info['macos_version']}")

        if hardware_info['cuda_available']:
            print(f"\n✅ CUDA Available")
            print(f"   GPU: {hardware_info.get('cuda_device_name', 'Unknown')}")
            print(f"   Memory: {hardware_info.get('cuda_memory_gb', 0):.1f} GB")
            if 'cuda_capability' in hardware_info:
                print(f"   Compute Capability: {hardware_info['cuda_capability']}")
        elif 'cuda_error' in hardware_info:
            print(f"\n❌ CUDA Not Available")
            print(f"   Reason: {hardware_info['cuda_error']}")
        else:
            print(f"\n❌ CUDA Not Available")
            print(f"   Reason: No NVIDIA GPU detected")

        if hardware_info['mps_available']:
            print(f"\n✅ Metal Performance Shaders (MPS) Available")
            print(f"   Chip: {hardware_info.get('apple_chip', 'Apple Silicon')}")
            print(f"   Unified Memory Architecture")
            if 'mps_warning' in hardware_info:
                print(f"   ⚠️  {hardware_info['mps_warning']}")
        elif hardware_info.get('is_apple_silicon'):
            print(f"\n❌ MPS Not Available")
            if 'mps_error' in hardware_info:
                print(f"   Reason: {hardware_info['mps_error']}")
            else:
                print(f"   Reason: Unknown error")

        print("\n" + "=" * 70)
        print("⚙️  DEVICE COMPATIBILITY TESTS")
        print("=" * 70)

        available_devices = []
        device_details = {}

        for device_name, test_results in devices.items():
            if test_results['available']:
                print(f"\n{device_name.upper()}:")
                status_items = []

                if test_results['tensor_creation']:
                    print(f"  ✓ Tensor operations: PASS")
                    status_items.append("✓")
                else:
                    print(f"  ✗ Tensor operations: FAIL")
                    status_items.append("✗")

                if test_results['forward_pass']:
                    print(f"  ✓ Forward pass: PASS")
                    status_items.append("✓")
                else:
                    print(f"  ✗ Forward pass: FAIL")
                    status_items.append("✗")

                if test_results['backward_pass']:
                    print(f"  ✓ Backward pass: PASS")
                    status_items.append("✓")
                else:
                    print(f"  ✗ Backward pass: FAIL")
                    status_items.append("✗")
                    if device_name == 'mps':
                        print(f"    ⚠️  Some operations may fallback to CPU")

                if test_results['memory_available']:
                    print(f"  Memory: {test_results['memory_available']:.1f} GB")

                if test_results['errors']:
                    print(f"  ⚠️  Issues detected:")
                    for error in test_results['errors']:
                        print(f"    - {error}")

                # Only add to available if basic operations work
                if test_results['tensor_creation'] and test_results['forward_pass']:
                    available_devices.append(device_name)
                    device_details[device_name] = test_results
                else:
                    print(f"  ❌ Device not usable due to failed operations")

        if not available_devices:
            print("\n❌ CRITICAL ERROR: No devices available!")
            print("Defaulting to CPU (may not work properly)")
            return torch.device('cpu')

        print("\n" + "=" * 70)
        print("📊 DEVICE SELECTION")
        print("=" * 70)
        print("\nAvailable devices:")

        for idx, device in enumerate(available_devices, 1):
            if device == 'cuda':
                desc = f"NVIDIA GPU ({hardware_info.get('cuda_device_name', 'Unknown')})"
                speed = "~4-6x faster than CPU"
            elif device == 'mps':
                chip = hardware_info.get('chip_generation', 'Apple Silicon')
                desc = f"Apple Metal GPU ({chip})"
                speed = "~3-5x faster than CPU"
            else:
                desc = "CPU Only"
                speed = "Baseline speed"

            # Show status indicators
            status = device_details[device]
            if status['backward_pass'] and status['forward_pass']:
                status_icon = "✅"
            elif status['forward_pass']:
                status_icon = "⚠️"
            else:
                status_icon = "❌"

            print(f"  {idx}. {device:<4} {status_icon} - {desc}")
            print(f"      → {speed}")

        # Auto-select best device
        if 'cuda' in available_devices and device_details['cuda']['backward_pass']:
            default_idx = available_devices.index('cuda') + 1
            default_device = 'cuda'
        elif 'mps' in available_devices and device_details['mps']['forward_pass']:
            default_idx = available_devices.index('mps') + 1
            default_device = 'mps'
        else:
            default_idx = 1
            default_device = 'cpu'

        print(f"\n💡 Recommended: {default_device} (option {default_idx})")

        while True:
            try:
                choice = input(f"\nSelect device [1-{len(available_devices)}] (Enter for recommended): ").strip()

                if choice == '':
                    selected_device = default_device
                    break

                idx = int(choice)
                if 1 <= idx <= len(available_devices):
                    selected_device = available_devices[idx - 1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(available_devices)}")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\n\nInterrupted. Using default device.")
                selected_device = default_device
                break

        print(f"\n✓ Selected device: {selected_device}")

        # Final compatibility check and warning
        final_device = torch.device(selected_device)
        if selected_device == 'mps':
            print("ℹ️  Note: MPS may fallback to CPU for unsupported operations")
            print("  Setting PYTORCH_ENABLE_MPS_FALLBACK=1")
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

        if not device_details[selected_device]['backward_pass']:
            print("⚠️  Warning: Backward pass may have issues on this device")
            print("  Training might be unstable or fail")

        print("=" * 70)

        return final_device

    @staticmethod
    def select_device_auto(verbose: bool = True) -> torch.device:
        """Auto-select best available device without interaction"""
        devices = DeviceManager.get_available_devices()

        # Priority: CUDA > MPS > CPU
        if 'cuda' in devices and devices['cuda']['available'] and devices['cuda']['backward_pass']:
            device = torch.device('cuda')
            if verbose:
                print(f"🖥️  Auto-selected device: cuda (NVIDIA GPU)")
        elif 'mps' in devices and devices['mps']['available'] and devices['mps']['forward_pass']:
            device = torch.device('mps')
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            if verbose:
                print(f"🖥️  Auto-selected device: mps (Apple Metal)")
                if not devices['mps']['backward_pass']:
                    print("  ⚠️  Some operations may fallback to CPU")
        else:
            device = torch.device('cpu')
            if verbose:
                print(f"🖥️  Auto-selected device: cpu")
                if 'cuda' in devices and not devices['cuda']['available']:
                    print(f"  ℹ️  CUDA unavailable: {', '.join(devices['cuda']['errors'])}")
                if 'mps' in devices and not devices['mps']['available']:
                    print(f"  ℹ️  MPS unavailable: {', '.join(devices['mps']['errors'])}")

        return device

    @staticmethod
    def move_to_device(obj, device: torch.device, verbose: bool = False):
        """Safely move object to device with error handling and reporting"""
        try:
            return obj.to(device)
        except Exception as e:
            error_msg = str(e)

            if verbose:
                print(f"\n⚠️  Failed to move to {device}")
                print(f"  Error: {error_msg}")

                # Provide specific guidance based on error
                if "out of memory" in error_msg.lower():
                    print("  → Try reducing batch size or sequence length")
                elif "mps" in error_msg.lower():
                    print("  → MPS operation not supported, falling back to CPU")
                elif "cuda" in error_msg.lower():
                    print("  → CUDA error detected, check GPU availability")
                else:
                    print("  → Falling back to CPU")

            # Try CPU fallback
            try:
                if verbose:
                    print("  ✓ Successfully moved to CPU")
                return obj.to('cpu')
            except Exception as cpu_error:
                if verbose:
                    print(f"  ❌ CPU fallback also failed: {cpu_error}")
                raise

    @staticmethod
    def setup_mps_environment():
        """Configure environment for MPS compatibility"""
        # Enable fallback for unsupported operations
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

        # Optionally limit memory growth (commented out by default)
        # os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.8'

        print("✓ MPS environment configured")
        print("  - Fallback to CPU enabled for unsupported ops")

    @staticmethod
    def get_device_memory_info(device: torch.device) -> Dict[str, float]:
        """Get memory information for the device"""
        info = {}

        if device.type == 'cuda':
            info['total_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            info['allocated_gb'] = torch.cuda.memory_allocated(0) / 1024**3
            info['cached_gb'] = torch.cuda.memory_reserved(0) / 1024**3
            info['free_gb'] = info['total_gb'] - info['allocated_gb']
        elif device.type == 'mps':
            # MPS uses unified memory
            import psutil
            mem = psutil.virtual_memory()
            info['total_gb'] = mem.total / 1024**3
            info['available_gb'] = mem.available / 1024**3
            info['used_gb'] = mem.used / 1024**3
            info['free_gb'] = mem.available / 1024**3
        else:  # CPU
            import psutil
            mem = psutil.virtual_memory()
            info['total_gb'] = mem.total / 1024**3
            info['available_gb'] = mem.available / 1024**3
            info['used_gb'] = mem.used / 1024**3
            info['free_gb'] = mem.available / 1024**3

        return info

    @staticmethod
    def print_device_summary(device: torch.device):
        """Print a summary of the selected device"""
        print(f"\n{'='*70}")
        print(f"🎯 DEVICE SUMMARY: {device}")
        print(f"{'='*70}")

        mem_info = DeviceManager.get_device_memory_info(device)

        if device.type == 'cuda':
            print(f"Type: NVIDIA GPU")
            print(f"Name: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {mem_info['total_gb']:.1f} GB total, {mem_info['free_gb']:.1f} GB free")
        elif device.type == 'mps':
            print(f"Type: Apple Metal GPU")
            hardware_info = DeviceManager.detect_hardware()
            if 'apple_chip' in hardware_info:
                print(f"Chip: {hardware_info['apple_chip']}")
            print(f"Unified Memory: {mem_info['total_gb']:.1f} GB total, {mem_info['free_gb']:.1f} GB available")
            print(f"Note: MPS may fallback to CPU for some operations")
        else:
            print(f"Type: CPU")
            print(f"Threads: {torch.get_num_threads()}")
            print(f"Memory: {mem_info['total_gb']:.1f} GB total, {mem_info['free_gb']:.1f} GB available")

        print(f"{'='*70}\n")

    @staticmethod
    def get_system_ram_gb() -> float:
        """Get total system RAM in GB"""
        import psutil
        return psutil.virtual_memory().total / 1024**3

    @staticmethod
    def get_available_ram_gb() -> float:
        """Get available system RAM in GB"""
        import psutil
        return psutil.virtual_memory().available / 1024**3