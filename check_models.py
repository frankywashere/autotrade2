import torch

for tf in ['15min', '1hour', '4hour', 'daily']:
    ckpt = torch.load(f'models/lnn_{tf}.pth')
    metadata = ckpt['metadata']
    print(f'{tf}:')
    print(f'  input_size: {metadata["input_size"]} (expected: 245)')
    print(f'  feature_names count: {len(metadata["feature_names"])}')
    print(f'  Has tsla_channel_1h_position: {"tsla_channel_1h_position" in metadata["feature_names"]}')
    print(f'  Has spy_channel_1h_position: {"spy_channel_1h_position" in metadata["feature_names"]}')
    print()
