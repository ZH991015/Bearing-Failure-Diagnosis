import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import re
from pathlib import Path
from collections import Counter


def natural_sort_key(path):
    """Sort part1, part2, ... part10 by the numeric part of the file name."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', str(path))]


class BearingFaultDataset(Dataset):
    """
    Sliding windows for bearing fault classification.

    Each sample is (5, window_size): four z-scored acceleration channels plus
    the raw rpm channel, which the model needs for the physics branch.
    Normalising rpm with the global statistics is left to the model.
    """

    def __init__(self, data_root, fault_types, window_size=2048, stride=512,
                 normalize=True, use_sensors='all', 
                 enable_rpm_filter=True, min_rpm=5000, rpm_filter_mode='mean'):
        """
        data_root: root folder holding the four class subfolders
        fault_types: class name -> label, e.g. {'无损伤': 0, '滚动体故障': 1, ...}
        window_size / stride: sliding window setup
        normalize: z-score the acceleration channels per file
        use_sensors: 'all', or a list of sensor indices
        enable_rpm_filter: drop windows whose rpm is below min_rpm
        rpm_filter_mode: 'mean' or 'min' over the window
        """
        self.data_root = data_root
        self.fault_types = fault_types
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.use_sensors = use_sensors
        self.enable_rpm_filter = enable_rpm_filter
        self.min_rpm = min_rpm
        self.rpm_filter_mode = rpm_filter_mode
        
        # filter bookkeeping
        self.filtered_windows_count = 0
        self.total_windows_before_filter = 0

        # global rpm stats, filled in on the first pass over the files
        self.global_rpm_mean = None
        self.global_rpm_std = None

        # class name -> folder on disk
        self.folder_names = {
            '无损伤': 'raw_data_combined_dynamic_无损伤_16kHz',
            '滚动体故障': 'raw_data_combined_滚动体故障_hdf5_16kHz',
            '内圈故障': 'raw_data_combined_内圈故障_hdf5_16kHz',
            '外圈故障': 'raw_data_combined_外圈故障_hdf5_16kHz'
        }

        # 1) global rpm statistics
        print("=" * 80)
        print("step 1: global rpm statistics ...")
        print("=" * 80)
        self._compute_global_rpm_stats()
        
        # 2) load all files
        print("\n" + "=" * 80)
        print("step 2: loading the dataset ...")
        print("=" * 80)
        if self.enable_rpm_filter:
            print(f"rpm filter on: dropping windows below {self.min_rpm} RPM (mode: {self.rpm_filter_mode})")
        
        self.samples = []
        self.labels = []
        self.metadata = []

        self._load_all_files()

        print(f"\ndataset ready")
        print(f"  windows: {len(self.samples)}")
        if self.enable_rpm_filter:
            print(f"  windows before filtering: {self.total_windows_before_filter}")
            print(f"  dropped: {self.filtered_windows_count}")
            print(f"  drop rate: {self.filtered_windows_count / self.total_windows_before_filter * 100:.2f}%")
        print(f"  per class: {self._count_labels()}")

    def _compute_global_rpm_stats(self):
        """
        First pass over all files, reading only the rpm column.
        Results are cached in rpm_stats_cache.json next to the data.
        """
        cache_path = os.path.join(self.data_root, 'rpm_stats_cache.json')
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                stats = json.load(f)
            self.global_rpm_mean = stats['mean']
            self.global_rpm_std = stats['std']
            print(f"global rpm stats from cache: mean={self.global_rpm_mean:.2f}, std={self.global_rpm_std:.2f}")
            return
        all_rpm_values = []
        
        for fault_name in sorted(self.fault_types.keys()):
            folder_name = self.folder_names.get(fault_name)
            
            if folder_name is None:
                print(f"warning: unknown class {fault_name}")
                continue
            
            folder_path = os.path.join(self.data_root, folder_name)
            if not os.path.exists(folder_path):
                print(f"warning: folder not found {folder_path}")
                continue
            
            h5_files = sorted(list(Path(folder_path).glob('*.h5')), key=natural_sort_key)
            print(f"  {fault_name}: {len(h5_files)} files")
            
            for h5_file in h5_files:
                try:
                    with h5py.File(str(h5_file), 'r') as f:
                        data = f['data']['table'][:]
                        values = data['values_block_0']
                        rpm_data = values[:, 5]  # column 6 is rpm
                        all_rpm_values.extend(rpm_data.tolist())
                
                except Exception as e:
                    print(f"    warning: could not read {h5_file.name}: {str(e)}")
                    continue
        
        all_rpm_values = np.array(all_rpm_values)
        self.global_rpm_mean = float(np.mean(all_rpm_values))
        self.global_rpm_std = float(np.std(all_rpm_values))
        
        print(f"\nglobal rpm stats:")
        print(f"  mean: {self.global_rpm_mean:.2f} RPM")
        print(f"  std:  {self.global_rpm_std:.2f} RPM")
        print(f"  range: [{all_rpm_values.min():.2f}, {all_rpm_values.max():.2f}] RPM")
        print(f"  samples: {len(all_rpm_values):,}")

    def _load_all_files(self):
        """Load every h5 file, sorted by class then by file name to keep the time order."""
        for fault_name in sorted(self.fault_types.keys()):
            label = self.fault_types[fault_name]
            folder_name = self.folder_names.get(fault_name)

            if folder_name is None:
                print(f"warning: unknown class {fault_name}")
                continue

            folder_path = os.path.join(self.data_root, folder_name)
            if not os.path.exists(folder_path):
                print(f"warning: folder not found {folder_path}")
                continue

            h5_files = sorted(list(Path(folder_path).glob('*.h5')), key=natural_sort_key)
            print(f"\nloading {fault_name} ({len(h5_files)} files) ...")

            for i, h5_file in enumerate(h5_files):
                if (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{len(h5_files)}")

                try:
                    windows = self._load_single_file(str(h5_file))

                    for window_idx, window in enumerate(windows):
                        self.samples.append(window)
                        self.labels.append(label)
                        self.metadata.append({
                            'file_name': h5_file.name,
                            'fault_type': fault_name,
                            'window_idx': window_idx,
                            'total_windows_in_file': len(windows)
                        })

                except Exception as e:
                    print(f"  warning: could not load {h5_file.name}: {str(e)}")
                    continue

    def _load_single_file(self, file_path):
        """
        Read one h5 file and cut it into windows.

        Returns windows of shape (window_size, 5): the first four columns are
        z-scored acceleration, the fifth is the raw rpm.
        """
        with h5py.File(file_path, 'r') as f:
            data = f['data']['table'][:]
            values = data['values_block_0']
            
            # keep the raw rpm for filtering
            original_rpm = values[:, 5:6]  # (N, 1)

            # acceleration and rpm columns
            if self.use_sensors == 'all':
                accel_data = values[:, 1:5]  # first four columns
                speed_data = values[:, 5:6]  # fifth column
            else:
                raise ValueError("only use_sensors='all' is supported")

            # normalisation
            if self.normalize:
                # z-score the acceleration channels
                accel_mean = np.mean(accel_data, axis=0, keepdims=True)
                accel_std = np.std(accel_data, axis=0, keepdims=True)
                accel_data = (accel_data - accel_mean) / (accel_std + 1e-8)

                # rpm stays raw, it is normalised with the global stats in the model
                pass

                # normalised acceleration + raw rpm
                sensor_data = np.concatenate([accel_data, speed_data], axis=1)
            else:
                sensor_data = np.concatenate([accel_data, speed_data], axis=1)

            # sliding windows
            windows = []
            num_samples = len(sensor_data)

            for start in range(0, num_samples - self.window_size + 1, self.stride):
                end = start + self.window_size
                window = sensor_data[start:end]
                
                self.total_windows_before_filter += 1
                
                if self.enable_rpm_filter:
                    # filter on the raw rpm
                    window_rpm_original = original_rpm[start:end]
                    
                    if self.rpm_filter_mode == 'mean':
                        rpm_value = np.mean(window_rpm_original)
                    elif self.rpm_filter_mode == 'min':
                        rpm_value = np.min(window_rpm_original)
                    else:
                        raise ValueError(f"invalid rpm_filter_mode: {self.rpm_filter_mode}")
                    
                    if rpm_value < self.min_rpm:
                        self.filtered_windows_count += 1
                        continue
                
                windows.append(window)

            return windows
            
    def _count_labels(self):
        """Window count per class name."""
        label_counts = Counter(self.labels)
        label_names = {v: k for k, v in self.fault_types.items()}
        result = {}
        for label, count in label_counts.items():
            name = label_names.get(label, f"unknown({label})")
            result[name] = count
        return result

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        x: (5, window_size), x[0:4] = z-scored acceleration, x[4] = raw rpm
        y: label
        """
        sample = self.samples[idx]  # (window_size, 5)
        label = self.labels[idx]

        # (window_size, 5) -> (5, window_size)
        x = torch.FloatTensor(sample).T
        y = torch.LongTensor([label])[0]

        return x, y


def create_dataloaders(data_root, batch_size=32, window_size=2048, stride=512,
                       train_ratio=0.85, val_ratio=0.1, random_seed=42,
                       split_mode='temporal', temporal_direction='backward',
                       enable_rpm_filter=True, min_rpm=5000, rpm_filter_mode='mean'):
    """Build the train / val / test loaders."""
    
    fault_types = {
        '无损伤': 0,
        '滚动体故障': 1,
        '内圈故障': 2,
        '外圈故障': 3
    }

    full_dataset = BearingFaultDataset(
        data_root=data_root,
        fault_types=fault_types,
        window_size=window_size,
        stride=stride,
        normalize=True,
        use_sensors='all',
        enable_rpm_filter=enable_rpm_filter,
        min_rpm=min_rpm,
        rpm_filter_mode=rpm_filter_mode
    )

    print(f"\ngrouping indices by class ...")
    class_indices = {label: [] for label in fault_types.values()}

    for idx, label in enumerate(full_dataset.labels):
        class_indices[label].append(idx)

    label_names = {v: k for k, v in fault_types.items()}
    print(f"\nindices per class:")
    for label, indices in class_indices.items():
        print(f"  {label_names[label]}: {len(indices)} windows")

    train_indices = []
    val_indices = []
    test_indices = []

    if split_mode == 'random':
        print(f"\nwarning: random split")
        np.random.seed(random_seed)
    elif split_mode == 'temporal':
        if temporal_direction == 'forward':
            print(f"\ntemporal split - forward")
        elif temporal_direction == 'backward':
            print(f"\ntemporal split - backward")
            print(f"   later fault data is used to predict earlier fault patterns")
        else:
            raise ValueError(f"unknown temporal_direction: {temporal_direction}")
    else:
        raise ValueError(f"unknown split_mode: {split_mode}")

    print(f"\nsplit per class:")
    for label in sorted(class_indices.keys()):
        indices = class_indices[label]

        if split_mode == 'random':
            indices_copy = indices.copy()
            np.random.shuffle(indices_copy)
            indices = indices_copy

        n = len(indices)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        test_size = n - train_size - val_size

        if split_mode == 'temporal' and temporal_direction == 'backward':
            class_test = indices[:test_size]
            class_val = indices[test_size:test_size + val_size]
            class_train = indices[test_size + val_size:]
        else:
            train_end = train_size
            val_end = train_size + val_size
            class_train = indices[:train_end]
            class_val = indices[train_end:val_end]
            class_test = indices[val_end:]

        train_indices.extend(class_train)
        val_indices.extend(class_val)
        test_indices.extend(class_test)

        print(f"  {label_names[label]}:")
        print(f"    train: {len(class_train)} ({len(class_train) / n * 100:.1f}%)")
        print(f"    val:   {len(class_val)} ({len(class_val) / n * 100:.1f}%)")
        print(f"    test:  {len(class_test)} ({len(class_test) / n * 100:.1f}%)")

    dataset_size = len(full_dataset)
    mode_desc = f"{split_mode}"
    if split_mode == 'temporal':
        mode_desc += f"-{temporal_direction}"
    print(f"\nsplit summary ({mode_desc} mode):")
    print(f"  train: {len(train_indices)} windows ({len(train_indices) / dataset_size * 100:.1f}%)")
    print(f"  val:   {len(val_indices)} windows ({len(val_indices) / dataset_size * 100:.1f}%)")
    print(f"  test:  {len(test_indices)} windows ({len(test_indices) / dataset_size * 100:.1f}%)")

    print(f"\nclass distribution check:")
    for split_name, split_indices in [('train', train_indices),
                                      ('val', val_indices),
                                      ('test', test_indices)]:
        split_labels = [full_dataset.labels[i] for i in split_indices]
        label_counts = Counter(split_labels)

        print(f"  {split_name}:")
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            percentage = count / len(split_indices) * 100
            print(f"    {label_names[label]}: {count} ({percentage:.1f}%)")

    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # also return the dataset so callers can read the global rpm stats
    num_classes = len(fault_types)
    return train_loader, val_loader, test_loader, num_classes, full_dataset