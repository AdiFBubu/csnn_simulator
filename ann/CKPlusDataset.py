import os
import csv

import numpy as np
import torch
import random
import cv2
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


class CKPlusDataset(Dataset):
    def __init__(self, csv_path, images_dir, fold_to_use, is_train=True,
                 num_folds=10, random_seed=42, img_width=48, img_height=48):
        """
        PyTorch equivalent of dataset::CK_Plus
        """
        self.images_dir = images_dir
        self.img_width = img_width
        self.img_height = img_height

        # 1. Read CSV
        all_sequences = []
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3: continue
                subject, ipostase, emotion = row[0].strip(), int(row[1].strip()), int(row[2].strip())
                all_sequences.append({
                    'subject': subject,
                    'ipostase': ipostase,
                    'emotion': emotion
                })

        # 2. Replicate C++ Distribution Logic EXACTLY
        sequences_by_emotion = defaultdict(list)
        for seq in all_sequences:
            sequences_by_emotion[seq['emotion']].append(seq)

        rng = random.Random(random_seed)
        folds_data = {f: [] for f in range(1, num_folds + 1)}

        for emotion, emotion_sequences in sequences_by_emotion.items():
            # Standard python random.shuffle behaves slightly differently than std::shuffle,
            # but setting the seed ensures consistency across python runs.
            rng.shuffle(emotion_sequences)

            for i, seq in enumerate(emotion_sequences):
                fold = (i % num_folds) + 1  # 1-indexed folds just like C++
                folds_data[fold].append(seq)

        # 3. Filter for Train/Test
        self.data = []
        if is_train:
            for f in range(1, num_folds + 1):
                if f != fold_to_use:
                    self.data.extend(folds_data[f])
        else:
            self.data.extend(folds_data[fold_to_use])

        # În interiorul __init__, după ce încarci self.data din CSV
        valid_data = []
        for seq in self.data:
            subject = seq['subject']
            ipostase_str = f"{seq['ipostase']:03d}"
            seq_dir = os.path.join(self.images_dir, subject, ipostase_str)

            if os.path.exists(seq_dir):
                valid_data.append(seq)
            else:
                print(f"Atenție: Folder lipsă, ignorăm secvența: {seq_dir}")

        self.data = valid_data  # Păstrăm doar ce există pe disc

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        subject = seq['subject']
        ipostase_str = f"{seq['ipostase']:03d}"  # e.g., 001

        # C++ uses emotion 1-7, PyTorch loss functions require 0-indexed classes
        # Assuming your emotions are 1-7, we subtract 1. Adjust if your labels are different.
        label = seq['emotion'] - 1

        seq_dir = os.path.join(self.images_dir, subject, ipostase_str)

        # Load and sort image paths
        valid_extensions = ('.png', '.jpg', '.jpeg')
        img_names = [f for f in os.listdir(seq_dir) if f.lower().endswith(valid_extensions)]
        img_names.sort()

        frames = []
        for img_name in img_names:
            img_path = os.path.join(seq_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None: continue

            # Resize like in C++: cv::resize(image, image, cv::Size(m_image_width, m_image_height))
            if img.shape[0] != self.img_height or img.shape[1] != self.img_width:
                img = cv2.resize(img, (self.img_width, self.img_height))

            # Normalize to 0.0 - 1.0
            img = img.astype('float32') / 255.0
            frames.append(img)

        # Stack frames into a tensor
        # Shape becomes: (Depth/Frames, Height, Width)
        tensor_3d = torch.tensor(np.array(frames))

        # Add channel dimension so PyTorch Conv3d is happy: (Channels, Depth, Height, Width)
        tensor_3d = tensor_3d.unsqueeze(0)

        return tensor_3d, torch.tensor(label, dtype=torch.long)

MAX_FRAMES = 30 # Adjust this to the actual maximum frame count in your CK+ dataset

def pad_collate_fn(batch):
    """
    Since CK+ sequences have different lengths (e.g., some are 12 frames, some 25),
    we must pad the temporal/depth dimension so we can stack them into batches.
    """
    # batch is a list of tuples: [(tensor1, label1), (tensor2, label2), ...]
    tensors, labels = zip(*batch)

    padded_tensors = []
    for t in tensors:
        c, d, h, w = t.shape
        pad_size = MAX_FRAMES - d

        # PyTorch pad syntax starts from the last dimension and moves backwards.
        # Format: (PadLeft, PadRight, PadTop, PadBottom, PadFront, PadBack)
        # We only want to pad the 'Back' of the temporal dimension.
        if pad_size > 0:
            padded = torch.nn.functional.pad(t, (0, 0, 0, 0, 0, pad_size), "constant", 0)
        else:
            padded = t  # If you have sequences longer than MAX_FRAMES, you would slice them here: t[:, :MAX_FRAMES, :, :]
        padded_tensors.append(padded)

    return torch.stack(padded_tensors), torch.stack(labels)