import torch
import torch.nn as nn


class EquivalentBaselineANN(nn.Module):
    def __init__(self, sequence_frames, in_channels=1, num_classes=7):
        """
        sequence_frames: The depth/time dimension of your video clips.
        in_channels: 1 for raw grayscale, 2 if you are still applying the OnOffFilter.
        num_classes: Number of emotions in CK+ (usually 7 or 8).
        """
        super(EquivalentBaselineANN, self).__init__()

        # 1. 3D Convolution (Maps to layer::Convolution3D)
        # PyTorch format: (Depth, Height, Width) -> (3, 5, 5)
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=(3, 5, 5),
            stride=1,
            padding=0
        )

        # 2. Activation (Replaces Spiking Dynamics/WTA)
        self.relu = nn.ReLU()

        # 3. Pooling (Maps to TemporalPooling(2) and SumPooling(4))
        # PyTorch format: (Temporal, Spatial_H, Spatial_W) -> (2, 4, 4)
        self.pool3d = nn.AvgPool3d(
            kernel_size=(2, 4, 4),
            stride=(2, 4, 4)
        )

        self.flatten = nn.Flatten()

        # --- Calculate spatial dimensions for the Classifier ---
        # Input spatial size: 48 x 48
        # After 5x5 spatial conv: (48 - 5 + 1) = 44 x 44
        # After 4x4 spatial pool: (44 // 4) = 11 x 11

        # --- Calculate temporal dimensions for the Classifier ---
        # Input depth: sequence_frames (e.g., 10 frames)
        # After depth 3 conv: (sequence_frames - 3 + 1)
        # After temporal pool 2: (sequence_frames - 2) // 2
        conv_depth = sequence_frames - 3 + 1
        pool_depth = conv_depth // 2

        flattened_size = 64 * pool_depth * 11 * 11

        # 4. Classifier (Replaces analysis::Svm)
        self.classifier = nn.Linear(flattened_size, num_classes)

    def forward(self, x):
        # x shape expects: (Batch, Channels, Depth/Time, Height, Width)
        x = self.conv3d(x)
        x = self.relu(x)
        x = self.pool3d(x)

        features = self.flatten(x)
        output = self.classifier(features)

        return output, features  # Returning features in case you want to use Scikit-Learn SVM


# === Example Usage ===
# Assuming CK+ sequences of 10 frames, 1 channel (grayscale), 48x48 resolution
model = EquivalentBaselineANN(sequence_frames=10, in_channels=1, num_classes=7)

# Dummy input: Batch size of 16, 1 channel, 10 frames, 48x48
dummy_input = torch.randn(16, 1, 10, 48, 48)
predictions, extracted_features = model(dummy_input)

print("Output shape:", predictions.shape)  # Should be [16, 7]
print("Features shape:", extracted_features.shape)  # For SVM training