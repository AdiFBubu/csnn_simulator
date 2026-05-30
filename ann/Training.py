import os

import torch.optim as optim
import torch
from torch.utils.data import DataLoader

from ann.CKPlusDataset import MAX_FRAMES, CKPlusDataset, pad_collate_fn
from ann.EquivalentBaselineANN import EquivalentBaselineANN

# --- Configuration ---
EPOCHS = 800  # Matching your C++ default

CSV_PATH = os.getenv("CK_PLUS_CSV_PATH")
IMAGES_DIR = os.getenv("CK_PLUS_IMAGES_DIR")
RANDOM_SEED = 42
BATCH_SIZE = 16

LEARNING_RATE = 0.001
NUM_FOLDS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {DEVICE}")

for fold in range(1, NUM_FOLDS + 1):
    print(f"\n========== RUNNING FOLD {fold} ==========")

    # 1. Initialize Datasets and DataLoaders (from previous step)
    train_dataset = CKPlusDataset(
        csv_path=CSV_PATH,
        images_dir=IMAGES_DIR,
        fold_to_use=fold,
        is_train=True,
        random_seed=RANDOM_SEED
    )

    test_dataset = CKPlusDataset(
        csv_path=CSV_PATH,
        images_dir=IMAGES_DIR,
        fold_to_use=fold,
        is_train=False,
        random_seed=RANDOM_SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=pad_collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=pad_collate_fn
    )

    # 2. Initialize Model, Loss, and Optimizer
    # Make sure sequence_frames matches the MAX_FRAMES used in collate_fn
    model = EquivalentBaselineANN(sequence_frames=MAX_FRAMES, in_channels=1, num_classes=7).to(DEVICE)

    # CrossEntropyLoss automatically applies Softmax, which replaces your SNN's WTA (Winner-Takes-All)
    criterion = torch.nn.CrossEntropyLoss()

    # Adam is the standard optimizer for CNNs, replacing your STDP biological learning rule
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. The Epoch Loop
    for epoch in range(1, EPOCHS + 1):

        # --- TRAINING PHASE ---
        model.train()  # Set model to training mode (enables dropout/batchnorm if you add them later)
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)

            # Step A: Clear old gradients
            optimizer.zero_grad()

            # Step B: Forward pass (Extract features and get predictions)
            predictions, _ = model(sequences)

            # Step C: Calculate the error
            loss = criterion(predictions, labels)

            # Step D: Backward pass (Calculate gradients - this replaces STDP)
            loss.backward()

            # Step E: Update weights
            optimizer.step()

            # Track metrics
            train_loss += loss.item() * sequences.size(0)
            _, predicted_classes = torch.max(predictions, 1)
            total_train += labels.size(0)
            correct_train += (predicted_classes == labels).sum().item()

        train_accuracy = 100 * correct_train / total_train
        avg_train_loss = train_loss / total_train

        # --- TESTING PHASE ---
        model.eval()  # Set model to evaluation mode
        test_loss = 0.0
        correct_test = 0
        total_test = 0

        # torch.no_grad() disables gradient calculation to save memory and compute during testing
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)

                predictions, _ = model(sequences)
                loss = criterion(predictions, labels)

                test_loss += loss.item() * sequences.size(0)
                _, predicted_classes = torch.max(predictions, 1)
                total_test += labels.size(0)
                correct_test += (predicted_classes == labels).sum().item()

        test_accuracy = 100 * correct_test / total_test
        avg_test_loss = test_loss / total_test

        # Print progress (e.g., every 10 epochs to avoid spamming the console)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Fold {fold} | Epoch [{epoch}/{EPOCHS}] | "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
                  f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

    print(f"Fold {fold} Completed! Final Test Accuracy: {test_accuracy:.2f}%")