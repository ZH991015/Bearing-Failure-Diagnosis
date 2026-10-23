"""
Train the STL-Net + physics model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import os
from datetime import datetime
import json

from dataset_final_modified import create_dataloaders
from model_final import create_model

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class Trainer:

    def __init__(self, model, train_loader, val_loader, test_loader,
                 device, num_classes, save_dir='checkpoints', split_mode='temporal',num_epochs=50):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        self.save_dir = save_dir
        self.split_mode = split_mode
        self.num_epochs=num_epochs

        os.makedirs(save_dir, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        self.warmup_epochs = 3
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.num_epochs - self.warmup_epochs, eta_min=1e-6
        )

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_auc': [],
            # test metrics are only logged, they never drive the model selection
            'test_acc': [], 'test_normal_recall': [],
            'split_mode': split_mode
        }

        # model selection is based on the validation accuracy
        self.best_val_acc = 0.0
        self.best_val_auc = 0.0
        # test accuracy of the selected checkpoint, kept for reporting only
        self.best_test_acc = 0.0

    def train_epoch(self):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 50 == 0:
                print(f'  Batch [{batch_idx + 1}/{len(self.train_loader)}] '
                      f'Loss: {loss.item():.4f} Acc: {100. * correct / total:.2f}%')

        return running_loss / len(self.train_loader), 100. * correct / total

    def validate(self):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_probs.append(F.softmax(outputs, dim=1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        try:
            all_probs = np.concatenate(all_probs, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            epoch_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except Exception as e:
            print(f"AUC failed: {e}")
            epoch_auc = 0.0

        return epoch_loss, epoch_acc, epoch_auc

    def calculate_normal_recall(self):
        """Recall of the healthy class (label 0) on the test set."""
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        normal_mask = (all_labels == 0)

        if normal_mask.sum() > 0:
            return (all_preds[normal_mask] == 0).sum() / normal_mask.sum()
        return 0.0

    def _evaluate_test(self):
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return 100. * correct / total

    def train(self, num_epochs=50, early_stopping_patience=3):
        print(f"\ntraining ({self.split_mode} mode) ...")
        print(f"device: {self.device}")
        print(f"epochs: {num_epochs}")
        print("=" * 80)

        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
            print("-" * 80)

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_auc = self.validate()
            test_acc_epoch = self._evaluate_test()
            test_normal_recall = self.calculate_normal_recall()

            if epoch < self.warmup_epochs:
                # linear warmup
                warmup_lr = 5e-4 * (epoch + 1) / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            else:
                self.scheduler.step()

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            self.history['test_acc'].append(test_acc_epoch)
            self.history['test_normal_recall'].append(test_normal_recall)

            torch.cuda.empty_cache()

            print(f"\ntrain - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, AUC: {val_auc:.4f}")
            print(f"test  - Acc: {test_acc_epoch:.2f}%, Normal Recall: {test_normal_recall:.4f} (monitoring only)")

            # pick the checkpoint by validation accuracy, the test set is not used here
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_auc = val_auc
                self.best_test_acc = test_acc_epoch  # recorded, not used for selection
                self.save_checkpoint(epoch, 'best_model.pth')
                print(f"saved best model (val Acc: {val_acc:.2f}%, val AUC: {val_auc:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"\nearly stop: val acc did not improve for {early_stopping_patience} epochs")
                break

        print("\ntraining finished")
        print(f"best val Acc (selection criterion): {self.best_val_acc:.2f}%")
        print(f"best val AUC: {self.best_val_auc:.4f}")
        print(f"test Acc of the selected checkpoint: {self.best_test_acc:.2f}%")

        self.save_history()

    def test(self):
        print("\ntesting ...")
        self.model.eval()

        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        test_acc = 100. * np.sum(all_preds == all_labels) / len(all_labels)

        class_names = ['无损伤', '滚动体故障', '内圈故障', '外圈故障']
        print("\nclassification report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))

        self.plot_confusion_matrix(all_labels, all_preds, class_names)
        self.plot_roc_curves(all_labels, all_probs, class_names)

        return test_acc, all_preds, all_labels, all_probs

    def plot_confusion_matrix(self, labels, preds, class_names):
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.title(f'混淆矩阵 ({self.split_mode} 模式)', fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"confusion matrix saved to {save_path}")
        plt.close()

    def plot_roc_curves(self, labels, probs, class_names):
        try:
            labels_bin = label_binarize(labels, classes=range(len(class_names)))
            fpr, tpr, roc_auc = {}, {}, {}
            for i in range(len(class_names)):
                fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            plt.figure(figsize=(10, 8))
            colors = ['blue', 'red', 'green', 'orange']
            for i, color in zip(range(len(class_names)), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label=f'{class_names[i]} (AUC = {roc_auc[i]:.4f})')

            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='随机猜测')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('假阳性率', fontsize=12)
            plt.ylabel('真阳性率', fontsize=12)
            plt.title(f'ROC曲线 ({self.split_mode} 模式)', fontsize=14)
            plt.legend(loc="lower right", fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            save_path = os.path.join(self.save_dir, 'roc_curves.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
            plt.close()
        except Exception as e:
            print(f"ROC plotting failed: {e}")

    def plot_training_history(self):
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))

        axes[0].plot(self.history['train_loss'], label='训练损失')
        axes[0].plot(self.history['val_loss'], label='验证损失')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('损失曲线')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self.history['train_acc'], label='训练准确率')
        axes[1].plot(self.history['val_acc'], label='验证准确率')
        axes[1].plot(self.history['test_acc'], label='测试准确率', linestyle='--')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('准确率曲线')
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(self.history['val_auc'], label='验证AUC', color='green')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUC')
        axes[2].set_title('验证AUC')
        axes[2].legend()
        axes[2].grid(True)
        axes[2].set_ylim([0, 1.05])

        axes[3].plot(self.history['test_normal_recall'], label='Normal Recall', color='purple')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Recall')
        axes[3].set_title('无损伤类Recall')
        axes[3].legend()
        axes[3].grid(True)
        axes[3].set_ylim([0, 1.05])

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"training history saved to {save_path}")
        plt.close()

    def save_checkpoint(self, epoch, filename):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_auc': self.best_val_auc,
            'best_test_acc': self.best_test_acc,
            'history': self.history,
            'split_mode': self.split_mode
        }
        torch.save(checkpoint, os.path.join(self.save_dir, filename))

    def save_history(self):
        save_path = os.path.join(self.save_dir, 'training_history.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=4, ensure_ascii=False)
        print(f"training history saved to {save_path}")

    def load_checkpoint(self, filename):
        load_path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.best_val_auc = checkpoint.get('best_val_auc', 0.0)
        self.best_test_acc = checkpoint.get('best_test_acc', 0.0)
        self.history = checkpoint['history']
        print(f"checkpoint loaded: {load_path}")


def single_experiment(split_mode='temporal', temporal_direction='backward'):
    DATA_ROOT = r"E:\航空发动机\故障诊断"
    MODEL_TYPE = 'stl_physics'
    BATCH_SIZE = 128
    WINDOW_SIZE = 2048
    STRIDE = 512
    NUM_EPOCHS = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # data
    print("\n" + "=" * 80)
    mode_desc = f"{split_mode}-{temporal_direction}" if split_mode == 'temporal' else split_mode
    print(f"building dataloaders ({mode_desc} mode)")
    print("=" * 80)

    train_loader, val_loader, test_loader, num_classes, full_dataset = create_dataloaders(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        split_mode=split_mode,
        temporal_direction=temporal_direction
    )

    # model
    print("\n" + "=" * 80)
    print("building model")
    print("=" * 80)

    model = create_model(
        model_type=MODEL_TYPE,
        num_sensors=5,
        window_size=WINDOW_SIZE,
        num_classes=num_classes,
        fs=16000
    )

    # rpm statistics
    model.set_rpm_stats(
        mean=full_dataset.global_rpm_mean,
        std=full_dataset.global_rpm_std
    )

    # training
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'checkpoints/{MODEL_TYPE}_{split_mode}_{timestamp}'

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        num_classes=num_classes,
        save_dir=save_dir,
        split_mode=split_mode,
        num_epochs = NUM_EPOCHS
    )

    trainer.train(num_epochs=NUM_EPOCHS, early_stopping_patience=3)
    trainer.plot_training_history()

    # test with the best checkpoint
    trainer.load_checkpoint('best_model.pth')
    test_acc, preds, labels, probs = trainer.test()

    print("\n" + "=" * 80)
    print("done")
    print("=" * 80)
    print(f"split: {mode_desc}")
    print(f"best val Acc: {trainer.best_val_acc:.2f}%")
    print(f"best val AUC: {trainer.best_val_auc:.4f}")
    print(f"test Acc of the selected checkpoint: {trainer.best_test_acc:.2f}%")
    print(f"final test Acc: {test_acc:.2f}%")
    print(f"checkpoints in {save_dir}")


if __name__ == "__main__":
    single_experiment(split_mode='temporal', temporal_direction='forward')
