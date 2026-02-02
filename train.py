import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
from tqdm import tqdm
import json

from models.model import create_model
from utils.data_loader import get_dataloaders


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cpu')
        print(f"Using device: {self.device}")

        self.train_loader, self.val_loader, self.test_loader, self.class_to_idx = get_dataloaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            image_size=config['image_size'],
            num_workers=config['num_workers']
        )

        self.num_classes = len(self.class_to_idx)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.model = create_model(
            num_classes=self.num_classes,
            model_type=config['model_type'],
            pretrained=config['pretrained']
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config['learning_rate'] * 0.01
        )

        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(config['checkpoint_dir'], 'logs'))

        self.best_val_acc = 0.0

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Train]")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']} [Val]")
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({
                    'loss': f'{running_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'class_to_idx': self.class_to_idx,
            'config': self.config
        }

        checkpoint_path = os.path.join(self.config['checkpoint_dir'], 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")

    def train(self):
        print(f"\nStarting training...")
        print(f"Number of classes: {self.num_classes}")
        print(f"Class mapping: {self.class_to_idx}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        print()

        start_time = time.time()

        for epoch in range(self.config['epochs']):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            self.scheduler.step()

            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}\n")

            is_best = val_acc > self.best_val_acc
            self.best_val_acc = max(val_acc, self.best_val_acc)
            self.save_checkpoint(epoch, val_acc, is_best)

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")

        self.writer.close()

        return self.model

    def test(self):
        print("\nRunning test evaluation...")
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Testing"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_acc = 100. * correct / total
        print(f"Test Accuracy: {test_acc:.2f}%")

        return test_acc


def main():
    config = {
        'data_dir': 'data/flowers',
        'batch_size': 16,
        'image_size': 224,
        'num_workers': 0,
        'model_type': 'resnet18',
        'pretrained': True,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'epochs': 30,
        'checkpoint_dir': 'checkpoints'
    }

    trainer = Trainer(config)
    model = trainer.train()
    test_acc = trainer.test()

    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")


if __name__ == '__main__':
    main()
