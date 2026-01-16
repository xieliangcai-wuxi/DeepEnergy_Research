import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
from .logger import setup_logger
from .metrics import calculate_metrics

class StandardTrainer:
    """
    [科研工具] 标准训练器
    Update Log:
    - [Fix]: 移除了 ReduceLROnPlateau 中的 verbose=True 参数，适配 PyTorch 最新版本。
    - [Feature]: 加入 LR Scheduler，当 Val Loss 不降时自动减小学习率。
    - [Feature]: 加入反归一化 (Denormalization) 逻辑。
    """
    def __init__(self, model, train_loader, val_loader, test_loader, config, experiment_name):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.experiment_name = experiment_name
        
        # 路径与日志
        self.result_dir = os.path.join(config['paths']['result_dir'], experiment_name)
        os.makedirs(self.result_dir, exist_ok=True)
        self.logger = setup_logger(config['paths']['log_dir'], f"{experiment_name}_train")
        
        # 强制类型转换
        lr = float(config['train']['learning_rate'])
        weight_decay = float(config['train']['weight_decay'])
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # [关键修改] 初始化学习率调度器 (移除了 verbose=True)
        # mode='min': 当 loss 不再下降时触发
        # factor=0.5: 学习率减半
        # patience=3: 容忍 3 个 Epoch 不降
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        self.criterion = nn.SmoothL1Loss(beta=1.0)
        self.patience = int(config['train']['patience'])
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # 反归一化参数提取
        try:
            scaler = self.train_loader.dataset.scaler
            target_col_name = config['preprocessing']['target_col']
            numeric_cols = self.train_loader.dataset.numeric_cols
            target_idx = numeric_cols.index(target_col_name)
            self.target_mean = scaler.mean_[target_idx]
            self.target_std = scaler.scale_[target_idx]
            self.logger.info(f"反归一化参数: Mean={self.target_mean:.2f}, Std={self.target_std:.2f}")
        except Exception:
            self.target_mean = 0
            self.target_std = 1

    def _process_batch(self, batch):
        (x_num, x_text), x_sim, y = batch
        x_num = x_num.to(self.device, dtype=torch.float32)
        x_text = x_text.to(self.device, dtype=torch.float32)
        x_sim = x_sim.to(self.device, dtype=torch.float32)
        y = y.to(self.device, dtype=torch.float32)
        return (x_num, x_text), x_sim, y

    def _denormalize(self, tensor):
        return tensor * self.target_std + self.target_mean

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, leave=False, desc="Train")
        
        for batch in loop:
            self.optimizer.zero_grad()
            inputs, x_sim, y = self._process_batch(batch)
            preds = self.model(inputs, x_sim)
            loss = self.criterion(preds, y)
            
            if torch.isnan(loss):
                raise ValueError("Loss is NaN.")
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        return total_loss / len(self.train_loader)

    def evaluate(self, loader, mode="Val"):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for batch in tqdm(loader, leave=False, desc=mode):
                inputs, x_sim, y = self._process_batch(batch)
                preds = self.model(inputs, x_sim)
                loss = self.criterion(preds, y)
                total_loss += loss.item()
                
                pred_real = self._denormalize(preds)
                y_real = self._denormalize(y)
                all_preds.append(pred_real.cpu().numpy())
                all_trues.append(y_real.cpu().numpy())
                
        if len(all_preds) > 0:
            preds_concat = np.concatenate(all_preds, axis=0)
            trues_concat = np.concatenate(all_trues, axis=0)
            metrics = calculate_metrics(preds_concat.flatten(), trues_concat.flatten())
        else:
            metrics = {"mae": 0.0, "rmse": 0.0, "mape": 0.0}
        
        return total_loss / len(loader), metrics

    def fit(self):
        epochs = int(self.config['train']['epochs'])
        self.logger.info(f"开始训练 {epochs} Epochs...")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.evaluate(self.val_loader, "Val")
            
            # [Scheduler Step]
            # 更新学习率 (基于 Validation Loss)
            self.scheduler.step(val_loss)
            
            # 手动获取当前 LR 用于打印
            current_lr = self.optimizer.param_groups[0]['lr']

            self.logger.info(
                f"Epoch {epoch+1:03d} | LR: {current_lr:.6f} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Real MAE: {val_metrics['mae']:.2f} MW | Real MAPE: {val_metrics['mape']:.2f}%"
            )
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.result_dir, "best_model.pth"))
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    self.logger.info("Early Stopping Triggered.")
                    break
        
        # Final Test
        if os.path.exists(os.path.join(self.result_dir, "best_model.pth")):
            self.logger.info(">>> Loading best model for testing...")
            self.model.load_state_dict(torch.load(os.path.join(self.result_dir, "best_model.pth")))
            _, test_metrics = self.evaluate(self.test_loader, "Test")
            self.logger.info("="*30)
            self.logger.info(f"FINAL TEST RESULT:")
            self.logger.info(f"MAE:  {test_metrics['mae']:.2f} MW")
            self.logger.info(f"RMSE: {test_metrics['rmse']:.2f} MW")
            self.logger.info(f"MAPE: {test_metrics['mape']:.2f} %")
            self.logger.info("="*30)