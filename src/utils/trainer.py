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
    - [Feature]: 加入反归一化 (Denormalization) 逻辑，确保输出的 MAE/RMSE/MAPE 是真实的物理量(MW)。
    """
    def __init__(self, model, train_loader, val_loader, test_loader, config, experiment_name):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.experiment_name = experiment_name
        
        # 路径设置
        self.result_dir = os.path.join(config['paths']['result_dir'], experiment_name)
        os.makedirs(self.result_dir, exist_ok=True)
        
        self.logger = setup_logger(config['paths']['log_dir'], f"{experiment_name}_train")
        
        # 强制类型转换
        lr = float(config['train']['learning_rate'])
        weight_decay = float(config['train']['weight_decay'])
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        
        self.patience = int(config['train']['patience'])
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # [关键] 获取反归一化参数
        # 我们需要从 train_loader 的 dataset 中提取 target 列的 mean 和 std
        # 这是一个 Trick: 我们直接访问 dataset 的 scaler
        # 注意: 这里假设 dataset 使用了 StandardScaler 且 target 是其中的一列
        try:
            scaler = self.train_loader.dataset.scaler
            # 获取 target 列的索引
            target_col_name = config['preprocessing']['target_col']
            numeric_cols = self.train_loader.dataset.numeric_cols
            target_idx = numeric_cols.index(target_col_name)
            
            self.target_mean = scaler.mean_[target_idx]
            self.target_std = scaler.scale_[target_idx]
            self.logger.info(f"反归一化参数加载成功: Mean={self.target_mean:.4f}, Std={self.target_std:.4f}")
        except Exception as e:
            self.logger.warning(f"无法加载反归一化参数，指标将基于归一化数值计算: {e}")
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
        """将归一化的张量还原为真实物理值"""
        # y_real = y_norm * std + mean
        return tensor * self.target_std + self.target_mean

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, leave=False, desc="Train")
        
        for batch in loop:
            self.optimizer.zero_grad()
            inputs, x_sim, y = self._process_batch(batch)
            preds = self.model(inputs, x_sim)
            loss = self.criterion(preds, y) # Loss 依然在归一化空间计算，利于梯度下降
            
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
                loss = self.criterion(preds, y) # Val Loss 保持归一化空间，用于 Early Stopping
                total_loss += loss.item()
                
                # [关键] 反归一化后再存入列表，用于计算 MAE/MAPE
                pred_real = self._denormalize(preds)
                y_real = self._denormalize(y)
                
                all_preds.append(pred_real.cpu().numpy())
                all_trues.append(y_real.cpu().numpy())
                
        # 计算物理指标
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
            
            self.logger.info(
                f"Epoch {epoch+1:03d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Real MAE: {val_metrics['mae']:.2f} MW | "  # 显示单位 MW
                f"Real MAPE: {val_metrics['mape']:.2f}%"     # 显示百分比
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
        
        if os.path.exists(os.path.join(self.result_dir, "best_model.pth")):
            self.model.load_state_dict(torch.load(os.path.join(self.result_dir, "best_model.pth")))
            _, test_metrics = self.evaluate(self.test_loader, "Test")
            self.logger.info("="*30)
            self.logger.info(f"FINAL TEST (Real World Metrics):")
            self.logger.info(f"MAE:  {test_metrics['mae']:.2f} MW")
            self.logger.info(f"RMSE: {test_metrics['rmse']:.2f} MW")
            self.logger.info(f"MAPE: {test_metrics['mape']:.2f} %")
            self.logger.info("="*30)