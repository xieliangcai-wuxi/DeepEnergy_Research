import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import time
from .logger import setup_logger

class StandardTrainer:
    """
    [StandardTrainer] SOTA-Ready Trainer
    修复: 强制 weight_decay 类型转换，防止 YAML 解析错误
    """
    def __init__(self, model, train_loader, val_loader, test_loader, config, experiment_name):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.exp_name = experiment_name
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 结果保存路径
        self.result_dir = os.path.join(config['paths']['result_dir'], experiment_name)
        if not os.path.exists(self.result_dir): os.makedirs(self.result_dir)
        
        self.logger = setup_logger(self.result_dir, "training_log")
        self.criterion = nn.MSELoss() 
        
        self._setup_optimization()

    def _setup_optimization(self):
        # 1. 获取参数并强制转为 float
        lr = float(self.config['train']['learning_rate'])
        
        # [Fix] 这里加了 float()，无论 yaml 里写的是 "1e-3" 还是 1e-3，都能变回数字
        wd = float(self.config['train'].get('weight_decay', 1e-3))
        
        # 2. 定义优化器 AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=wd
        )
        
        # 3. Cosine 退火调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config['train']['epochs'], 
            eta_min=1e-6
        )
        
        self.logger.info(f"Optimizer: AdamW (LR={lr}, WD={wd})")

    def fit(self):
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config['train']['patience']
        epochs = self.config['train']['epochs']
        
        # 获取反归一化参数
        scaler = self.train_loader.dataset.scaler
        target_col = self.config['preprocessing']['target_col']
        num_cols = self.train_loader.dataset.numeric_cols
        
        if target_col in num_cols:
            target_idx = num_cols.index(target_col)
            self.target_std = scaler.scale_[target_idx]
            self.target_mean = scaler.mean_[target_idx]
        else:
            self.target_std = 1.0; self.target_mean = 0.0

        self.logger.info(f"开始训练 {epochs} Epochs...")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # --- Train ---
            self.model.train()
            train_losses = []
            
            for batch in self.train_loader:
                (x_num, x_text), x_sim, y = batch
                x_num = x_num.to(self.device, dtype=torch.float32)
                x_text = x_text.to(self.device, dtype=torch.float32)
                x_sim = x_sim.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.float32)
                
                self.optimizer.zero_grad()
                pred = self.model((x_num, x_text), x_sim)
                loss = self.criterion(pred, y)
                loss.backward()
                
                # 梯度裁剪 (防止梯度爆炸)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['train'].get('clip_grad', 5.0))
                
                self.optimizer.step()
                train_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            
            # --- Val ---
            val_loss, real_mae, real_mape = self._validate()
            
            # --- Scheduler ---
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            
            # --- Log ---
            self.logger.info(
                f"Epoch {epoch+1:03d} | LR: {current_lr:.6f} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Real MAE: {real_mae:.2f} MW | Real MAPE: {real_mape:.2f}%"
            )
            
            # --- Checkpoint ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.result_dir, "best_model.pth"))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.info("Early Stopping Triggered.")
                    break
                    
        self.logger.info(">>> Loading best model for testing...")
        self.model.load_state_dict(torch.load(os.path.join(self.result_dir, "best_model.pth")))
        
        # Final Test
        val_loss, final_mae, final_mape = self._validate()
        self.logger.info("="*30)
        self.logger.info("FINAL TEST RESULT:")
        self.logger.info(f"MAE:  {final_mae:.2f} MW")
        self.logger.info(f"RMSE: {np.sqrt(val_loss) * self.target_std:.2f} MW")
        self.logger.info(f"MAPE: {final_mape:.2f} %")
        self.logger.info("="*30)

    def _validate(self):
        self.model.eval()
        val_losses = []
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                (x_num, x_text), x_sim, y = batch
                x_num = x_num.to(self.device, dtype=torch.float32)
                x_text = x_text.to(self.device, dtype=torch.float32)
                x_sim = x_sim.to(self.device, dtype=torch.float32)
                y = y.to(self.device, dtype=torch.float32)
                
                pred = self.model((x_num, x_text), x_sim)
                loss = self.criterion(pred, y)
                val_losses.append(loss.item())
                
                pred_real = pred.cpu().numpy() * self.target_std + self.target_mean
                y_real = y.cpu().numpy() * self.target_std + self.target_mean
                all_preds.append(pred_real)
                all_trues.append(y_real)
                
        val_loss = np.mean(val_losses)
        
        preds_arr = np.concatenate(all_preds).flatten()
        trues_arr = np.concatenate(all_trues).flatten()
        
        mae = np.mean(np.abs(preds_arr - trues_arr))
        mape = np.mean(np.abs((preds_arr - trues_arr) / (trues_arr + 1e-5))) * 100
        
        return val_loss, mae, mape