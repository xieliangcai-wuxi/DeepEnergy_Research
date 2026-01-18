import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import time
from .logger import setup_logger

class StandardTrainer:
    """
    [StandardTrainer] L1 Loss + Warmup Edition
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
        
        self.result_dir = os.path.join(config['paths']['result_dir'], experiment_name)
        if not os.path.exists(self.result_dir): os.makedirs(self.result_dir)
        
        self.logger = setup_logger(self.result_dir, "training_log")
        
        # 🚨 [关键修改] 使用 L1Loss (MAE Loss) 而不是 MSE
        # L1 Loss 对异常值更鲁棒，适合 MAPE 指标
        self.criterion = nn.L1Loss() 
        
        self._setup_optimization()

    def _setup_optimization(self):
        lr = float(self.config['train']['learning_rate'])
        wd = float(self.config['train'].get('weight_decay', 1e-3))
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=wd
        )
        
        # 🚨 [关键修改] 引入 Warmup + Cosine Annealing
        # 前 5 个 Epoch 预热，防止模型一开始就掉进局部最优
        from torch.optim.lr_scheduler import LambdaLR
        
        warmup_epochs = 5
        max_epochs = self.config['train']['epochs']
        
        def lr_lambda(current_step):
            # Warmup logic
            if current_step < warmup_epochs:
                return float(current_step) / float(max(1, warmup_epochs))
            # Cosine decay logic
            progress = float(current_step - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        
        self.logger.info(f"Optimizer: AdamW (LR={lr}, WD={wd}) | Loss: L1Loss | Scheduler: Warmup({warmup_epochs}) + Cosine")

    def fit(self):
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config['train']['patience']
        epochs = self.config['train']['epochs']
        
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
                
                # Forward (Pass debug=False)
                pred = self.model((x_num, x_text), x_sim)
                
                loss = self.criterion(pred, y)
                loss.backward()
                
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['train'].get('clip_grad', 5.0))
                self.optimizer.step()
                train_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            
            # --- Scheduler Step (Per Epoch) ---
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # --- Val ---
            val_loss, real_mae, real_mape = self._validate()
            
            # --- Log ---
            self.logger.info(
                f"Epoch {epoch+1:03d} | LR: {current_lr:.6f} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Real MAE: {real_mae:.2f} MW | Real MAPE: {real_mape:.2f}%"
            )
            
            # --- Checkpoint ---
            # 注意：因为换了 L1Loss，Loss 的数值会变（变大或变小），但这不影响比较逻辑
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