import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import time
from copy import deepcopy
from torch.optim.swa_utils import AveragedModel, SWALR
from .logger import setup_logger

class StandardTrainer:
    """
    [StandardTrainer Ultimate]
    集成特性:
    1. L1 Loss: 对电力负荷预测更鲁棒
    2. Warmup + Cosine Scheduler: 优化收敛曲线
    3. Gradient Clipping: 防止梯度爆炸
    4. SWA (Stochastic Weight Averaging): 智能启用与回退
    5. Correct Metrics: 修复 RMSE 计算公式
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
        
        # Loss: 使用 L1 Loss (MAE) 作为优化目标
        self.criterion = nn.L1Loss() 
        
        # SWA Config
        self.use_swa = config['train'].get('use_swa', False)
        self.epochs = config['train']['epochs']
        # SWA 默认在最后 25% 的 Epochs 开启
        self.swa_start = config['train'].get('swa_start', int(self.epochs * 0.75))
        self.swa_model = None
        
        self._setup_optimization()

    def _setup_optimization(self):
        lr = float(self.config['train']['learning_rate'])
        wd = float(self.config['train'].get('weight_decay', 1e-2))
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=wd
        )
        
        # Scheduler: Warmup (5) + Cosine
        from torch.optim.lr_scheduler import LambdaLR
        warmup_epochs = 5
        
        def lr_lambda(current_epoch):
            # Warmup Phase
            if current_epoch < warmup_epochs:
                return float(current_epoch + 1) / float(max(1, warmup_epochs))
            # Cosine Phase
            progress = float(current_epoch - warmup_epochs) / float(max(1, self.epochs - warmup_epochs))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        
        # SWA Scheduler setup
        if self.use_swa:
            self.swa_model = AveragedModel(self.model)
            # SWA 通常使用一个恒定的较小学习率，这里取初始 LR 的一半
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=lr * 0.5)
        
        self.logger.info(f"Optimizer: AdamW (LR={lr}, WD={wd}) | SWA Enabled: {self.use_swa} (Start Epoch: {self.swa_start})")

    def fit(self):
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config['train']['patience']
        
        # 获取反归一化所需的统计量
        scaler = self.train_loader.dataset.scaler
        target_col = self.config['preprocessing']['target_col']
        num_cols = self.train_loader.dataset.numeric_cols
        
        if hasattr(scaler, 'mean_') and target_col in num_cols:
            target_idx = num_cols.index(target_col)
            self.target_std = scaler.scale_[target_idx]
            self.target_mean = scaler.mean_[target_idx]
        else:
            self.target_std = 1.0; self.target_mean = 0.0

        self.logger.info(f"🚀 开始训练 {self.epochs} Epochs... Target Std: {self.target_std:.2f}")
        
        # 记录实际训练到了哪个 epoch，用于判断 SWA 是否触发
        last_epoch = 0 

        for epoch in range(self.epochs):
            last_epoch = epoch
            start_time = time.time()
            
            # --- Train Step ---
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
                
                # 🛡️ Gradient Clipping (防止梯度爆炸)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['train'].get('clip_grad', 5.0))
                
                self.optimizer.step()
                train_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            
            # --- Scheduler Step (Mixed Strategy) ---
            if self.use_swa and epoch >= self.swa_start:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
                current_lr = self.swa_scheduler.get_last_lr()[0]
                sched_mode = "SWA"
            else:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                sched_mode = "Cos"

            # --- Validation Step ---
            # 如果在 SWA 阶段，验证 SWA 模型的效果
            val_model_to_use = self.swa_model if (self.use_swa and epoch >= self.swa_start) else self.model
            val_loss, real_mae, real_mape, real_rmse = self._validate(val_model_to_use)
            
            # --- Logging ---
            self.logger.info(
                f"Epoch {epoch+1:03d} [{sched_mode}] | LR: {current_lr:.6f} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"MAE: {real_mae:.2f} | MAPE: {real_mape:.2f}%"
            )
            
            # --- Checkpoint & Early Stopping ---
            # 只有在非 SWA 阶段才进行早停计数
            # (因为 SWA 阶段 Loss 可能不会降，但泛化性在变好，所以不建议早停)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # 保存当前最佳模型 (如果是 SWA 阶段存的就是 SWA 权重)
                state_dict = val_model_to_use.state_dict()
                torch.save(state_dict, os.path.join(self.result_dir, "best_model.pth"))
            else:
                # 只有还没进 SWA 才增加计数器
                if not (self.use_swa and epoch >= self.swa_start):
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.logger.info("Early Stopping Triggered.")
                        break

        # ==========================================
        # Final Test Logic (Smart Fallback)
        # ==========================================
        # 判断 SWA 是否真的生效了
        swa_actually_ran = self.use_swa and (last_epoch >= self.swa_start)
        
        if swa_actually_ran:
            self.logger.info(">>> SWA Finalizing: Updating BN statistics...")
            # 更新 BatchNorm 统计量 (对于 Transformer 类模型其实是非必须的，但为了严谨)
            torch.optim.swa_utils.update_bn(self.train_loader, self.swa_model, device=self.device)
            torch.save(self.swa_model.state_dict(), os.path.join(self.result_dir, "best_swa_model.pth"))
            
            self.logger.info(">>> [Test Mode] Using SWA Model.")
            final_model = self.swa_model
            model_tag = "SWA"
        else:
            if self.use_swa:
                self.logger.warning("⚠️ SWA was configured but Early Stopping triggered BEFORE SWA started.")
                self.logger.warning(f"   (Stopped at epoch {last_epoch}, SWA start was {self.swa_start})")
                self.logger.info(">>> [Test Mode] Fallback to Standard Best Model.")
            else:
                self.logger.info(">>> [Test Mode] Using Standard Best Model.")
            
            # 回退加载最佳普通模型
            self.model.load_state_dict(torch.load(os.path.join(self.result_dir, "best_model.pth")))
            final_model = self.model
            model_tag = "Standard"
        
        # 最终测试
        val_loss, final_mae, final_mape, final_rmse = self._validate(final_model)
        
        self.logger.info("="*40)
        self.logger.info(f"✅ FINAL TEST RESULT ({model_tag}):")
        self.logger.info(f"   MAE :  {final_mae:.2f} MW")
        self.logger.info(f"   RMSE:  {final_rmse:.2f} MW")
        self.logger.info(f"   MAPE:  {final_mape:.2f} %")
        self.logger.info("="*40)

    def _validate(self, model_to_use):
        """
        验证函数
        Return: Scaled_L1, Real_MAE, Real_MAPE, Real_RMSE
        """
        model_to_use.eval()
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
                
                pred = model_to_use((x_num, x_text), x_sim)
                loss = self.criterion(pred, y)
                val_losses.append(loss.item())
                
                # 反归一化
                pred_real = pred.cpu().numpy() * self.target_std + self.target_mean
                y_real = y.cpu().numpy() * self.target_std + self.target_mean
                
                all_preds.append(pred_real)
                all_trues.append(y_real)
                
        val_loss = np.mean(val_losses)
        
        preds_arr = np.concatenate(all_preds).flatten()
        trues_arr = np.concatenate(all_trues).flatten()
        
        # 真实指标计算
        mae = np.mean(np.abs(preds_arr - trues_arr))
        mape = np.mean(np.abs((preds_arr - trues_arr) / (trues_arr + 1e-5))) * 100
        
        # [Correct RMSE Logic] sqrt(mean(square_error))
        mse_real = np.mean((preds_arr - trues_arr) ** 2)
        rmse = np.sqrt(mse_real)
        
        return val_loss, mae, mape, rmse