import sys
import os
import yaml
import torch
import numpy as np
import random
import traceback
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import PowerDataset
from src.data.similarity import SimilarityEngine
from src.models.ra_st_glru import RA_ST_GLRU
from src.utils.trainer import StandardTrainer
from src.utils.logger import setup_logger
from src.utils.metrics import calculate_metrics

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class LogicAuditor:
    """逻辑审计员：负责检查数据和模型的逻辑一致性"""
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.target_col = config['preprocessing']['target_col']

    def check_data_alignment(self, batch, dataset):
        self.logger.info("\n🕵️ [Logic Check] Analyzing Batch Data Flow...")
        (x_num, x_text), x_sim, y = batch
        
        # 1. Target Index Check
        target_idx = dataset.get_target_idx()
        self.logger.info(f"   - Target Column: '{self.target_col}' (Index {target_idx})")
        
        # 2. Residual Assumption Check (关键!)
        # 提取 Input 里的"昨天"(Lag-24) 和 Label 里的"今天"
        # 理论上，这两者应该非常接近。如果差异巨大，说明数据错位了。
        
        # x_num: [Batch, Seq, Feat]
        # 取倒数 24 个点
        baseline = x_num[:, -24:, target_idx] # [Batch, 24]
        truth = y[:, :, 0]                    # [Batch, 24]
        
        # 计算平均差异 (MAE of Persistence Model)
        diff = torch.abs(baseline - truth).mean().item()
        baseline_mean = baseline.mean().item()
        
        self.logger.info(f"   - Baseline Mean: {baseline_mean:.4f}")
        self.logger.info(f"   - Label Mean:    {truth.mean().item():.4f}")
        self.logger.info(f"   - Persistence Error (Diff): {diff:.4f}")
        
        if diff > 1.0: # 归一化后的数据，差异不应超过 1个标准差
            self.logger.warning("⚠️ [WARNING] Large difference between Input-Lag and Label! Data might be shuffled or misaligned.")
        else:
            self.logger.info("✅ [PASS] Input-Lag matches Label reasonably well. Residual learning is valid.")

def run_single_experiment(config_path):
    # --- 0. Config Load & Path Fix ---
    if not os.path.exists(config_path): raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f: config = yaml.safe_load(f)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"DEBUG: Project Root: {project_root}")

    # Windows Path Fix
    for key in ['raw_data_path', 'processed_dir', 'output_dir', 'log_dir', 'result_dir']:
        if key in config['paths']:
            rel = config['paths'][key].replace("./", "").replace("../", "") # Cleanup
            config['paths'][key] = os.path.normpath(os.path.join(project_root, rel))
            
    # Check Data Existence
    train_file = os.path.join(config['paths']['output_dir'], "train.csv")
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"❌ Data file missing at: {train_file}")

    # --- 1. Init ---
    SEED = 42
    set_seed(SEED)
    EXP_NAME = f"SOTA_Residual_MLP_Seed{SEED}"
    logger = setup_logger(config['paths']['log_dir'], "Main_Exp")
    logger.info(f"🚀 Starting Experiment: {EXP_NAME}")

    # --- 2. Data ---
    logger.info(">>> [1/5] Loading Datasets...")
    sim_engine = SimilarityEngine(config)
    train_ds = PowerDataset(config, mode='train', similarity_engine=sim_engine)
    val_ds = PowerDataset(config, mode='val', similarity_engine=sim_engine)
    test_ds = PowerDataset(config, mode='test', similarity_engine=sim_engine)
    
    train_loader = DataLoader(train_ds, batch_size=config['train']['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config['train']['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config['train']['batch_size'], shuffle=False, num_workers=0)

    # --- 3. Logic Audit ---
    logger.info(">>> [2/5] Auditing Logic...")
    auditor = LogicAuditor(config, logger)
    sample_batch = next(iter(train_loader))
    auditor.check_data_alignment(sample_batch, train_ds)
    
    # Get Shapes
    (x_num, x_text), x_sim, y = sample_batch
    in_features = x_num.shape[-1]
    target_idx = train_ds.get_target_idx()

    # --- 4. Model ---
    logger.info(">>> [3/5] Building Model (MLP Head + Residual)...")
    model = RA_ST_GLRU(
        num_nodes=5,
        in_features=in_features,
        d_model=config['model']['d_model'],
        layers=config['model']['layers'],
        out_len=config['model']['out_len'],
        top_k=config['model']['top_k'],
        target_idx=target_idx,
        use_retrieval=config['model']['use_retrieval'],
        dropout=config['model'].get('dropout', 0.1)
    )
    
    # --- 5. Dry Run ---
    logger.info(">>> [4/5] Dry Run...")
    model.to('cuda')
    model.eval()
    try:
        with torch.no_grad():
            out = model((x_num.cuda(), x_text.cuda()), x_sim.cuda())
            if out.shape != y.shape:
                raise ValueError(f"Shape Mismatch: Out {out.shape} != Label {y.shape}")
            logger.info("✅ Dry Run Passed.")
    except Exception as e:
        logger.error(traceback.format_exc())
        raise e

    # --- 6. Train ---
    logger.info(">>> [5/5] Training...")
    trainer = StandardTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        experiment_name=EXP_NAME
    )
    trainer.fit()
    
    # Final Eval
    best_model_path = os.path.join(config['paths']['result_dir'], EXP_NAME, "best_model.pth")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    # Denormalize
    mean_val = train_ds.scaler.mean_[target_idx]
    std_val = train_ds.scaler.scale_[target_idx]
    
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            (x_num, x_text), x_sim, y = batch
            preds = model((x_num.cuda(), x_text.cuda()), x_sim.cuda())
            all_preds.append(preds.cpu().numpy() * std_val + mean_val)
            all_trues.append(y.numpy() * std_val + mean_val)
            
    final_preds = np.concatenate(all_preds, axis=0)
    final_trues = np.concatenate(all_trues, axis=0)
    
    metrics = calculate_metrics(final_preds.flatten(), final_trues.flatten())
    logger.info("="*40)
    logger.info(f"✅ FINAL SOTA RESULT")
    logger.info(f"  MAPE: {metrics['mape']:.2f} %")
    logger.info("="*40)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'configs', 'exp_main_ra_st_glru.yaml')
    run_single_experiment(config_path)