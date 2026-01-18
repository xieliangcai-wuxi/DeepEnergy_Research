import sys
import os
import yaml
import torch
import numpy as np
import random
import traceback
from tqdm import tqdm
from torch.utils.data import DataLoader

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

def run_single_experiment(config_path):
    # --- 0. Config & Path ---
    if not os.path.exists(config_path): raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f: config = yaml.safe_load(f)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for key in ['raw_data_path', 'processed_dir', 'output_dir', 'log_dir', 'result_dir']:
        if key in config['paths']:
            rel = config['paths'][key].replace("./", "").replace("../", "")
            config['paths'][key] = os.path.normpath(os.path.join(project_root, rel))
            
    SEED = 42
    set_seed(SEED)
    EXP_NAME = f"SOTA_Audit_Run_Seed{SEED}"
    logger = setup_logger(config['paths']['log_dir'], "Main_Exp")
    logger.info(f"🚀 Starting Forensic Experiment: {EXP_NAME}")

    # --- 1. Data ---
    logger.info(">>> [1/5] Loading Data...")
    sim_engine = SimilarityEngine(config)
    train_ds = PowerDataset(config, mode='train', similarity_engine=sim_engine)
    val_ds = PowerDataset(config, mode='val', similarity_engine=sim_engine)
    test_ds = PowerDataset(config, mode='test', similarity_engine=sim_engine)
    
    train_loader = DataLoader(train_ds, batch_size=config['train']['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config['train']['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config['train']['batch_size'], shuffle=False, num_workers=0)

    # --- 2. Data Audit (Before Model) ---
    logger.info(">>> [2/5] Auditing Data Stream...")
    sample_batch = next(iter(train_loader))
    (x_num, x_text), x_sim, y = sample_batch
    target_idx = train_ds.get_target_idx()
    in_features = x_num.shape[-1]
    
    # Check 1: Persistence Error (The Baseline)
    baseline = x_num[:, -24:, target_idx]
    truth = y[:, :, 0]
    diff = torch.abs(baseline - truth).mean().item()
    logger.info(f"   📊 [Data Check] Baseline (Lag-24) vs Truth Diff: {diff:.4f}")
    if diff > 1.5:
        logger.warning("   ⚠️ WARNING: Large Persistence Error. Data might be shuffled or misaligned!")
    else:
        logger.info("   ✅ Data Alignment looks good.")

    # --- 3. Model ---
    logger.info(">>> [3/5] Building Model (RevIN + NoMask + MLP)...")
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
    
    # --- 4. Dry Run (With Internal Debug Print) ---
    logger.info(">>> [4/5] Executing Dry Run (Debug Mode ON)...")
    model.to('cuda')
    model.eval()
    try:
        with torch.no_grad():
            # 🚨 触发模型内部的打印逻辑
            out = model((x_num.cuda(), x_text.cuda()), x_sim.cuda(), debug=True)
            
            logger.info(f"   📦 Output Shape: {out.shape}")
            if torch.isnan(out).any():
                raise ValueError("❌ NaNs detected in output!")
            logger.info("✅ Dry Run Passed.")
    except Exception as e:
        logger.error(traceback.format_exc())
        raise e

    # --- 5. Train ---
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
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        
        mean_val = train_ds.scaler.mean_[target_idx]
        std_val = train_ds.scaler.scale_[target_idx]
        
        all_preds, all_trues = [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                (x_num, x_text), x_sim, y = batch
                preds = model((x_num.cuda(), x_text.cuda()), x_sim.cuda(), debug=False) # Train/Test 关闭 debug
                all_preds.append(preds.cpu().numpy() * std_val + mean_val)
                all_trues.append(y.numpy() * std_val + mean_val)
                
        final_preds = np.concatenate(all_preds, axis=0)
        final_trues = np.concatenate(all_trues, axis=0)
        metrics = calculate_metrics(final_preds.flatten(), final_trues.flatten())
        logger.info("="*40)
        logger.info(f"✅ FINAL RESULT")
        logger.info(f"  MAPE: {metrics['mape']:.2f} %")
        logger.info("="*40)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'configs', 'exp_main_ra_st_glru.yaml')
    run_single_experiment(config_path)