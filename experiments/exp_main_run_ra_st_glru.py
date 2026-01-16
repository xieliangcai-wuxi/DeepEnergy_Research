import sys
import os
import yaml
import torch
import numpy as np
import random
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import PowerDataset
from src.data.similarity import SimilarityEngine
from src.models.ra_st_glru import RA_ST_GLRU
from src.utils.trainer import StandardTrainer
from src.utils.logger import setup_logger
from torch.utils.data import DataLoader
from src.utils.metrics import calculate_metrics

def set_seed(seed):
    """固定所有随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def run_experiment(seed, config, logger):
    """运行单个种子的完整实验流程"""
    EXP_NAME = f"Ensemble_Seed_{seed}"
    
    # 1. 设置种子
    set_seed(seed)
    logger.info(f"启动实验子任务: {EXP_NAME} ...")

    # 2. 初始化 Similarity Engine
    sim_engine = SimilarityEngine(config)
    
    # 3. 数据集
    logger.info(f"[Seed {seed}] Loading Datasets...")
    train_ds = PowerDataset(config, mode='train', similarity_engine=sim_engine)
    val_ds = PowerDataset(config, mode='val', similarity_engine=sim_engine)
    test_ds = PowerDataset(config, mode='test', similarity_engine=sim_engine)
    
    train_loader = DataLoader(train_ds, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['train']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config['train']['batch_size'], shuffle=False)
    
    # 4. 模型初始化 (自动检测维度)
    # [修复] 正确的解包逻辑
    # Dataset 返回结构: ((x_num, x_text), x_sim, y)
    full_sample = train_ds[0] 
    (x_num, x_text), x_sim, y = full_sample
    
    in_features = x_num.shape[-1]
    logger.info(f"[Seed {seed}] Detected input features: {in_features}")
    
    model = RA_ST_GLRU(
        num_nodes=5,
        in_features=in_features,
        d_model=config['model']['d_model'],
        layers=config['model']['layers'],
        out_len=config['model']['out_len'],
        top_k=config['model']['top_k'],
        use_retrieval=config['model']['use_retrieval'],
        dropout=config['model'].get('dropout', 0.1)
    )
    
    # 5. 训练
    trainer = StandardTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        experiment_name=EXP_NAME
    )
    
    trainer.fit()
    
    # 6. 加载最优模型并生成预测结果
    logger.info(f"[Seed {seed}] Generating Predictions using Best Model...")
    best_model_path = os.path.join(config['paths']['result_dir'], EXP_NAME, "best_model.pth")
    
    if not os.path.exists(best_model_path):
        logger.error(f"[Seed {seed}] Model file not found: {best_model_path}")
        return None, None

    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    all_preds = []
    all_trues = []
    
    # 获取反归一化参数
    target_mean = trainer.target_mean
    target_std = trainer.target_std
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Predicting Seed {seed}", leave=False):
            (x_num, x_text), x_sim, y = batch
            x_num = x_num.to(trainer.device, dtype=torch.float32)
            x_text = x_text.to(trainer.device, dtype=torch.float32)
            x_sim = x_sim.to(trainer.device, dtype=torch.float32)
            
            # 预测
            preds = model((x_num, x_text), x_sim) # [Batch, Out_Len, 1]
            
            # 反归一化
            preds_real = preds.cpu().numpy() * target_std + target_mean
            y_real = y.numpy() * target_std + target_mean
            
            all_preds.append(preds_real)
            all_trues.append(y_real)
            
    # [Total_Samples, Out_Len, 1]
    final_preds = np.concatenate(all_preds, axis=0)
    final_trues = np.concatenate(all_trues, axis=0)
    
    return final_preds, final_trues

if __name__ == "__main__":
    # 1. 加载配置
    CONFIG_PATH = '../configs/exp_main_ra_st_glru.yaml'
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 2. 初始化主日志记录器
    main_logger = setup_logger(config['paths']['log_dir'], "Ensemble_Main")
    
    main_logger.info("="*60)
    main_logger.info(">>> STARTING ENSEMBLE EXPERIMENT")
    main_logger.info("="*60)

    # 定义要运行的种子列表
    SEEDS = [42, 2024, 1234, 777, 999]
    
    ensemble_preds = []
    ground_truth = None
    
    # 3. 循环运行实验
    for seed in SEEDS:
        main_logger.info(f"\n>>> Running for Seed: {seed}")
        try:
            preds, trues = run_experiment(seed, config, main_logger)
            if preds is not None:
                ensemble_preds.append(preds)
                if ground_truth is None:
                    ground_truth = trues
            else:
                main_logger.warning(f"Seed {seed} failed to produce predictions.")
        except Exception as e:
            main_logger.exception(f"Error occurred during seed {seed}: {str(e)}")
            
    # 4. 计算集成平均 (Model Averaging)
    if not ensemble_preds:
        main_logger.error("No predictions collected. Ensemble failed.")
        exit(1)

    # Stack: [5, N, 24, 1] -> Mean -> [N, 24, 1]
    avg_preds = np.mean(ensemble_preds, axis=0)
    
    main_logger.info("\n" + "="*40)
    main_logger.info(">>> ENSEMBLE RESULTS (Average of 5 Runs)")
    main_logger.info("="*40)
    
    # 5. 计算并记录指标
    metrics = calculate_metrics(avg_preds.flatten(), ground_truth.flatten())
    
    main_logger.info(f"Final Ensemble MAE:  {metrics['mae']:.2f} MW")
    main_logger.info(f"Final Ensemble RMSE: {metrics['rmse']:.2f} MW")
    main_logger.info(f"Final Ensemble MAPE: {metrics['mape']:.2f} %")
    main_logger.info("="*40)