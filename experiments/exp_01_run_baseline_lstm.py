import sys
import os
import yaml
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import PowerDataset
from src.models.baseline_lstm import BaselineLSTM # (基线模型)
from src.utils.trainer import StandardTrainer

def run_experiment():
    EXP_NAME = "exp_01_baseline_lstm"
    # 加载专门针对 LSTM 的配置
    CONFIG_PATH = "./configs/exp_01_baseline_lstm.yaml"
    
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # ... (数据加载逻辑相同) ...
    
    # 初始化基线模型
    model = BaselineLSTM(
        input_size=config['model']['input_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['layers']
    )
    
    # 训练
    trainer = StandardTrainer(..., experiment_name=EXP_NAME)
    trainer.fit()

if __name__ == "__main__":
    run_experiment()