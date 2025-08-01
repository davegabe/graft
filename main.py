import sys
sys.path.append('.')
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test, load_content
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
import shutil

from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
from models.DLinear import DLinear
from models.FSCA import FSCA
from models.GraFT import GraFT

torch.set_num_threads(4)

def remove_directory_if_exists(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Dir '{path}' deleted。")
    else:
        print(f"Dir '{path}' does not exist.")

def randomize_weights_uniform(model, min_v=-1.0, max_v=1.0):
    for param in model.parameters():
        nn.init.uniform_(param, a=min_v, b=max_v) 

def randomize_weights_uniform_minmax(model):
    for param in model.parameters():
        min_v = param.min().detach()
        max_v = param.max().detach()
        nn.init.uniform_(param, a=min_v, b=max_v)

def get_data(data):
    if data == 'ETTh1':
        root_path = './datasets/ETT-small/'
        data_path = 'ETTh1.csv'
    elif data == 'ETTh2':
        root_path = './datasets/ETT-small/'
        data_path = 'ETTh2.csv'
    elif data == 'ETTm1':
        root_path = './datasets/ETT-small/'
        data_path = 'ETTm1.csv'
    elif data == 'ETTm2':
        root_path = './datasets/ETT-small/'
        data_path = 'ETTm2.csv'
    else:
        pass

    return root_path, data_path


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='GPT4TS')

parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
parser.add_argument('--root_path', type=str, default='./datasets/ETT-small/')
parser.add_argument('--data_path', type=str, default='ETTh1.csv')
parser.add_argument('--data', type=str, default='ETTh1')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--freq', type=int, default=0)
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--pred_len', type=int, default=720)
parser.add_argument('--label_len', type=int, default=48)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--decay_fac', type=float, default=0.5)
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--train_epochs', type=int, default=100)
parser.add_argument('--lradj', type=str, default='COS')
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--enc_in', type=int, default=7)
parser.add_argument('--c_out', type=int, default=7)
parser.add_argument('--kernel_size', type=int, default=25)
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--model', type=str, default='GraFT')
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--tmax', type=int, default=20)
parser.add_argument('--patch_size_stride', type=str, default="48 48", help='A space-separated string of integers')
parser.add_argument('--itr', type=int, default=1)
parser.add_argument('--cos', type=int, default=0)
parser.add_argument('--fname', default='./checkpoints/', type=str, help='specify checkpoint run name')
parser.add_argument('--run_name', default='test', type=str)
parser.add_argument('--wd_project', default='llm_test', type=str)
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--freeze_all', type=int, default=0) 
parser.add_argument('--test_random', type=int, default=0) 
parser.add_argument('--test_random_type', type=str, default='ori') 
parser.add_argument('--sweep_flag', type=int, default=0) 
parser.add_argument('--test_flag', type=int, default=0)
parser.add_argument('--in_dropout', type=float, default=0)
parser.add_argument('--out_dropout', type=float, default=0.1)
parser.add_argument('--fix_len', type=int, default=70)
parser.add_argument('--split_len', type=int, default=2)
parser.add_argument('--layer_index', type=str, default="6*0_2_4*", help='A space-separated string of integers')
parser.add_argument('--colon_token', type=int, default=25)
parser.add_argument('--num_statistics', type=int, default=4)
parser.add_argument('--len_statistics', type=int, default=5)
parser.add_argument('--lr_edge_weights', type=float, default=0.001) 
parser.add_argument('--eta_min', type=float, default=1e-8)
parser.add_argument('--edge_eta_min', type=float, default=1e-8)
parser.add_argument('--revin_flag', type=int, default=0)
parser.add_argument('--test', type=int, default=0)
parser.add_argument('--multi', type=int, default=1)
parser.add_argument('--train_shuffle_int', type=int, default=0)
parser.add_argument('--w_l2s_flag', type=int, default=0)
parser.add_argument('--w_l2s_v', type=float, default=0.0001)
parser.add_argument('--d_l_comp', type=int, default=512)
parser.add_argument('--gnn_hidden_prg', type=int, default=768)
parser.add_argument('--gnn_layers_prg', type=int, default=2)
parser.add_argument('--k_sim', type=int, default=5)
parser.add_argument('--gate_init_prg', type=float, default=0.5)
parser.add_argument('--data_flow', type=str, default=None, help='for zero-shot, e.g., ETTh1_ETTh2')

args = parser.parse_args()

args.train_shuffle = bool(args.train_shuffle_int)

if not args.multi:
    args.enc_in = 1

parts = args.layer_index.split('*')

args.gpt_layers = int(parts[0]) if parts[0] else None

if len(parts) > 1 and parts[1]:
    args.gnn_layer_index = [int(x) for x in parts[1].split('_')]
    args.gnn_layer_index_str = '_'.join(str(x) for x in args.gnn_layer_index)
else:
    args.gnn_layer_index = []
    args.gnn_layer_index_str = ""

if len(parts) > 2 and parts[2]:
    args.l_gnn_layer_index = [int(x) for x in parts[2].split('_')]
    args.l_gnn_layer_index_str = '_'.join(str(x) for x in args.l_gnn_layer_index)
else:
    args.l_gnn_layer_index = []
    args.l_gnn_layer_index_str = ""


args.patch_size, args.stride = [int(x) for x in args.patch_size_stride.split()]

is_zero_shot = args.data_flow is not None
if is_zero_shot:
    args.data, args.test_data = args.data_flow.split('_')
    args.root_path, args.data_path = get_data(args.data)
    args.test_root_path, args.test_data_path = get_data(args.test_data)


if args.sweep_flag == 1:
    if args.data == 'ETTh1':
        args.root_path = './datasets/ETT-small/'
        args.data_path = 'ETTh1.csv'
    elif args.data == 'ETTh2':
        args.root_path = './datasets/ETT-small/'
        args.data_path = 'ETTh2.csv'
    elif args.data == 'ETTm1':
        args.root_path = './datasets/ETT-small/'
        args.data_path = 'ETTm1.csv'
    elif args.data == 'ETTm2':
        args.root_path = './datasets/ETT-small/'
        args.data_path = 'ETTm2.csv'
    elif args.data == 'custom_traffic':
        args.root_path = './datasets/traffic/'
        args.data_path = 'traffic.csv'
    elif args.data == 'custom_weather':
        args.root_path = './datasets/weather/'
        args.data_path = 'weather.csv'
    elif args.data == 'custom_electricity':
        args.root_path = './datasets/electricity/'
        args.data_path = 'electricity.csv'
    elif args.data == 'custom_illness':
        args.root_path = './datasets/illness/'
        args.data_path = 'national_illness.csv'
else:
    pass

args.data_name = args.data_path.split('.')[0]

fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

SEASONALITY_MAP = {
   "minutely": 1440,
   "10_minutes": 144,
   "half_hourly": 48,
   "hourly": 24,
   "daily": 7,
   "weekly": 1,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}

mses = []
maes = []

args.train_shuffle_int = 1 if args.train_shuffle else 0

for ii in range(args.itr):
    
    group_name = '{}_ps{}_{}_sl{}_df{}_gl{}_{}*{}_spl{}_b{}_l{}_em{}_rf{}_e{}_mul{}_ts{}_wf{}_wv{}_id{}_od{}_dl{}_s{}{}'.format(
        args.model, 
        args.patch_size,
        args.stride,
        args.seq_len,
        args.d_ff,

        args.gpt_layers,
        args.gnn_layer_index_str,
        args.l_gnn_layer_index_str,
        args.split_len,
        args.batch_size,

        args.learning_rate,
        args.eta_min,
        args.revin_flag,
        args.train_epochs,
        args.multi,

        args.train_shuffle_int,
        args.w_l2s_flag,
        args.w_l2s_v,
        args.in_dropout,
        args.out_dropout,

        args.d_l_comp,

        args.seed,
        args.patience,
        )
           
    setting = 'r{}_nh{}_dn{}_pl{}'.format(
        args.run_name,
        args.n_heads,
        args.data_name,
        args.pred_len
        )
    
    import datetime
    import uuid
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    random_string = uuid.uuid4().hex[:6]
    args.it_fname = os.path.join(args.fname, args.wd_project, group_name + '_' + setting, f'seed-{args.seed}' + f'-itr-{ii}', f'{current_time}_{random_string}')
    args.log_path = os.path.join(args.it_fname, 'log')
    args.pth_path = os.path.join(args.it_fname, 'pth')
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.pth_path, exist_ok=True)
    if args.freq == 0:
        args.freq = 'h'
    train_data, train_loader = data_provider(args, 'train', train_shuffle=args.train_shuffle)
    vali_data, vali_loader = data_provider(args, 'val')
    if is_zero_shot:
        # For zero-shot, switch to test dataset
        args.root_path = args.test_root_path
        args.data_path = args.test_data_path
        args.data = args.test_data
        test_data, test_loader = data_provider(args, 'test')

        # Handle frequency for zero-shot
        if args.freq != 'h':
            args.freq = SEASONALITY_MAP[test_data.freq]
            print("freq = {}".format(args.freq))
    else:
        test_data, test_loader = data_provider(args, 'test')

    device = torch.device('cuda:0')

    train_steps = len(train_loader)

    model_class = globals()[args.model]
    model = model_class(args, device)
    model.to(device)

    params = model.parameters()
    model_optim = torch.optim.Adam(params, lr=args.learning_rate)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    criterion = nn.MSELoss()
    
    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=args.eta_min)
    elif args.lradj == 'TST':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)
    
    for epoch in range(args.train_epochs):
        
        train_loss = []
        model.train()
        epoch_start_time = time.time()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            if args.test_flag == 1 and i > 3:
                break
                
            model_optim.zero_grad()

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            outputs = model(batch_x, ii)
            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())
            loss.backward()
            model_optim.step()

        log_epoch_time = time.time() - epoch_start_time
        log = "Epoch: {} cost time: {}".format(epoch + 1, log_epoch_time)
        print(log)
        with open(os.path.join(args.log_path, 'log.txt'), 'a') as f:
            f.write(log)
            f.write('\n')
            
        train_loss = np.average(train_loss)
        vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
            
        log = "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss)
        print(log)
        with open(os.path.join(args.log_path, 'log.txt'), 'a') as f:
            f.write(log)
            f.write('\n')

        if args.lradj in ['COS', 'TST']:
            scheduler.step()
            log = "lr = {:.10f}".format(model_optim.param_groups[0]['lr'])
            with open(os.path.join(args.log_path, 'log.txt'), 'a') as f:
                f.write(log)
                f.write('\n')

        else:
            adjust_learning_rate(model_optim, epoch + 1, args)
            
        early_stopping(vali_loss, model, args.pth_path)
        if early_stopping.early_stop:
            print("Early stopping")
            with open(os.path.join(args.log_path, 'log.txt'), 'a') as f:
                f.write("Early stopping")
                f.write('\n')
            break

    best_model_path = args.pth_path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))
    print("------------------------------------")
    print("Starting testing phase...")
    
    preds = []
    trues = []
    
    model.eval()
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(test_loader), desc="Testing"):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            outputs = model(batch_x, ii)

            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)

            pred = outputs.detach().cpu().numpy()
            true = batch_y.detach().cpu().numpy()
            
            preds.append(pred)
            trues.append(true)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    mse = np.mean((preds - trues) ** 2)
    mae = np.mean(np.abs(preds - trues))
    
    print(f"Test MSE: {mse:.4f}, MAE: {mae:.4f}")
    
    log = "mse = {:.4f}".format(mse)
    with open(os.path.join(args.log_path, 'log.txt'), 'a') as f:
        f.write(log)
        f.write('\n')
    log = "mae = {:.4f}".format(mae)
    with open(os.path.join(args.log_path, 'log.txt'), 'a') as f:
        f.write(log)
        f.write('\n')