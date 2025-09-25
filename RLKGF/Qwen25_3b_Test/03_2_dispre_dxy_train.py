import sys
import os

print("Current working dir:", os.getcwd())
print("Script directory:", os.path.dirname(os.path.abspath(__file__)))
print("sys.path before append:", sys.path)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("sys.path after append:", sys.path)

import argparse
import ast
import gc

import torch
import random
import json
import time


from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.utils.data import DataLoader
#from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from utils.utils import *
from Data.DXY.KG import KGADJ
from utils import dialog_config

from DisPre_RWR_And_GCN.GCN import GCNReward
from DisPre_RWR_And_GCN.RWRModel import RWR
from DisPre_RWR_And_GCN.KGReward import KGReward
from DisPre_RWR_And_GCN.dataset import DialogueDataset, custom_collate_fn
from DisPre_RWR_And_GCN.qwen_ppo import PPO_KG

def set_seed(seed):
    """
    Set random seeds to ensure reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True  # For deterministic behavior in CUDA
    torch.backends.cudnn.benchmark = False  # Disable the auto-tuner that picks the best algorithm

def GetAllSym(goal_set):
    '''
Output form:
    new_goal_test = {
    'train': [
        {
            'current_slots': {
                'inform_slots': {
                    'Cough': dialog_config.TRUE,
                    'Chest tightness and shortness of breath': dialog_config.FALSE,
                    'Post sternum pain': dialog_config.FALSE,
                    'Chest tightness': dialog_config.FALSE,
                    'Hemoptysis': dialog_config.TRUE,
                    'Coughing sputum': dialog_config.TRUE
                }
            },
            'disease_tag': 'esophagealitis'
        }
...
    ],
    'test': [...]
    }
    '''
    new_goal_test = {'train': [], 'test': []}
    for g in goal_set['train']:
        g_ = {'current_slots': {'inform_slots': {}}, 'disease_tag': g['disease_tag']}
        for ex in g['explicit_inform_slots'].keys():
            if g['explicit_inform_slots'][ex]:
                g_['current_slots']['inform_slots'][ex] = dialog_config.TRUE
            elif not g['explicit_inform_slots'][ex]:
                g_['current_slots']['inform_slots'][ex] = dialog_config.FALSE

        for im in g['implicit_inform_slots'].keys():
            if g['implicit_inform_slots'][im] == True:
                g_['current_slots']['inform_slots'][im] = dialog_config.TRUE
            elif g['implicit_inform_slots'][im] == False:
                g_['current_slots']['inform_slots'][im] = dialog_config.FALSE

        new_goal_test['train'].append(g_)

    for g in goal_set['test']:
        g_ = {'current_slots': {'inform_slots': {}}, 'disease_tag': g['disease_tag']}
        for ex in g['explicit_inform_slots'].keys():
            if g['explicit_inform_slots'][ex]:
                g_['current_slots']['inform_slots'][ex] = dialog_config.TRUE
            elif not g['explicit_inform_slots'][ex]:
                g_['current_slots']['inform_slots'][ex] = dialog_config.FALSE

        for im in g['implicit_inform_slots'].keys():
            if g['implicit_inform_slots'][im] == True:
                g_['current_slots']['inform_slots'][im] = dialog_config.TRUE
            elif g['implicit_inform_slots'][im] == False:
                g_['current_slots']['inform_slots'][im] = dialog_config.FALSE

        new_goal_test['test'].append(g_)

    return new_goal_test

def log_print(message: str, f_log=None):
    print(message, flush=True)
    if f_log:
        f_log.write(message + "\n")
        f_log.flush()

timeStr = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))




# data
embed_size = 100
current_path = os.path.abspath(__file__)
print(current_path)
father_path = os.path.abspath(os.path.dirname(current_path))
print(father_path)
grand_path = os.path.abspath(os.path.dirname(father_path))
print(grand_path)



log_filename = os.path.join(father_path, "dxy_dispre_model_save", f"training_{timeStr}.log")
os.makedirs(os.path.dirname(log_filename), exist_ok=True)
f_log = open(log_filename, "a")  # append mode to keep old logs if any
# Redirect standard output to log file
sys.stdout = f_log

DXY_path = os.path.join(grand_path, 'Data', 'DXY', 'dataset_dxy')
print(DXY_path)
DXY_Disease = text_to_list(os.path.join(DXY_path, 'diseases_dxy.txt'))
print(DXY_Disease)
# print(DXY_Disease)
DXY_Symptom = text_to_list(os.path.join(DXY_path, 'symptoms_dxy.txt'))

DXY_goal = load_pickle(os.path.join(DXY_path, 'goal_dict_original_dxy.p'))

dis_sym_num_to_graph_path = os.path.join(DXY_path, 'dise_sym_num_dict_dxy.txt')
# dis_sym_num_to_graph
with open(dis_sym_num_to_graph_path, 'r', encoding='utf-8') as f:
    content = f.readlines()
    dis_sym_num_to_graph = ast.literal_eval(content[0])

gcn_path = os.path.join(grand_path, 'useful_models', 'DXY_GCN', 'gcn_0.6675.pth.tar')
slot_set = text_to_dict('{}/slot_set_dxy.txt'.format(DXY_path))  # all slots with symptoms + all disease

# Set seeds
set_seed(1616)
# device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# KG
kg = KGADJ(device, len(DXY_Disease), len(DXY_Symptom), DXY_path)

gcn_tem = 1
rwr_tem = 1
mu = 0.1
# Reward Model
gcn_reward = GCNReward(device, len(DXY_Disease) + len(DXY_Symptom), len(DXY_Disease), kg, slot_set, embed_size, gcn_tem)  # device, kg_node, dis_num, kggraph, slot_set, embed_size=100, temperature=1
gcn_ = torch.load(gcn_path, map_location=device)
gcn_reward.load_state_dict(gcn_['state_dict'])  # Load offline training model
# alpha
alpha = 0.3
# RWR Reward Model
rwr_reward = RWR(len(DXY_Disease), len(DXY_Symptom), DXY_Disease, DXY_Symptom, slot_set, dis_sym_num_to_graph, kg, device, alpha=alpha, temperature=rwr_tem)
kg_reward = KGReward(gcn_reward, rwr_reward, mu=mu)

# data
goal_ = GetAllSym(DXY_goal)
data_type = 'train'  # Training data
dataset = DialogueDataset(goal_, data_type)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
batch_size_ = 8

# Define PPO
# for i in [40 - 5 * j for j in range(9)] + [45]:
for i in [45 - 5 * j for j in range(2)]:
    # if i == 30:
    #     l_= [3e-5]
    # else:
    #     l_= [1e-5, 2e-5, 3e-5]
    #for l in [1e-5, 2e-5, 6e-6, 9e-6]:
    for l in [1e-5]:
        # LLM
        # Loading the model
        # Loading models and word participlers
        llm_path = "Qwen/Qwen2.5-VL-3B-Instruct"  # LLM Mutual information and final generation

        # model = AutoModelForCausalLM.from_pretrained(
        #     llm_path,
        #     torch_dtype="auto",
        #     device_map=device
        # )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            llm_path,
            torch_dtype="auto",
            device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained(llm_path, padding_side='left')
        update_freq = i
        lr = l
        dispre_ppo_kg = PPO_KG(model, kg_reward, tokenizer, DXY_Symptom, DXY_Disease, device, lr=lr,
                               update_freq=update_freq)

        # Timestamp
        timeStr = time.strftime('%Y.%m.%d-%H-%M-%S', time.localtime(time.time()))
        log_dir = os.path.join(father_path, "runs", f"freq_{update_freq}_lr_{lr}_{timeStr}")
        writer = SummaryWriter(log_dir=log_dir)

        log_print('========== FREQ, LR:==============', f_log)
        log_print(f' FREQ {dispre_ppo_kg.update_freq} ', f_log)
        log_print(f' LR {dispre_ppo_kg.lr} ', f_log)

        success_rate = 0.
        train_losses = []
        average_loss = [10000.]
        val_accuracies = []

        log_print('==========Initial Test==============', f_log)
        initial_test_rate = dispre_ppo_kg.eval_step(goal_['test'])
        for epoch in range(2): # Simplified training cycle 5
            log_print(f'==========epoch: {epoch+1} train==============', f_log)
            for batch in dataloader:
                goals = batch
                # Perform one-step training
                dispre_ppo_kg.train_step(epoch, timeStr, goals)

            # Average loss per epoch print
            avg_loss = np.mean(dispre_ppo_kg.losses[-len(dataloader):])  # Calculate the average loss of the current epoch

            writer.add_scalar('Loss/train', avg_loss, epoch + 1)

            log_print(f"Epoch {epoch + 1}: Average PPO Loss = {avg_loss:.4f}", f_log)
            average_loss.append(avg_loss)

            # # verify
            log_print(f'==========epoch: {epoch+1} test==============', f_log)
            test_rate = dispre_ppo_kg.eval_step(goal_['test'])

            # Log validation accuracy per epoch
            writer.add_scalar('Accuracy/val', test_rate, epoch + 1)

            val_accuracies.append(test_rate)
            if test_rate > success_rate or avg_loss < average_loss[-2] or test_rate > initial_test_rate:
                success_rate = test_rate
                # Create save directory based on accuracy, add accuracy to folder name using string formatting
                # model_directory = os.path.join(father_path, "dxy_dispre_model_save", str(timeStr),
                #                                f"model_directory_epoch{epoch}_accuracy_{test_rate:.4f}_loss_{avg_loss:.4f}")

                # Overwrite same folder each time
                model_directory = os.path.join(father_path, "dxy_dispre_model_save", "best_model")

                # Ensure the directory exists
                os.makedirs(model_directory, exist_ok=True)

                # Save the model and tokenizer
                model.save_pretrained(model_directory)
                tokenizer.save_pretrained(model_directory)
                log_print(f"Model saved at {model_directory}", f_log)

            # metrics = {
            #     "initial_test": initial_test_rate,
            #     "update_freq": update_freq,
            #     "lr": lr,
            #     "batch": batch_size_,
            #     "seed": 1616,
            #     "gcn_tem": gcn_tem,
            #     "rwr_tem": rwr_tem,
            #     "mu": mu,
            #     "gcn": gcn_path,
            #     "train_losses": average_loss,
            #     "val_accuracies": val_accuracies
            # }
            # metrics_path = os.path.join(father_path, "dxy_dispre_model_save", str(timeStr), 'training_metrics.json')
            # os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

            # with open(metrics_path, 'w') as f:
            #     json.dump(metrics, f, ensure_ascii=False, indent=4)

            # log_print(f"Saved training metrics to {metrics_path}", f_log)

            import json
            import os

            # Define this *inside the epoch loop*, after computing test_rate and avg_loss
            epoch_metrics = {
                "epoch": epoch,
                "initial_test": initial_test_rate,
                "update_freq": update_freq,
                "lr": lr,
                "batch": batch_size_,
                "seed": 1616,
                "gcn_tem": gcn_tem,
                "rwr_tem": rwr_tem,
                "mu": mu,
                "gcn": gcn_path,
                "train_loss": avg_loss,
                "val_accuracy": test_rate
            }

            metrics_path = os.path.join(father_path, "dxy_dispre_model_save", str(timeStr), 'training_metrics.json')
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

            # Load existing metrics if file exists
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = []

            # Append current epoch's metrics
            all_metrics.append(epoch_metrics)

            # Save the updated list back to the file
            with open(metrics_path, 'w') as f:
                json.dump(all_metrics, f, ensure_ascii=False, indent=4)

            log_print(f"Appended training metrics to {metrics_path}", f_log)
            writer.flush()
            time.sleep(10)

        # === FREE UP MEMORY ===
        del model
        del tokenizer
        del dispre_ppo_kg

        torch.cuda.empty_cache()
        gc.collect()

# model_directory = os.path.join(father_path, "dxy_dispre_model_save", str(timeStr),f"model_directory_epoch{epoch}_accuracy_{test_rate:.4f}_loss_{avg_loss:.4f}")

# # Save the model and tokenizer
# model.save_pretrained(model_directory)
# tokenizer.save_pretrained(model_directory)
# log_print(f"Model saved at {model_directory}", f_log)

f_log.close()
writer.close()