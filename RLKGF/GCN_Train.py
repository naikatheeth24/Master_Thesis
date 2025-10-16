# import sys
# import os
# import ast # Assuming you need these from your original code
# # Helper functions from your original code (placeholders)
# def text_to_list(path):
#     with open(path, 'r') as f:
#         return [line.strip() for line in f.readlines()]

# def load_pickle(path):
#     import pickle
#     with open(path, 'rb') as f:
#         return pickle.load(f)

# def text_to_dict(path):
#     with open(path, 'r') as f:
#         # This is just a guess at the function's logic
#         return {line.strip(): i for i, line in enumerate(f.readlines())}


# from Qwen25_3b_Test.DisPre_RWR_And_GCN.GCN import GCNReward

# # --- Start of the corrected logic ---

# # Get the current working directory instead of the script directory
# base_path = os.getcwd() 
# print("Current working dir:", base_path)

# # If your code is in a subdirectory (e.g., 'scripts') of your project root,
# project_root = os.path.abspath(os.path.join(base_path, '..'))
# print("project_root",project_root)

# # This replaces `sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))`
# sys.path.append(project_root)
# print("sys.path after append:", sys.path)

# grand_path = base_path # Assuming grand_path is the project root

# DXY_path = os.path.join(grand_path, 'Data', 'DXY', 'dataset_dxy')
# print("DXY_path",DXY_path)

# DXY_Disease = text_to_list(os.path.join(DXY_path, 'omega_problems.txt')) # problems
# print("DXY_Disease",DXY_Disease)

# DXY_Symptom = text_to_list(os.path.join(DXY_path, 'omega_symptoms.txt'))
# print("DXY_Symptom",DXY_Symptom)

# DXY_goal = load_pickle(os.path.join(DXY_path, 'omega_goals.p'))
# print("DXY_goal",DXY_goal)

# dis_sym_num_to_graph_path = os.path.join(DXY_path, 'prob_slot_num_dict_maintenance_weighted.txt')
# print("dis_sym_num_to_graph_path",dis_sym_num_to_graph_path)

# with open(dis_sym_num_to_graph_path, 'r', encoding='utf-8') as f:
#     content = f.readlines()
#     dis_sym_num_to_graph = ast.literal_eval(content[0])

# #gcn_path = os.path.join(grand_path, 'useful_models', 'DXY_GCN', 'gcn_0.6675.pth.tar')
# slot_set = text_to_dict('{}/omega_slot.txt'.format(DXY_path))
# print("slot_set",slot_set)

# print("\nSuccessfully defined paths.")

# from utils.utils import *
# from Data.DXY.KG import KGADJ
# import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# kgraph = KGADJ(device, len(DXY_Disease), len(DXY_Symptom), DXY_path)

# #kgraph = KGADJ(device, dis_num, sym_num, data_path)
# # kgraph.initialize_adj()
# # adj_matrix = kgraph.kg_matrix  # tensor of shape [kg_node, kg_node]

# kgraph.initialize_adj()
# adj_matrix = kgraph.kg_matrix  # tensor of shape [kg_node, kg_node]


# model = GCNReward(
#     device=device,
#     kg_node=len(DXY_Disease) + len(DXY_Symptom),
#     dis_num=len(DXY_Disease),
#     kggraph=kgraph,
#     slot_set=slot_set,  # mapping from slot name to index
#     embed_size=100,
#     temperature=0.5
# )

# def GetAllSym(goal_set):
#     new_goal_test = {'train': [], 'test': []}
#     for g in goal_set['train']:
#         g_ = {'current_slots': {'inform_slots': {}}, 'problem_tag': g['problem_tag']}
#         for ex in g['explicit_inform_slots']:
#             g_['current_slots']['inform_slots'][ex] = dialog_config.TRUE if g['explicit_inform_slots'][ex] else dialog_config.FALSE
#         for im in g['implicit_inform_slots']:
#             g_['current_slots']['inform_slots'][im] = dialog_config.TRUE if g['implicit_inform_slots'][im] else dialog_config.FALSE
#         new_goal_test['train'].append(g_)

#     for g in goal_set['test']:
#         g_ = {'current_slots': {'inform_slots': {}}, 'problem_tag': g['problem_tag']}
#         for ex in g['explicit_inform_slots']:
#             g_['current_slots']['inform_slots'][ex] = dialog_config.TRUE if g['explicit_inform_slots'][ex] else dialog_config.FALSE
#         for im in g['implicit_inform_slots']:
#             g_['current_slots']['inform_slots'][im] = dialog_config.TRUE if g['implicit_inform_slots'][im] else dialog_config.FALSE
#         new_goal_test['test'].append(g_)

#     return new_goal_test


# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# from utils import dialog_config

# import pickle

# with open('omega_goals.p', 'rb') as f:
#     data = pickle.load(f)

# goal_ = GetAllSym(data)   # converts to format with 'current_slots'
# train_data = goal_['train']


# disease2id = {disease: slot_set[disease] for disease in DXY_Disease}

# print(train_data[0])

# # train_data = data['train']
# kg_node = len(DXY_Disease) + len(DXY_Symptom)

# for epoch in range(5):
#     total_loss = 0
#     total_correct = 0

#     for sample in train_data:
#         model.train()

#         label = torch.tensor([disease2id[sample['problem_tag']]]).to(device)
#         state_input = model.state_representation(sample)

#         model.kgraph.initialize_adj()
#         edge_index = model.construct_kg_index(model.kgraph.kg_matrix.view(1, kg_node, kg_node))

#         optimizer.zero_grad()
#         correct, loss = model.forward(label, edge_index, state_input)

#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         total_correct += correct

#     print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}, Accuracy = {total_correct}/{len(train_data)}")



# import sys
# import os
# import ast
# import torch
# import pickle

# from Qwen25_3b_Test.DisPre_RWR_And_GCN.GCN import GCNReward
# from utils.utils import *
# from Data.DXY.KG import KGADJ
# from utils import dialog_config

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Helper functions
# def text_to_list(path):
#     with open(path, 'r') as f:
#         return [line.strip() for line in f.readlines()]

# def load_pickle(path):
#     with open(path, 'rb') as f:
#         return pickle.load(f)

# def text_to_dict(path):
#     with open(path, 'r') as f:
#         return {line.strip(): i for i, line in enumerate(f.readlines())}

# # --- Setup paths and sys.path ---
# base_path = os.getcwd()
# project_root = os.path.abspath(os.path.join(base_path, '..'))
# sys.path.append(project_root)
# grand_path = base_path
# DXY_path = os.path.join(grand_path, 'Data', 'DXY', 'dataset_dxy')
# DXY_Disease = text_to_list(os.path.join(DXY_path, 'omega_problems.txt'))
# DXY_Symptom = text_to_list(os.path.join(DXY_path, 'omega_symptoms.txt'))
# DXY_goal = load_pickle(os.path.join(DXY_path, 'omega_goals.p'))
# dis_sym_num_to_graph_path = os.path.join(DXY_path, 'prob_slot_num_dict_maintenance_weighted.txt')

# with open(dis_sym_num_to_graph_path, 'r', encoding='utf-8') as f:
#     content = f.readlines()
#     dis_sym_num_to_graph = ast.literal_eval(content[0])

# slot_set = text_to_dict(os.path.join(DXY_path, 'omega_slot.txt'))

# # Imports for model and utils

# kgraph = KGADJ(device, len(DXY_Disease), len(DXY_Symptom), DXY_path)
# kgraph.initialize_adj()
# adj_matrix = kgraph.kg_matrix

# model = GCNReward(
#     device=device,
#     kg_node=len(DXY_Disease) + len(DXY_Symptom),
#     dis_num=len(DXY_Disease),
#     kggraph=kgraph,
#     slot_set=slot_set,
#     embed_size=1024,
#     temperature=0.5
# ).to(device)  # make sure model is on device

# def GetAllSym(goal_set):
#     new_goal_test = {'train': [], 'test': []}
#     for g in goal_set['train']:
#         g_ = {'current_slots': {'inform_slots': {}}, 'problem_tag': g['problem_tag']}
#         for ex in g['explicit_inform_slots']:
#             g_['current_slots']['inform_slots'][ex] = dialog_config.TRUE if g['explicit_inform_slots'][ex] else dialog_config.FALSE
#         for im in g['implicit_inform_slots']:
#             g_['current_slots']['inform_slots'][im] = dialog_config.TRUE if g['implicit_inform_slots'][im] else dialog_config.FALSE
#         new_goal_test['train'].append(g_)

#     for g in goal_set['test']:
#         g_ = {'current_slots': {'inform_slots': {}}, 'problem_tag': g['problem_tag']}
#         for ex in g['explicit_inform_slots']:
#             g_['current_slots']['inform_slots'][ex] = dialog_config.TRUE if g['explicit_inform_slots'][ex] else dialog_config.FALSE
#         for im in g['implicit_inform_slots']:
#             g_['current_slots']['inform_slots'][im] = dialog_config.TRUE if g['implicit_inform_slots'][im] else dialog_config.FALSE
#         new_goal_test['test'].append(g_)

#     return new_goal_test

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# with open('omega_goals.p', 'rb') as f:
#     data = pickle.load(f)

# goal_ = GetAllSym(data)
# train_data = goal_['train']

# disease2id = {disease: slot_set[disease] for disease in DXY_Disease}

# kg_node = len(DXY_Disease) + len(DXY_Symptom)

# # Training with batching
# batch_size = 128

# model.kgraph.initialize_adj()
# edge_index = model.construct_kg_index(model.kgraph.kg_matrix.view(1, kg_node, kg_node))

# for epoch in range(5):
#     total_loss = 0.0
#     total_correct = 0
#     model.train()

#     for i in range(0, len(train_data), batch_size):
#         batch = train_data[i:i + batch_size]

#         # Prepare batched labels and state inputs
#         labels = []
#         state_inputs = []
#         for sample in batch:
#             labels.append(disease2id[sample['problem_tag']])
#             state_inputs.append(model.state_representation(sample))

#         labels = torch.tensor(labels, dtype=torch.long).to(device)
#         state_inputs = torch.stack(state_inputs).to(device)  # assuming same shape per sample

#         # Initialize adjacency matrix once per batch

#         optimizer.zero_grad()
#         correct, loss = model.forward(labels, edge_index, state_inputs)

#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         total_correct += correct

#     print(f"Epoch {epoch + 1}: Loss = {total_loss:.4f}, Accuracy = {total_correct}/{len(train_data)}")



import sys
import os
import ast
import torch
import pickle
import time

from Qwen25_3b_Test.DisPre_RWR_And_GCN.GCN import GCNReward
from utils.utils import *
from Data.DXY.KG import KGADJ
from utils import dialog_config

start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Setup] Using device: {device}")

# Helper functions
def text_to_list(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def text_to_dict(path):
    with open(path, 'r') as f:
        return {line.strip(): i for i, line in enumerate(f.readlines())}

# --- Setup paths ---
base_path = os.getcwd()
project_root = os.path.abspath(os.path.join(base_path, '..'))
sys.path.append(project_root)
grand_path = base_path
DXY_path = os.path.join(grand_path, 'Data', 'DXY', 'dataset_dxy')

print(f"[Setup] Base path: {base_path}")
print(f"[Setup] Project root appended to sys.path: {project_root}")

# Load lists and dicts
DXY_Disease = text_to_list(os.path.join(DXY_path, 'omega_problems.txt'))
DXY_Symptom = text_to_list(os.path.join(DXY_path, 'omega_symptoms.txt'))
print(f"[Setup] Loaded Diseases ({len(DXY_Disease)}), Symptoms ({len(DXY_Symptom)})")

DXY_goal = load_pickle(os.path.join(DXY_path, 'omega_goals.p'))

with open(os.path.join(DXY_path, 'prob_slot_num_dict_maintenance_weighted.txt'), 'r', encoding='utf-8') as f:
    content = f.readlines()
    dis_sym_num_to_graph = ast.literal_eval(content[0])

slot_set = text_to_dict(os.path.join(DXY_path, 'omega_slot.txt'))
print(f"[Setup] Loaded slot_set with {len(slot_set)} items")

# Initialize KG graph
kgraph = KGADJ(device, len(DXY_Disease), len(DXY_Symptom), DXY_path)
kgraph.initialize_adj()
print(f"[Setup] Knowledge graph adjacency matrix initialized")

# # Initialize model
# model = GCNReward(
#     device=device,
#     kg_node=len(DXY_Disease) + len(DXY_Symptom),
#     dis_num=len(DXY_Disease),
#     kggraph=kgraph,
#     slot_set=slot_set,
#     embed_size=1024,
#     temperature=0.5
# ).to(device)

import os
import torch

# Set synchronous CUDA execution for better error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Clear any previous CUDA errors
torch.cuda.empty_cache()
torch.cuda.synchronize()

print("[Setup] CUDA_LAUNCH_BLOCKING=1 enabled for better error reporting")
print("[Setup] CUDA cache cleared")

# Reinitialize the model to clear any corrupted state
model = GCNReward(
    device=device,
    kg_node=len(DXY_Disease) + len(DXY_Symptom),
    dis_num=len(DXY_Disease),
    kggraph=kgraph,
    slot_set=slot_set,
    embed_size=1024,
    temperature=0.5
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("[Setup] Model reinitialized")


def GetAllSym(goal_set):
    new_goal_test = {'train': [], 'test': []}
    for g in goal_set['train']:
        g_ = {'current_slots': {'inform_slots': {}}, 'problem_tag': g['problem_tag']}
        for ex in g['explicit_inform_slots']:
            g_['current_slots']['inform_slots'][ex] = dialog_config.TRUE if g['explicit_inform_slots'][ex] else dialog_config.FALSE
        for im in g['implicit_inform_slots']:
            g_['current_slots']['inform_slots'][im] = dialog_config.TRUE if g['implicit_inform_slots'][im] else dialog_config.FALSE
        new_goal_test['train'].append(g_)

    for g in goal_set['test']:
        g_ = {'current_slots': {'inform_slots': {}}, 'problem_tag': g['problem_tag']}
        for ex in g['explicit_inform_slots']:
            g_['current_slots']['inform_slots'][ex] = dialog_config.TRUE if g['explicit_inform_slots'][ex] else dialog_config.FALSE
        for im in g['implicit_inform_slots']:
            g_['current_slots']['inform_slots'][im] = dialog_config.TRUE if g['implicit_inform_slots'][im] else dialog_config.FALSE
        new_goal_test['test'].append(g_)

    return new_goal_test

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

goal_ = GetAllSym(DXY_goal)
train_data = goal_['train']

disease2id = {disease: slot_set[disease] for disease in DXY_Disease}

kg_node = len(DXY_Disease) + len(DXY_Symptom)

print(f"[Setup] Finished data preparation and model setup in {time.time() - start_time:.2f} seconds")

import time
print("[Precompute] Starting precomputation of state representations...")
start_precompute = time.time()

all_state_inputs = []
all_labels = []

for idx, sample in enumerate(train_data):
    if idx % 100 == 0 and idx > 0:
        print(f"[Precompute] Processed {idx}/{len(train_data)} samples")
    state_tensor = model.state_representation(sample)  # This should be CPU tensor
    all_state_inputs.append(state_tensor)
    all_labels.append(disease2id[sample['problem_tag']])

all_state_inputs = torch.stack(all_state_inputs).to(device)  # Move to GPU once
all_labels = torch.tensor(all_labels, dtype=torch.long).to(device)

print(f"[Precompute] Completed in {time.time() - start_precompute:.2f} seconds")
print(f"[Precompute] State inputs shape: {all_state_inputs.shape}")
print(f"[Precompute] Labels shape: {all_labels.shape}")


# Verify labels are still correct after reinitialization
print(f"[Verify] Label range: min={all_labels.min().item()}, max={all_labels.max().item()}")
print(f"[Verify] Expected range: [0, {len(DXY_Disease)-1}]")

# Now run the training loop again

print(f"[Setup] Model initialized with embed_size=1024")



import time

#print("[Training] Starting training...")
start_training = time.time()

batch_size = 1
num_samples = all_labels.size(0)

#print("Before model.kgraph.initialize_adj()")
model.kgraph.initialize_adj()
#print("After model.kgraph.initialize_adj()")

start = time.time()
#print("Before edge_index = model.construct_kg_index")
edge_index = model.construct_kg_index(model.kgraph.kg_matrix.view(1, kg_node, kg_node)).to(device)
#print(f"Constructing and moving edge_index took {time.time() - start:.4f} seconds")

for epoch in range(5):
    print(f"[Training] Epoch {epoch+1} started")
    total_loss = 0.0
    total_correct = 0
    model.train()
    
    #print("Starting batch loop")
    epoch_start = time.time()

    for i in range(0, num_samples, batch_size):
        batch_inputs = all_state_inputs[i:i + batch_size].squeeze(1)
        batch_labels = all_labels[i:i + batch_size]

        #print(f"  [Batch {i//batch_size}] Batch inputs shape: {batch_inputs.shape}, Batch labels shape: {batch_labels.shape}")

        batch_start = time.time()
        optimizer.zero_grad()

        #print(f"  [Batch {i//batch_size}] Starting forward pass...")
        correct, loss = model.forward(batch_labels, edge_index, batch_inputs)
        #print(f"  [Batch {i//batch_size}] Forward pass done in {time.time() - batch_start:.4f} seconds")

        print(f"  [Batch {i//batch_size}] Loss computed: {loss.item():.4f}")

        backward_start = time.time()
        loss.backward()
        optimizer.step()
        #print(f"  [Batch {i//batch_size}] Backward and optimizer step done in {time.time() - backward_start:.4f} seconds")

        total_loss += loss.item()
        total_correct += correct

        if i % (batch_size * 10) == 0:
            print(f"  [Batch {i//batch_size}] Cumulative Loss={total_loss:.4f}, Correct={total_correct}")

    print("Finished batch loop")
    print(f"[Training] Epoch {epoch+1} completed in {time.time() - epoch_start:.2f} seconds")
    print(f"[Training] Epoch {epoch+1}: Total Loss={total_loss:.4f}, Accuracy={total_correct}/{num_samples}")

print(f"[Training] Total training time: {time.time() - start_training:.2f} seconds")

