from .deterministic_init import setup_deterministic

import os
import json
import gc

import pandas as pd
import torch

from .psichic_utils.dataset import ProteinMoleculeDataset
from .psichic_utils.data_utils import DataLoader, virtual_screening
from .psichic_utils import protein_init, ligand_init
from .models.net import net

from runtime_config import RuntimeConfig

class PsichicWrapper:
    def __init__(self):
        self.runtime_config = RuntimeConfig()
        self.device = self.runtime_config.DEVICE
        
        with open(os.path.join(self.runtime_config.MODEL_PATH, 'config.json'), 'r') as f:
            self.model_config = json.load(f)
            
    def load_model(self):
        degree_dict = torch.load(os.path.join(self.runtime_config.MODEL_PATH,
                                              'degree.pt'), 
                                 weights_only=True
                                 )
        param_dict = os.path.join(self.runtime_config.MODEL_PATH, 'model.pt')
        mol_deg, prot_deg = degree_dict['ligand_deg'], degree_dict['protein_deg']
        
        self.model = net(mol_deg, prot_deg,
                         # MOLECULE
                         mol_in_channels=self.model_config['params']['mol_in_channels'],  
                         prot_in_channels=self.model_config['params']['prot_in_channels'], 
                         prot_evo_channels=self.model_config['params']['prot_evo_channels'],
                         hidden_channels=self.model_config['params']['hidden_channels'], 
                         pre_layers=self.model_config['params']['pre_layers'], 
                         post_layers=self.model_config['params']['post_layers'],
                         aggregators=self.model_config['params']['aggregators'], 
                         scalers=self.model_config['params']['scalers'],
                         total_layer=self.model_config['params']['total_layer'],                
                         K=self.model_config['params']['K'],
                         heads=self.model_config['params']['heads'], 
                         dropout=self.model_config['params']['dropout'],
                         dropout_attn_score=self.model_config['params']['dropout_attn_score'],
                         # output
                         regression_head=self.model_config['tasks']['regression_task'],
                         classification_head=self.model_config['tasks']['classification_task'] ,
                         multiclassification_head=self.model_config['tasks']['mclassification_task'],
                         device=self.device).to(self.device)
        self.model.reset_parameters()    
        self.model.load_state_dict(torch.load(param_dict, 
                                              map_location=self.device, 
                                              weights_only=True
                                              )
                                   )
        self.model.eval()
        
    def initialize_protein(self, protein_seq:str) -> dict:
        allowed_chars = set(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X'])
        sanitized_protein_seq = ''.join([aa if aa in allowed_chars else 'X' for aa in protein_seq])
        self.protein_seq = [sanitized_protein_seq]

        protein_dict = protein_init(self.protein_seq)
        return protein_dict
    
    def initialize_smiles(self, smiles_list:list) -> dict:
        self.smiles_list = smiles_list
        smiles_dict = ligand_init(smiles_list)
        return smiles_dict
    
    def create_screen_loader(self, protein_dict, smiles_dict):
        self.screen_df = pd.DataFrame({'Protein': [k for k in self.protein_seq for _ in self.smiles_list],
                                       'Ligand': [l for l in self.smiles_list for _ in self.protein_seq],
                                       })
        
        dataset = ProteinMoleculeDataset(self.screen_df, 
                                         smiles_dict, 
                                         protein_dict, 
                                         device=self.device
                                         )
        
        self.screen_loader = DataLoader(dataset,
                                        batch_size=self.runtime_config.BATCH_SIZE,
                                        shuffle=False,
                                        follow_batch=['mol_x', 'clique_x', 'prot_node_aa']
                                        )
        
    def run_challenge_start(self, protein_seq:str):
        torch.cuda.empty_cache()
        self.load_model()
        self.protein_dict = self.initialize_protein(protein_seq)
        
    def run_validation(self, smiles_list:list) -> pd.DataFrame:
        self.smiles_dict = self.initialize_smiles(smiles_list)
        torch.cuda.empty_cache()
        self.create_screen_loader(self.protein_dict, self.smiles_dict)
        self.screen_df = virtual_screening(self.screen_df, 
                                           self.model, 
                                           self.screen_loader,
                                           os.getcwd(),
                                           save_interpret=False,
                                           ligand_dict=self.smiles_dict, 
                                           device=self.device,
                                           save_cluster=False,
                                           )
        return self.screen_df
    
    def clear_gpu_memory(self):
        """Clear GPU memory and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Reset CUDA context to allow other processes to initialize
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
    
    def cleanup_model(self):
        """Clean up model and free GPU memory."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, 'screen_loader'):
            del self.screen_loader
            self.screen_loader = None
        if hasattr(self, 'screen_df'):
            del self.screen_df
            self.screen_df = None
        self.clear_gpu_memory()
        
