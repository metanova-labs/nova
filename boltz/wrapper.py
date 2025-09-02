import os
import yaml
import sys
import traceback
import json
import numpy as np
import random
import gc
import shutil

import bittensor as bt
import torch

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(os.path.join(BASE_DIR, ".."))
sys.path.append(BASE_DIR)

from src.boltz.main import predict
from utils.proteins import get_sequence_from_protein_code
from utils.molecules import compute_maccs_entropy

class BoltzWrapper:
    def __init__(self):
        config_path = os.path.join(BASE_DIR, "boltz_config.yaml")
        self.config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
        self.base_dir = BASE_DIR

        self.tmp_dir = os.path.join(PARENT_DIR, "boltz_tmp_files")
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.input_dir = os.path.join(self.tmp_dir, "inputs")
        os.makedirs(self.input_dir, exist_ok=True)

        self.output_dir = os.path.join(self.tmp_dir, "outputs")
        os.makedirs(self.output_dir, exist_ok=True)

        bt.logging.debug(f"BoltzWrapper initialized")

    def preprocess_data_for_boltz(self, valid_molecules_by_uid: dict, score_dict: dict, final_block_hash: str) -> None:
        # Get protein sequence
        self.protein_sequence = get_sequence_from_protein_code(self.subnet_config['weekly_target'])

        # Collect all unique molecules across all UIDs
        self.unique_molecules = {}  # {smiles: [(uid, mol_idx), ...]}
        
        bt.logging.info("Preprocessing data for Boltz2")
        for uid, valid_molecules in valid_molecules_by_uid.items():
            # Select a subsample of n molecules to score
            if self.subnet_config['sample_selection'] == "random":
                seed = int(final_block_hash[2:], 16) + uid
                rng = random.Random(seed)

                unique_indices = rng.sample(range(len(valid_molecules['smiles'])), 
                                            k=self.subnet_config['num_molecules_boltz'])

                boltz_candidates_smiles = [valid_molecules['smiles'][i] for i in unique_indices]
            elif self.subnet_config['sample_selection'] == "first":
                boltz_candidates_smiles = valid_molecules['smiles'][:self.subnet_config['num_molecules_boltz']]
            else:
                bt.logging.error(f"Invalid sample selection method: {self.subnet_config['sample_selection']}")
                raise ValueError(f"Invalid sample selection method: {self.subnet_config['sample_selection']}")

            if self.subnet_config['num_molecules_boltz'] > 1:
                try:
                    score_dict[uid]["entropy_boltz"] = compute_maccs_entropy(boltz_candidates_smiles)
                except Exception as e:
                    bt.logging.error(f"Error computing Boltz subset entropy for UID={uid}: {e}")
                    score_dict[uid]["entropy_boltz"] = None
            else:
                score_dict[uid]["entropy_boltz"] = None

            for smiles in boltz_candidates_smiles:
                if smiles not in self.unique_molecules:
                    self.unique_molecules[smiles] = []
                mol_idx = list(self.unique_molecules.keys()).index(smiles)
                self.unique_molecules[smiles].append((uid, mol_idx))
        bt.logging.info(f"Unique Boltz candidates: {self.unique_molecules}")

        # Write unique molecules to input directory
        bt.logging.info(f"Writing {len(self.unique_molecules)} unique molecules to input directory")
        for smiles, ids in self.unique_molecules.items():
            yaml_content = self.create_yaml_content(smiles)
            with open(os.path.join(self.input_dir, f"{ids[0][1]}.yaml"), "w") as f:
                f.write(yaml_content)

        bt.logging.debug(f"Preprocessing data for Boltz2 complete")
            
    def create_yaml_content(self, ligand_smiles: str) -> str:
        """Create YAML content for Boltz2 prediction with no MSA"""

        yaml_content = f"""version: 1
sequences:
    - protein:
        id: A
        sequence: {self.protein_sequence}
        msa: empty
    - ligand:
        id: B
        smiles: '{ligand_smiles}'
        """

        if self.subnet_config['binding_pocket'] is not None:
            yaml_content += f"""
constraints:
    - pocket:
        binder: B
        contacts: {self.subnet_config['binding_pocket']}
        max_distance: {self.subnet_config['max_distance']}
        force: {self.subnet_config['force']}
        """

        yaml_content += f"""
properties:
    - affinity:
        binder: B
        """
        
        return yaml_content

    def score_molecules_target(self, valid_molecules_by_uid: dict, score_dict: dict, subnet_config: dict, final_block_hash: str) -> None:
        # Preprocess data
        self.subnet_config = subnet_config

        self.preprocess_data_for_boltz(valid_molecules_by_uid, score_dict, final_block_hash)

        # Run Boltz2 for unique molecules
        bt.logging.info("Running Boltz2")
        try:
            predict(
                data = self.input_dir,
                out_dir = self.output_dir,
                recycling_steps = self.config['recycling_steps'],
                sampling_steps = self.config['sampling_steps'],
                diffusion_samples = self.config['diffusion_samples'],
                sampling_steps_affinity = self.config['sampling_steps_affinity'],
                diffusion_samples_affinity = self.config['diffusion_samples_affinity'],
                output_format = self.config['output_format'],
                seed = 68,
                affinity_mw_correction = self.config['affinity_mw_correction'],
                no_kernels = self.config['no_kernels'],
            )
            bt.logging.info(f"Boltz2 predictions complete")

        except Exception as e:
            bt.logging.error(f"Error running Boltz2: {e}")
            bt.logging.error(traceback.format_exc())
            return None

        # Collect scores and distribute results to all UIDs
        self.postprocess_data(score_dict)

    def postprocess_data(self, score_dict: dict) -> None:
        # Collect scores - Results need to be saved to disk because of distributed predictions
        scores = {}
        for mol_idx, smiles in enumerate(self.unique_molecules.keys()):
            results_path = os.path.join(self.output_dir, 'boltz_results_inputs', 'predictions', f'{mol_idx}')
            if mol_idx not in scores:
                scores[mol_idx] = {}
            for filepath in os.listdir(results_path):
                if filepath.startswith('affinity'):
                    with open(os.path.join(results_path, filepath), 'r') as f:
                        affinity_data = json.load(f)
                    scores[mol_idx].update(affinity_data)
                elif filepath.startswith('confidence'):
                    with open(os.path.join(results_path, filepath), 'r') as f:
                        confidence_data = json.load(f)
                    scores[mol_idx].update(confidence_data)
        #bt.logging.debug(f"Collected scores: {scores}")

        if self.config['remove_files']:
            bt.logging.info("Removing files")
            os.system(f"rm -r {os.path.join(self.output_dir, 'boltz_results_inputs')}")
            os.system(f"rm {self.input_dir}/*.yaml")
            bt.logging.info("Files removed")

        # Distribute results to all UIDs
        final_boltz_scores = {}
        for smiles, id_list in self.unique_molecules.items():
            for uid, mol_idx in id_list:
                if uid not in final_boltz_scores:
                    final_boltz_scores[uid] = []
                final_boltz_scores[uid].append(scores[mol_idx][self.subnet_config['boltz_metric']])
        bt.logging.debug(f"final_boltz_scores: {final_boltz_scores}")

        for uid, data in score_dict.items():
            if uid in final_boltz_scores:
                data['boltz_score'] = np.mean(final_boltz_scores[uid])
            else:
                data['boltz_score'] = None
    
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
        # Clean up temporary files
        if hasattr(self, 'tmp_dir') and self.tmp_dir and os.path.exists(self.tmp_dir):
            try:
                shutil.rmtree(self.tmp_dir)
                bt.logging.info(f"Cleaned up Boltz temporary directory: {self.tmp_dir}")
            except Exception as e:
                bt.logging.warning(f"Could not clean up Boltz temp directory: {e}")
        
        # Clear any model-specific attributes
        if hasattr(self, 'unique_molecules'):
            del self.unique_molecules
            self.unique_molecules = None
        if hasattr(self, 'protein_sequence'):
            del self.protein_sequence
            self.protein_sequence = None
            
        self.clear_gpu_memory()
        