import os
import yaml
import sys
import traceback
import json
import numpy as np
import random
import hashlib
import math
import shutil
import glob

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch

NOVA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(NOVA_DIR)

import bittensor as bt

from boltz.src.boltz.main import predict
from utils.proteins import get_sequence_from_protein_code
from utils.molecules import compute_maccs_entropy, is_boltz_safe_smiles, get_heavy_atom_count

def _get_record_id(rec_id, base_seed):
    h = hashlib.sha256(str(rec_id).encode()).digest()
    return (int.from_bytes(h[:8], "little") ^ base_seed) % (2**31 - 1)

def _create_yaml_content(protein_sequence: str, ligand_smiles: str) -> str:
    """Create YAML content for Boltz2 prediction with no MSA"""

    yaml_content = f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {protein_sequence}
      msa: empty
  - ligand:
      id: B
      smiles: {ligand_smiles}
properties:
  - affinity:
      binder: B
"""
        
    return yaml_content

class BoltzWrapper:
    def __init__(self):
        config_path = os.path.join(NOVA_DIR, "config", "boltz_config.yaml")
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.base_dir = NOVA_DIR

        self.tmp_dir = os.path.join(NOVA_DIR, "boltz", "boltz_tmp_files")
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.input_dir = os.path.join(self.tmp_dir, "inputs")
        os.makedirs(self.input_dir, exist_ok=True)

        self.output_dir = os.path.join(self.tmp_dir, "outputs")
        os.makedirs(self.output_dir, exist_ok=True)

        bt.logging.info(f"BoltzWrapper initialized")
        self.per_molecule_components = {}
        
        self.base_seed = 68
        random.seed(self.base_seed)
        np.random.seed(self.base_seed)
        torch.manual_seed(self.base_seed)

    def _preprocess_data_for_boltz(self, valid_molecules_by_uid: dict, score_dict: dict) -> None:
        # Collect all unique molecules across all UIDs
        self.unique_molecules = {}  # {smiles: [(uid, mol_id), ...]}
        
        bt.logging.info("Preprocessing data for Boltz2")
        for uid, valid_molecules in valid_molecules_by_uid.items():
            for smiles in valid_molecules['smiles']:
                ok, reason = is_boltz_safe_smiles(smiles)
                if not ok:
                    bt.logging.warning(f"Skipping Boltz candidate {smiles} because it is not parseable: {reason}")
                    continue
                if smiles not in self.unique_molecules:
                    self.unique_molecules[smiles] = []
                mol_idx = _get_record_id(smiles, self.base_seed)

                self.unique_molecules[smiles].append((uid, mol_idx))
        bt.logging.info(f"Unique Boltz candidates: {self.unique_molecules}")

        bt.logging.info(f"Writing {len(self.unique_molecules)*len(self.subnet_config['small_molecule_target'])} unique molecules to input directory")
        try:
            for target in self.subnet_config['small_molecule_target']:
                protein_sequence = get_sequence_from_protein_code(target)
                for smiles, ids in self.unique_molecules.items():
                    yaml_content = _create_yaml_content(protein_sequence, smiles)
                    with open(os.path.join(self.input_dir, f"{ids[0][1]}_{target}.yaml"), "w") as f:
                        f.write(yaml_content)
            bt.logging.info(f"YAML files written successfully")
        except Exception as e:
            bt.logging.error(f"Error writing YAML files: {e}")
            bt.logging.error(traceback.format_exc())
            return None

        bt.logging.info(f"Preprocessing data for Boltz2 complete")

    def score_molecules(self, valid_molecules_by_uid: dict, score_dict: dict, subnet_config: dict) -> None:
        self.subnet_config = subnet_config

        self._preprocess_data_for_boltz(valid_molecules_by_uid, score_dict)

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
                seed = self.base_seed,  
                affinity_mw_correction = self.config['affinity_mw_correction'],
                override = self.config['override'],
                num_workers = 0,
            )
            bt.logging.info(f"Boltz2 predictions complete")

        except Exception as e:
            bt.logging.error(f"Error running Boltz2: {e}")
            bt.logging.error(traceback.format_exc())
            return None

        # Collect scores and distribute results to all UIDs
        self._postprocess_data(score_dict)
        # Defer cleanup tp preserve unique_molecules for result submission

    def _postprocess_data(self, score_dict: dict) -> None:
        scores = self._collect_scores()

        if self.config['remove_files']:
            self._cleanup_files()

        self._distribute_scores(scores) 
        bt.logging.debug(f"final_boltz_scores: {self.final_boltz_scores}")

         # update score_dict with scores that will be used for ranking
        for uid, data in score_dict.items():
            if uid in self.final_boltz_scores:
                data['molecule_scores'] = [
                    [self.final_boltz_scores[uid][target]] 
                    for target in self.subnet_config['small_molecule_target']
                ]
            else:
                target_count = len(self.subnet_config['small_molecule_target'])
                mode = self.subnet_config['boltz_mode']
                data['molecule_scores'] = [[math.inf] if mode == "min" else [-math.inf] for _ in range(target_count)]

    def _extract_metrics(self, metrics: dict) -> dict:
        """Extract all metrics from a metrics dict, returning None for missing values."""
        metric_names = [
            "affinity_probability_binary", "affinity_pred_value",
            "affinity_probability_binary1", "affinity_pred_value1",
            "affinity_probability_binary2", "affinity_pred_value2",
            "confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
            "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
            "chains_ptm", "pair_chains_iptm"
        ]
        return {name: metrics.get(name, None) for name in metric_names}
    
    def _combine_boltz_scores(self, scores: dict, smiles: str, heavy_atom_count: int) -> float:
        if self.subnet_config['combination_strategy'] == "average":
            return np.mean([scores[metric] for metric in self.subnet_config['boltz_metric']])
        elif self.subnet_config['combination_strategy'] == 'heavy_atom_normalization':
            if heavy_atom_count != 0:
                normalized_score = (scores[self.subnet_config['boltz_metric'][0]] - scores[self.subnet_config['boltz_metric'][1]]) / heavy_atom_count
                return normalized_score
            else:
                bt.logging.warning(f"Heavy atom count is 0 for smiles: {smiles}")
                return math.inf if self.subnet_config['boltz_mode'] == "min" else -math.inf
        else:
            bt.logging.error(f"Invalid combination strategy: {self.subnet_config['combination_strategy']}")
            return None

    def _load_prediction_files(self, results_path: str) -> dict:
        """Load affinity and confidence prediction files from results directory."""
        combined_data = {}
        if not os.path.exists(results_path):
            bt.logging.warning(f"Results path does not exist: {results_path}")
            return combined_data
        
        for filepath in os.listdir(results_path):
            file_path = os.path.join(results_path, filepath)
            if filepath.startswith('affinity') or filepath.startswith('confidence'):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    combined_data.update(data)
                except (json.JSONDecodeError, IOError) as e:
                    bt.logging.error(f"Failed to load {file_path}: {e}")
        return combined_data

    def _cleanup_files(self) -> None:
        try:
            shutil.rmtree(os.path.join(self.output_dir, 'boltz_results_inputs'))
            for yaml_file in glob.glob(os.path.join(self.input_dir, '*.yaml')):
                os.remove(yaml_file)
        except OSError as e:
            bt.logging.error(f"Error removing files: {e}")

    def _collect_scores(self) -> dict:
        scores = {}
        for smiles, id_list in self.unique_molecules.items():
            mol_idx = id_list[0][1] # unique molecule identifier, same for all UIDs
            if mol_idx not in scores:
                scores[mol_idx] = {}
            for target in self.subnet_config['small_molecule_target']:
                results_path = os.path.join(self.output_dir, 'boltz_results_inputs', 'predictions', f'{mol_idx}_{target}')
                scores[mol_idx][target] = self._load_prediction_files(results_path)

        return scores

    def _distribute_scores(self, scores: dict) -> None:
        self.per_molecule_components = {}
        self.final_boltz_scores = {}
        bt.logging.debug(f"molecules: {self.unique_molecules}")

        for smiles, id_list in self.unique_molecules.items():
            try:
                heavy_atom_count = get_heavy_atom_count(smiles)
            except Exception:
                heavy_atom_count = 0

            for uid, mol_idx in id_list:
                if uid not in self.final_boltz_scores:
                    self.final_boltz_scores[uid] = {}
                    
                for target in self.subnet_config['small_molecule_target']:
                    if len(self.subnet_config['boltz_metric']) > 1:
                        final_score_target = self._combine_boltz_scores(scores[mol_idx][target], smiles, heavy_atom_count)
                    else:
                        final_score_target = scores[mol_idx][target].get(self.subnet_config['boltz_metric'][0],
                                                                         math.inf if self.subnet_config['boltz_mode'] == "min" else -math.inf)

                    self.final_boltz_scores[uid][target] = final_score_target

                    # save all components for later use
                    metrics = scores[mol_idx][target]
                    if uid not in self.per_molecule_components:
                        self.per_molecule_components[uid] = {}
                    if smiles not in self.per_molecule_components[uid]:
                        self.per_molecule_components[uid][smiles] = {}
                    self.per_molecule_components[uid][smiles][target] = self._extract_metrics(metrics)
        
        bt.logging.debug(f"per_molecule_components: {self.per_molecule_components}")
        