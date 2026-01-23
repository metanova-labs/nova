import argparse
import os
import sys
import math
from pathlib import Path

NOVA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(NOVA_DIR)

import hashlib
import yaml

import bittensor as bt
import numpy as np
import pandas as pd

from utils import get_sequence_from_protein_code
from boltzgen.cli.boltzgen import (
    build_parser,
    configure_command,
    execute_command,
)

BOLTZGEN_TMP_FILES_DIR = os.path.join(NOVA_DIR, "boltzgen", "boltzgen_tmp_files")
BOLTZGEN_CONFIG_FILE = os.path.join(NOVA_DIR, "config", "boltzgen_config.yaml")


def _get_record_id(design_sequence: str) -> int:
    h = hashlib.sha256(str(design_sequence).encode()).digest()
    return int.from_bytes(h[:8], "little") % (2**31 - 1)


class BoltzgenWrapper:
    def __init__(self):
        with open(BOLTZGEN_CONFIG_FILE, 'r') as f:
            self.boltzgen_config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.tmp_dir = BOLTZGEN_TMP_FILES_DIR
        self.input_dir = os.path.join(self.tmp_dir, "inputs")
        self.output_dir = os.path.join(self.tmp_dir, "outputs")

        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.per_nanobody_components = {}
        bt.logging.info("BoltzgenWrapper initialized")

    def score_nanobodies(self, valid_nanobodies_by_uid: dict, subnet_config: dict):
        self.subnet_config = subnet_config
        self.unique_sequences = self._deduplicate_nanobodies(valid_nanobodies_by_uid)
        self._write_yaml_files()
        
        self._run_configure_then_execute()
        results = self._collect_results_and_rank_designs()
        final_boltzgen_scores, per_nanobody_components = self._distribute_scores(results)
        
        bt.logging.debug(f"final_boltzgen_scores: {final_boltzgen_scores}")
        return final_boltzgen_scores, per_nanobody_components

    def _deduplicate_nanobodies(self, valid_nanobodies_by_uid: dict):
        """Collect all unique nanobody sequences across all UIDs."""
        unique_sequences = {}

        for uid, nanobodies in valid_nanobodies_by_uid.items():
            sequences = nanobodies['sequences']
            for seq in sequences:
                if seq not in unique_sequences:
                    unique_sequences[seq] = []
                seq_idx = _get_record_id(seq)
                unique_sequences[seq].append((uid, seq_idx))

        bt.logging.debug(f"Unique sequences: {unique_sequences}")
        return unique_sequences

    def _create_yaml_content(self, design_sequence: str, target_sequence: str) -> str:
        """Create YAML content for Boltzgen prediction."""
        return f"""entities:
- protein:
    id: A
    sequence: "{target_sequence}"
- protein:
    id: B
    sequence: "{design_sequence}"
"""

    def _write_yaml_files(self):
        """Write YAML input files for each unique sequence and target."""
        for target in self.subnet_config['nanobody_target']:
            protein_sequence = get_sequence_from_protein_code(target)
            for seq, ids in self.unique_sequences.items():
                yaml_content = self._create_yaml_content(seq, protein_sequence)
                record_id = _get_record_id(seq)
                yaml_path = os.path.join(self.input_dir, f"{record_id}_{target}_input.yaml")
                with open(yaml_path, "w") as f:
                    f.write(yaml_content)

    def _run_configure_then_execute(self):
        """Run configure and execute commands for Boltzgen."""
        bt.logging.info(f"Running Boltzgen")
        
        parser = build_parser()

        # Configure
        configure_argv = [
            "configure",
            self.input_dir,
            "--output", self.output_dir,
            "--protocol", self.boltzgen_config['protocol'],
            "--num_designs", str(self.boltzgen_config['num_designs']),
        ]
        if self.boltzgen_config['skip_inverse_folding']:
            configure_argv.append("--skip_inverse_folding")
        if self.boltzgen_config.get('step_scale') is not None:
            configure_argv += ["--step_scale", str(self.boltzgen_config['step_scale'])]
        if self.boltzgen_config.get('noise_scale') is not None:
            configure_argv += ["--noise_scale", str(self.boltzgen_config['noise_scale'])]

        cfg_args = parser.parse_args(configure_argv)
        configure_command(cfg_args)

        # Execute
        execute_argv = ["execute", self.output_dir]
        if self.boltzgen_config.get('execute_steps'):
            execute_argv += ["--steps", *self.boltzgen_config['execute_steps']]
        if self.boltzgen_config.get('no_subprocess'):
            execute_argv.append("--no_subprocess")

        exe_args = parser.parse_args(execute_argv)
        execute_command(exe_args)

    def _collect_results_and_rank_designs(self) -> pd.DataFrame:
        """Collect results from CSV and rank designs by metrics."""
        results_path = os.path.join(
            self.output_dir,
            'intermediate_designs',
            'aggregate_metrics_analyze.csv'
        )
        results = pd.read_csv(results_path)
        results = results[['id', *self.boltzgen_config['metrics'].keys()]]
        results['target_id'] = results['id'].str.split('_').str[1]
        results['nanobody_id'] = results['id'].str.split('_').str[0].astype(int)
        results.drop(columns=['id'], inplace=True)

        for metric, mode in self.boltzgen_config['metrics'].items():
            results[f"{metric}_rank"] = results[metric].rank(
                method='dense',
                ascending=True if mode == 'min' else False
            )
        
        # Get worst rank and rank sum for each design
        rank_columns = [
            c for c in results.columns
            if c.endswith('_rank') and c not in ['worst_rank', 'rank_sum']
        ]
        results['worst_rank'] = results[rank_columns].max(axis=1)
        results['rank_sum'] = results[rank_columns].sum(axis=1)

        return results

    def _distribute_scores(self, results: pd.DataFrame):
        """Distribute scores to all UIDs and collect per-nanobody components."""
        final_boltzgen_scores = {}
        self.per_nanobody_components = {}

        for seq, ids in self.unique_sequences.items():
            for uid, seq_idx in ids:
                if uid not in final_boltzgen_scores:
                    final_boltzgen_scores[uid] = {}
                    final_boltzgen_scores[uid][seq] = {}
                if uid not in self.per_nanobody_components:
                    self.per_nanobody_components[uid] = {}
                    self.per_nanobody_components[uid][seq] = {}

                for target in self.subnet_config['nanobody_target']:
                    try:
                        final_score_target = results.loc[
                            (results['nanobody_id'] == seq_idx) &
                            (results['target_id'] == target),
                            self.subnet_config['boltzgen_rank_by']
                        ].values[0]
                    except (IndexError, KeyError):
                        final_score_target = (
                            math.inf if self.subnet_config['boltzgen_rank_mode'] == "min"
                            else -math.inf
                        )

                    final_boltzgen_scores[uid][seq][target] = final_score_target

                    # Save components for later use
                    filtered_results = results.loc[
                        (results['nanobody_id'] == seq_idx) &
                        (results['target_id'] == target),
                        self.boltzgen_config['metrics'].keys()
                    ].reset_index(drop=True)
                    
                    if not filtered_results.empty:
                        metrics = filtered_results.iloc[0].to_dict()
                    else:
                        bt.logging.error(
                            f"No metrics found for nanobody {seq_idx} and target {target}"
                        )
                        metrics = {}
                    
                    self.per_nanobody_components[uid][seq][target] = metrics

        return final_boltzgen_scores, self.per_nanobody_components