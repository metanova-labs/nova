import os
import sys
import math
from pathlib import Path

NOVA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(NOVA_DIR)

import yaml

import bittensor as bt
import pandas as pd

from utils import get_sequence_from_protein_code, seq_hash
from boltzgen.cli.boltzgen import (
    build_parser,
    configure_command,
    execute_command,
)

BOLTZGEN_TMP_FILES_DIR = os.path.join(NOVA_DIR, "external_tools", "boltzgen", "boltzgen_tmp_files")
BOLTZGEN_CONFIG_FILE = os.path.join(NOVA_DIR, "config", "boltzgen_config.yaml")


class BoltzgenWrapper:
    def __init__(self):
        with open(BOLTZGEN_CONFIG_FILE, 'r') as f:
            self.boltzgen_config = yaml.load(f, Loader=yaml.FullLoader)
        self.flat_metrics = self._flatten_metrics()

        self.tmp_dir = BOLTZGEN_TMP_FILES_DIR
        self.input_dir = os.path.join(self.tmp_dir, "inputs")
        self.output_dir = os.path.join(self.tmp_dir, "outputs")

        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.per_nanobody_components = {}
        bt.logging.info("BoltzgenWrapper initialized")

    def _flatten_metrics(self):
        flat = {}
        for category, metric_dict in self.boltzgen_config['metrics'].items():
            flat.update(metric_dict)
        return flat

    def run_nanobody_inference(self, valid_nanobodies_by_uid: dict, subnet_config: dict) -> dict:
        """Run configure/execute and fill per_nanobody_components from CSV. No global ranking."""
        self.subnet_config = subnet_config
        self.unique_sequences = self._deduplicate_nanobodies(valid_nanobodies_by_uid)
        self._write_yaml_files()
        self._run_configure_then_execute()
        results = self._collect_local_results()
        self._populate_per_nanobody_components(results)
        bt.logging.debug("run_nanobody_inference: populated per_nanobody_components (ranking deferred)")
        return self.per_nanobody_components

    def finalize_ranking_from_components(
        self, valid_nanobodies_by_uid: dict, subnet_config: dict
    ) -> tuple[dict, dict]:
        """
        Rank designs and produce final_boltzgen_scores from current per_nanobody_components
        (e.g. after apply_external_scores updates metrics).
        """
        self.subnet_config = subnet_config
        self.unique_sequences = self._deduplicate_nanobodies(valid_nanobodies_by_uid)
        df = self._per_nanobody_components_to_dataframe()
        ranked = self._rank_metrics_dataframe(df)
        self._inject_category_ranks_into_components(ranked)
        final_boltzgen_scores = self._distribute_scores(ranked)
        bt.logging.debug(
            f"finalize_ranking_from_components: final_boltzgen_scores keys={list(final_boltzgen_scores.keys())}"
        )
        return final_boltzgen_scores, self.per_nanobody_components

    @classmethod
    def finalize_from_shared_components(
        cls,
        per_nanobody_components: dict,
        valid_nanobodies_by_uid: dict,
        subnet_config: dict,
    ) -> tuple[dict, dict]:
        """Rank using an existing per_nanobody_components dict (mutated in-place by score sharing)."""
        w = cls()
        w.per_nanobody_components = per_nanobody_components
        return w.finalize_ranking_from_components(valid_nanobodies_by_uid, subnet_config)

    def score_nanobodies(self, valid_nanobodies_by_uid: dict, subnet_config: dict):
        """End-to-end: inference + immediate ranking (for tests or callers that skip score sharing)."""
        self.run_nanobody_inference(valid_nanobodies_by_uid, subnet_config)
        return self.finalize_ranking_from_components(valid_nanobodies_by_uid, subnet_config)

    def _deduplicate_nanobodies(self, valid_nanobodies_by_uid: dict):
        """Collect all unique nanobody sequences across all UIDs."""
        unique_sequences = {}

        for uid, nanobodies in valid_nanobodies_by_uid.items():
            sequences = nanobodies['sequences']
            for seq in sequences:
                if seq not in unique_sequences:
                    unique_sequences[seq] = []
                seq_idx = seq_hash(seq)
                unique_sequences[seq].append((uid, seq_idx))

        bt.logging.debug(f"Unique sequences: {unique_sequences}")
        return unique_sequences

    def _create_yaml_content(self, design_sequence: str, target_sequence: str, target: str) -> str:
        """Create YAML content for Boltzgen prediction."""
        return f"""entities:
- protein:
    id: A
    sequence: "{target_sequence}"
    msa: {os.path.join(NOVA_DIR, 'data', 'msa_files', target + '.a3m')}
- protein:
    id: B
    sequence: "{design_sequence}"
    msa: empty
"""

    def _write_yaml_files(self):
        """Write YAML input files for each unique sequence and target."""
        for target in self.subnet_config['nanobody_target']:
            protein_sequence = get_sequence_from_protein_code(target)
            for seq, ids in self.unique_sequences.items():
                yaml_content = self._create_yaml_content(seq, protein_sequence, target)
                record_id = seq_hash(seq)
                yaml_path = os.path.join(self.input_dir, f"{record_id}_{target}_input.yaml")
                with open(yaml_path, "w") as f:
                    f.write(yaml_content)

    def _run_configure_then_execute(self):
        """Run configure and execute commands for Boltzgen."""
        bt.logging.info(f"Running Boltzgen")

        parser = build_parser()

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

        execute_argv = ["execute", self.output_dir]
        if self.boltzgen_config.get('execute_steps'):
            execute_argv += ["--steps", *self.boltzgen_config['execute_steps']]
        if self.boltzgen_config.get('no_subprocess'):
            execute_argv.append("--no_subprocess")

        exe_args = parser.parse_args(execute_argv)
        execute_command(exe_args)

    def _collect_local_results(self) -> pd.DataFrame:
        """Load aggregate metrics CSV without ranking."""
        results_path = os.path.join(
            self.output_dir,
            'intermediate_designs',
            'aggregate_metrics_analyze.csv'
        )
        results = pd.read_csv(results_path)
        results = results[['id', *self.flat_metrics.keys()]]
        results['target_id'] = results['id'].str.split('_').str[1]
        results['nanobody_id'] = results['id'].str.split('_').str[0].astype(str)
        results.drop(columns=['id'], inplace=True)
        return results

    def _populate_per_nanobody_components(self, results: pd.DataFrame) -> None:
        """Fill per_nanobody_components from an unranked metrics dataframe."""
        self.per_nanobody_components = {}
        metric_keys = list(self.flat_metrics.keys())

        for seq, ids in self.unique_sequences.items():
            for uid, seq_idx in ids:
                if uid not in self.per_nanobody_components:
                    self.per_nanobody_components[uid] = {}
                if seq not in self.per_nanobody_components[uid]:
                    self.per_nanobody_components[uid][seq] = {}

                for target in self.subnet_config['nanobody_target']:
                    sid = str(seq_idx)
                    filtered_results = results.loc[
                        (results['nanobody_id'] == sid) &
                        (results['target_id'] == target),
                        metric_keys
                    ].reset_index(drop=True)

                    if not filtered_results.empty:
                        metrics = filtered_results.iloc[0].to_dict()
                    else:
                        bt.logging.error(
                            f"No metrics found for nanobody {seq_idx} and target {target}"
                        )
                        metrics = {}

                    self.per_nanobody_components[uid][seq][target] = metrics

    def _per_nanobody_components_to_dataframe(self) -> pd.DataFrame:
        """Rebuild a metric table from per_nanobody_components for ranking."""
        rows = []
        metric_keys = list(self.flat_metrics.keys())

        for seq, ids in self.unique_sequences.items():
            uid0, seq_idx = ids[0]
            sid = str(seq_idx)
            for target in self.subnet_config['nanobody_target']:
                comp = (
                    self.per_nanobody_components.get(uid0, {})
                    .get(seq, {})
                    .get(target, {})
                )
                row = {k: comp.get(k) for k in metric_keys}
                row['nanobody_id'] = sid
                row['target_id'] = target
                rows.append(row)

        return pd.DataFrame(rows)

    def _rank_metrics_dataframe(self, results: pd.DataFrame) -> pd.DataFrame:
        """Dense ranks per metric, then worst_rank and rank_sum"""
        results = results.copy()
        for metric, mode in self.flat_metrics.items():
            if metric not in results.columns:
                continue
            results[f"{metric}_rank"] = results[metric].rank(
                method='dense',
                ascending=True if mode == 'min' else False
            )

        rank_columns = [
            c for c in results.columns
            if c.endswith('_rank') and c not in ['worst_rank', 'rank_sum']
        ]
        if rank_columns:
            results['worst_rank'] = results[rank_columns].max(axis=1)
            results['rank_sum'] = results[rank_columns].sum(axis=1)

        # Calculate rank sums per category (for dashboard display, not used for scoring)
        results = self._rank_metrics_dataframe_by_category(results)
        bt.logging.debug(f"Ranked metrics dataframe by category: {results}")
        return results

    def _rank_metrics_dataframe_by_category(self, results: pd.DataFrame) -> pd.DataFrame:
        """Group ranks by metric category: confidence, physical_interaction, developability"""
        results = results.copy()
        for category, metric_dict in self.boltzgen_config['metrics'].items():
            category_rank_columns = [f"{metric}_rank" for metric in metric_dict.keys()]
            results[f"{category}_rank_sum"] = results[category_rank_columns].sum(axis=1)
            results[f"{category}_worst_rank"] = results[category_rank_columns].max(axis=1)
            bt.logging.debug(f"Ranked {category} metrics: {results[f'{category}_rank_sum']}")
        return results

    def _distribute_scores(self, results: pd.DataFrame) -> dict:
        """Map ranked aggregate (e.g. rank_sum) back to per-UID sequences."""
        final_boltzgen_scores = {}
        rank_col = self.subnet_config['boltzgen_rank_by']
        mode = self.subnet_config['boltzgen_rank_mode']
        sentinel = math.inf if mode == "min" else -math.inf

        for seq, ids in self.unique_sequences.items():
            for uid, seq_idx in ids:
                if uid not in final_boltzgen_scores:
                    final_boltzgen_scores[uid] = {}
                if seq not in final_boltzgen_scores[uid]:
                    final_boltzgen_scores[uid][seq] = {}

                sid = str(seq_idx)
                for target in self.subnet_config['nanobody_target']:
                    try:
                        val = results.loc[
                            (results['nanobody_id'] == sid) &
                            (results['target_id'] == target),
                            rank_col,
                        ].values[0]
                        if hasattr(val, 'item'):
                            val = val.item()
                        final_score_target = float(val)
                    except (IndexError, KeyError, TypeError, ValueError):
                        final_score_target = sentinel

                    final_boltzgen_scores[uid][seq][target] = final_score_target

        return final_boltzgen_scores

    def _inject_category_ranks_into_components(self, ranked: pd.DataFrame) -> None:
        """Write category rank values from the ranked DataFrame back into per_nanobody_components."""
        categories = list(self.boltzgen_config['metrics'].keys())
        rank_cols = ['rank_sum', 'worst_rank']
        for cat in categories:
            for suffix in ['rank_sum', 'worst_rank']:
                col = f"{cat}_{suffix}"
                if col in ranked.columns:
                    rank_cols.append(col)

        for _, row in ranked.iterrows():
            sid = row['nanobody_id']
            target = row['target_id']
            for seq, ids in self.unique_sequences.items():
                if str(ids[0][1]) != sid:
                    continue
                for uid, _ in ids:
                    comp = (self.per_nanobody_components
                            .get(uid, {}).get(seq, {}).get(target))
                    if comp is not None:
                        for col in rank_cols:
                            comp[col] = row[col]
                break