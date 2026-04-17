import yaml
import os

def load_config(path: str = "config/config.yaml"):
    """
    Loads configuration from a YAML file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find config file at '{path}'")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load configuration options
    small_molecule_target = config["protein_selection"]["small_molecule"]["target"].split(",")
    small_molecule_target_clip_interval = [tuple(interval) if interval else None for interval in config["protein_selection"]["small_molecule"]["clip_interval"]]
    nanobody_target = config["protein_selection"]["nanobody"]["target"].split(",")
    nanobody_target_clip_interval = [tuple(interval) if interval else None for interval in config["protein_selection"]["nanobody"]["clip_interval"]]

    no_submission_blocks = config["competition"]["no_submission_blocks"]
    nanobody_weight = config["competition"]["nanobody_weight"]

    payout_config = config["payout"]
    emission_override_uid = payout_config["override_uid"]
    emission_api_base_url = payout_config["api_base_url"]
    emission_proportion_field = payout_config["proportion_field"]
    emission_override_enabled = payout_config["enabled"]

    molecule_config = config["molecule_requirements"]
    min_heavy_atoms = molecule_config["min_heavy_atoms"]
    min_rotatable_bonds = molecule_config["min_rotatable_bonds"]
    max_rotatable_bonds = molecule_config["max_rotatable_bonds"]
    num_molecules = molecule_config["num_molecules"]
    min_entropy = molecule_config["min_entropy"]
    banned_atom_types = molecule_config["banned_atom_types"]

    boltz_config = config["boltz2_config"]
    boltz_metric = boltz_config["boltz_metric"]
    combination_strategy = boltz_config["combination_strategy"]
    boltz_mode = boltz_config["boltz_mode"]

    boltzgen_config = config["boltzgen_config"]
    boltzgen_rank_mode = boltzgen_config["boltzgen_rank_mode"]
    boltzgen_rank_by = boltzgen_config["boltzgen_rank_by"]


    nanobody_config = config["nanobody_requirements"]
    num_sequences = nanobody_config["num_sequences"]
    min_sequence_length = nanobody_config["min_sequence_length"]
    max_sequence_length = nanobody_config["max_sequence_length"]
    min_cysteines = nanobody_config["min_cysteines"]
    cys_pair_min_separation = nanobody_config["cys_pair_min_separation"]
    cys_pair_max_separation = nanobody_config["cys_pair_max_separation"]
    max_homopolymer_run = nanobody_config["max_homopolymer_run"]
    max_di_repeat_pairs = nanobody_config["max_di_repeat_pairs"]
    reject_signal_peptides = nanobody_config["reject_signal_peptides"]
    sp_window = nanobody_config["sp_window"]
    sp_hydro_min_in_window = nanobody_config["sp_hydro_min_in_window"]
    sp_scan_prefix = nanobody_config["sp_scan_prefix"]
    min_nativeness_score = nanobody_config["min_nativeness_score"]
    min_human_framework_score = nanobody_config["min_human_framework_score"]
    max_similarity_score = nanobody_config["max_similarity_score"]
    num_top_sequences = nanobody_config["num_top_sequences"]

    # Load reaction filtering configuration
    reaction_config = config["reaction_filtering"]
    random_valid_reaction = reaction_config["random_valid_reaction"]
    allowed_reactions = reaction_config["allowed_reactions"]

    return {
        'small_molecule_target': small_molecule_target,
        'small_molecule_target_clip_interval': small_molecule_target_clip_interval,
        'nanobody_target': nanobody_target,
        'nanobody_target_clip_interval': nanobody_target_clip_interval,
        'no_submission_blocks': no_submission_blocks,
        'nanobody_weight': nanobody_weight,
        'emission_override_uid': emission_override_uid,
        'emission_api_base_url': emission_api_base_url,
        'emission_proportion_field': emission_proportion_field,
        'emission_override_enabled': emission_override_enabled,
        'min_heavy_atoms': min_heavy_atoms,
        'min_rotatable_bonds': min_rotatable_bonds,
        'max_rotatable_bonds': max_rotatable_bonds,
        'banned_atom_types': banned_atom_types,
        'num_molecules': num_molecules,
        'min_entropy': min_entropy,
        'random_valid_reaction': random_valid_reaction,
        'allowed_reactions': allowed_reactions,
        'boltz_metric': boltz_metric,
        'combination_strategy': combination_strategy,
        'boltz_mode': boltz_mode,
        'num_sequences': num_sequences,
        'min_sequence_length': min_sequence_length,
        'max_sequence_length': max_sequence_length,
        'min_cysteines': min_cysteines,
        'cys_pair_min_separation': cys_pair_min_separation,
        'cys_pair_max_separation': cys_pair_max_separation,
        'max_homopolymer_run': max_homopolymer_run,
        'max_di_repeat_pairs': max_di_repeat_pairs,
        'reject_signal_peptides': reject_signal_peptides,
        'sp_window': sp_window,
        'sp_hydro_min_in_window': sp_hydro_min_in_window,
        'sp_scan_prefix': sp_scan_prefix,
        'min_nativeness_score': min_nativeness_score,
        'min_human_framework_score': min_human_framework_score,
        'max_similarity_score': max_similarity_score,
        'num_top_sequences': num_top_sequences,
        'boltzgen_rank_mode': boltzgen_rank_mode,
        'boltzgen_rank_by': boltzgen_rank_by,
    }

def load_boltzgen_metrics(path: str = "config/boltzgen_config.yaml") -> dict:
    """Load boltzgen nanobody metrics as a flat {metric_name: mode} dict,
    regardless of category grouping in the YAML."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find boltzgen config at '{path}'")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    flat = {}
    for category, metric_dict in config.get("metrics", {}).items():
        if isinstance(metric_dict, dict):
            flat.update(metric_dict)
    return flat