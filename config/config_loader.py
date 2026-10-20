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
    nanobody_structure = config["protein_selection"]["nanobody"]["structure"]
    nanobody_structure_chain = config["protein_selection"]["nanobody"]["structure_chain"]
    nanobody_structure_res_index = config["protein_selection"]["nanobody"]["structure_res_index"]
    nanobody_structure_binding_site = config["protein_selection"]["nanobody"]["structure_binding_site"]

    no_submission_blocks = config["competition"]["no_submission_blocks"]
    nanobody_weight = config["competition"]["nanobody_weight"]

    payout_config = config.get("payout", {})
    emission_override_uid = payout_config.get("override_uid")
    emission_override_enabled = payout_config.get("enabled", False)

    molecule_config = config["molecule_requirements"]
    num_molecules = molecule_config["num_molecules"]
    min_entropy = molecule_config["min_entropy"]
    filters = molecule_config["filters"]
    max_similarity_to_historical = molecule_config["max_similarity_to_historical"]

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
    max_homopolymer_run = nanobody_config["max_homopolymer_run"]
    max_di_repeat_pairs = nanobody_config["max_di_repeat_pairs"]
    reject_signal_peptides = nanobody_config["reject_signal_peptides"]
    sp_window = nanobody_config["sp_window"]
    sp_hydro_min_in_window = nanobody_config["sp_hydro_min_in_window"]
    sp_scan_prefix = nanobody_config["sp_scan_prefix"]
    max_similarity_score = nanobody_config["max_similarity_score"]
    num_top_sequences = nanobody_config["num_top_sequences"]
    enforce_vhh_hallmarks = nanobody_config["enforce_vhh_hallmarks"]
    psh_strict_rejection_threshold = nanobody_config["psh_strict_rejection_threshold"]
    min_human_v_identity = nanobody_config["min_human_v_identity"]
    min_human_j_identity = nanobody_config["min_human_j_identity"]

    germline_filtering_config = config["germline_filtering"]
    v_neighborhood_categories = germline_filtering_config["v_neighborhood_categories"]
    cdr3_length_range = range(germline_filtering_config["cdr3_length_range"]["min"], 
        germline_filtering_config["cdr3_length_range"]["max"] + 1) # inclusive max

    # Load reaction filtering configuration
    reaction_config = config["reaction_filtering"]
    random_valid_reaction = reaction_config["random_valid_reaction"]
    allowed_reactions = reaction_config["allowed_reactions"]

    return {
        'small_molecule_target': small_molecule_target,
        'small_molecule_target_clip_interval': small_molecule_target_clip_interval,
        'nanobody_target': nanobody_target,
        'nanobody_target_clip_interval': nanobody_target_clip_interval,
        'nanobody_structure': nanobody_structure,
        'nanobody_structure_chain': nanobody_structure_chain,
        'nanobody_structure_res_index': nanobody_structure_res_index,
        'nanobody_structure_binding_site': nanobody_structure_binding_site,
        'no_submission_blocks': no_submission_blocks,
        'nanobody_weight': nanobody_weight,
        'emission_override_uid': emission_override_uid,
        'emission_override_enabled': emission_override_enabled,
        'filters': filters,
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
        'max_homopolymer_run': max_homopolymer_run,
        'max_di_repeat_pairs': max_di_repeat_pairs,
        'reject_signal_peptides': reject_signal_peptides,
        'sp_window': sp_window,
        'sp_hydro_min_in_window': sp_hydro_min_in_window,
        'sp_scan_prefix': sp_scan_prefix,
        'max_similarity_score': max_similarity_score,
        'num_top_sequences': num_top_sequences,
        'enforce_vhh_hallmarks': enforce_vhh_hallmarks,
        'boltzgen_rank_mode': boltzgen_rank_mode,
        'boltzgen_rank_by': boltzgen_rank_by,
        'max_similarity_to_historical': max_similarity_to_historical,
        'psh_strict_rejection_threshold': psh_strict_rejection_threshold,
        'min_human_v_identity': min_human_v_identity,
        'min_human_j_identity': min_human_j_identity,
        'v_neighborhood_categories': v_neighborhood_categories,
        'cdr3_length_range': cdr3_length_range,
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