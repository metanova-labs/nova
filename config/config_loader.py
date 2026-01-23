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
    small_molecule_target = config["protein_selection"]["small_molecule_target"].split(",")
    nanobody_target = config["protein_selection"]["nanobody_target"].split(",")

    no_submission_blocks = config["competition"]["no_submission_blocks"]
    nanobody_weight = config["competition"]["nanobody_weight"]
    
    molecule_config = config["molecule_requirements"]
    min_heavy_atoms = molecule_config["min_heavy_atoms"]
    min_rotatable_bonds = molecule_config["min_rotatable_bonds"]
    max_rotatable_bonds = molecule_config["max_rotatable_bonds"]
    num_molecules = molecule_config["num_molecules"]
    min_entropy = molecule_config["min_entropy"]

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

    # Load reaction filtering configuration
    reaction_config = config["reaction_filtering"]
    random_valid_reaction = reaction_config["random_valid_reaction"]

    return {
        'small_molecule_target': small_molecule_target,
        'nanobody_target': nanobody_target,
        'no_submission_blocks': no_submission_blocks,
        'nanobody_weight': nanobody_weight,
        'min_heavy_atoms': min_heavy_atoms,
        'min_rotatable_bonds': min_rotatable_bonds,
        'max_rotatable_bonds': max_rotatable_bonds,
        'num_molecules': num_molecules,
        'min_entropy': min_entropy,
        'random_valid_reaction': random_valid_reaction,
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
        'boltzgen_rank_mode': boltzgen_rank_mode,
        'boltzgen_rank_by': boltzgen_rank_by,
    }