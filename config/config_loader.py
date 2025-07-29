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
    weekly_target = config["protein_selection"]["weekly_target"]
    num_antitargets = config["protein_selection"]["num_antitargets"]

    no_submission_blocks = config["competition"]["no_submission_blocks"]
    
    validation_config = config["molecule_validation"]
    antitarget_weight = validation_config["antitarget_weight"]
    min_heavy_atoms = validation_config["min_heavy_atoms"]
    min_rotatable_bonds = validation_config["min_rotatable_bonds"]
    max_rotatable_bonds = validation_config["max_rotatable_bonds"]
    num_molecules = validation_config["num_molecules"]
    entropy_bonus_threshold = validation_config["entropy_bonus_threshold"]
    entropy_start_weight = validation_config["entropy_start_weight"]
    entropy_start_epoch = validation_config["entropy_start_epoch"]
    entropy_step_size = validation_config["entropy_step_size"]
    molecule_repetition_weight = validation_config["molecule_repetition_weight"]
    molecule_repetition_threshold = validation_config["molecule_repetition_threshold"]

    # Load reaction filtering configuration
    reaction_config = config["reaction_filtering"]
    random_valid_reaction = reaction_config["random_valid_reaction"]

    return {
        'weekly_target': weekly_target,
        'num_antitargets': num_antitargets,
        'no_submission_blocks': no_submission_blocks,
        'antitarget_weight': antitarget_weight,
        'min_heavy_atoms': min_heavy_atoms,
        'min_rotatable_bonds': min_rotatable_bonds,
        'max_rotatable_bonds': max_rotatable_bonds,
        'num_molecules': num_molecules,
        'entropy_bonus_threshold': entropy_bonus_threshold,
        'entropy_start_weight': entropy_start_weight,
        'entropy_start_epoch': entropy_start_epoch,
        'entropy_step_size': entropy_step_size,
        'molecule_repetition_weight': molecule_repetition_weight,
        'molecule_repetition_threshold': molecule_repetition_threshold,
        'random_valid_reaction': random_valid_reaction
    }