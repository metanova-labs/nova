import bittensor as bt


def calculate_dynamic_entropy(starting_weight: float, step_size: float, start_epoch: int, current_epoch: int) -> float:
    """
    Calculate entropy weight based on epochs elapsed since start epoch.
    """
    epochs_elapsed = current_epoch - start_epoch
    
    entropy_weight = starting_weight + (epochs_elapsed * step_size)
    entropy_weight = max(0, entropy_weight)
    
    bt.logging.info(f"Epochs elapsed: {epochs_elapsed}, entropy weight: {entropy_weight}")
    return entropy_weight
