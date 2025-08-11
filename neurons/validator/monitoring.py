"""
Validator monitoring and reporting functionality
"""

import math
import os
import requests
import bittensor as bt


def monitor_validator(score_dict, metagraph, current_epoch, current_block, validator_hotkey, winning_uid):
    """
    Send validator monitoring data to external monitoring service.
    
    Args:
        score_dict: Dictionary of scores for all UIDs
        metagraph: Current metagraph
        current_epoch: Current epoch number
        current_block: Current block number
        validator_hotkey: Validator's hotkey address
        winning_uid: UID of the winning miner
    """
    api_key = os.environ.get('VALIDATOR_API_KEY')
    if not api_key:
        return
    
    try:
        import torch
        machine_info = {
            "torch_version": torch.__version__
        }
        if torch.cuda.is_available():
            machine_info["cuda_version"] = torch.version.cuda
            machine_info["gpu_name"] = torch.cuda.get_device_name(0)
        
        best_rounded_score = max([round(d['final_score'], 4) for d in score_dict.values() if 'final_score' in d], default=-math.inf)
        
        winning_group = []
        for uid, data in score_dict.items():
            if 'final_score' in data and round(data['final_score'], 4) == best_rounded_score:
                winning_group.append({
                    "uid": uid,
                    "hotkey": metagraph.hotkeys[uid] if uid < len(metagraph.hotkeys) else "unknown",
                    "final_score": data['final_score'],
                    "blocks_elapsed": (data.get('block_submitted', 0) % 361),
                    "push_time": data.get('push_time', ''),
                    "winner": uid == winning_uid
                })
        
        requests.post("https://valiwatch-production.up.railway.app/weights-info", json={
            "epoch": current_epoch,
            "current_block": current_block,
            "blocks_into_epoch": current_block % 361,
            "validator_hotkey": validator_hotkey,
            "validator_version": 1.4, #reaction filtering
            "winning_group": winning_group,
            "machine_info": machine_info
        }, headers={"Authorization": f"Bearer {api_key}"}, timeout=5)
        
    except Exception as e:
        bt.logging.debug(f"API send failed: {e}")
