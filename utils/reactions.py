import os
import sqlite3
import bittensor as bt


def get_total_reactions() -> int:
    """Query database for total number of reactions, add 1 for savi option."""
    try:
        db_path = os.path.join(os.path.dirname(__file__), "../combinatorial_db/molecules.sqlite")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM reactions")
        count = cursor.fetchone()[0]
        conn.close()
        return count + 1  # +1 for savi option
    except Exception as e:
        bt.logging.warning(f"Could not query reaction count: {e}, defaulting to 4")
        return 4


def is_reaction_allowed(molecule: str, allowed_reaction: str = None) -> bool:
    """
    Check if molecule matches the allowed reaction for this epoch.
    - If allowed_reaction is None: all molecules allowed (filtering disabled)
    - If filtering enabled: only molecules matching the allowed_reaction type are allowed
    """
    if allowed_reaction is None:
        return True  
        
    if not molecule:
        return False 
    
    if molecule.startswith("rxn:"):
        try:
            parts = molecule.split(":")
            if len(parts) >= 2:
                rxn_id = int(parts[1])
                return allowed_reaction == f"rxn:{rxn_id}"
            return False  # Malformed rxn format
        except Exception as e:
            bt.logging.warning(f"Error parsing reaction molecule '{molecule}': {e}")
            return False
    else:
        # Not in reaction format, only allowed if savi is the allowed reaction
        return allowed_reaction == "savi"
