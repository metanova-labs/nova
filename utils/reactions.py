import os
import sqlite3
import bittensor as bt


def get_total_reactions() -> int:
    """Query database for total number of reactions"""
    try:
        db_path = os.path.join(os.path.dirname(__file__), "../combinatorial_db/molecules.sqlite")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM reactions")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        raise RuntimeError(
            f"Could not query the combinatorial database reaction count: {e}"
        ) from e


def is_reaction_allowed(molecule: str, allowed_reactions: list[str] | str = None) -> bool:
    """
    Check if a molecule's reaction id is allowed for this epoch.
    - If allowed_reactions is None: all molecules are allowed (filtering disabled)
    - A string is treated as a one-element list
    - Otherwise the id must be a member of the list (e.g. "rxn:3")
    """
    if allowed_reactions is None:
        return True

    if isinstance(allowed_reactions, str):
        allowed_reactions = [allowed_reactions]

    if not molecule:
        return False 
    
    if molecule.startswith("rxn:"):
        try:
            parts = molecule.split(":")
            if len(parts) >= 2:
                rxn_id = int(parts[1])
                return f"rxn:{rxn_id}" in allowed_reactions
            return False  # Malformed rxn format
        except Exception as e:
            bt.logging.warning(f"Error parsing reaction molecule '{molecule}': {e}")
            return False
