import requests
from datasets import load_dataset
import bittensor as bt


def get_sequence_from_protein_code(protein_code: str) -> str:
    """Get the amino acid sequence for a protein code from UniProt API or Hugging Face dataset."""
    url = f"https://rest.uniprot.org/uniprotkb/{protein_code}.fasta"
    response = requests.get(url)

    if response.status_code == 200:
        lines = response.text.splitlines()
        sequence_lines = [line.strip() for line in lines if not line.startswith('>')]
        amino_acid_sequence = ''.join(sequence_lines)
        if amino_acid_sequence:
            return amino_acid_sequence
        bt.logging.warning(f"Retrieved empty sequence for {protein_code} from UniProt API")
    else:
        bt.logging.info(f"Failed to retrieve sequence for {protein_code} from UniProt API. Trying Hugging Face dataset.")

    try:
        dataset = load_dataset('Metanova/Proteins', split='train')
        for i in range(len(dataset)):
            if dataset[i]['Entry'] == protein_code:
                sequence = dataset[i]['Sequence']
                bt.logging.info(f"Found sequence for {protein_code} in Hugging Face dataset")
                return sequence
        bt.logging.error(f"Could not find protein {protein_code} in Hugging Face dataset")
    except Exception as e:
        bt.logging.error(f"Error accessing Hugging Face dataset: {e}")
    return None
