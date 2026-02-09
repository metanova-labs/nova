# NOVA Compound miner documentation

## Overview
NOVA Compound is a competition that incentivizes miners to come up with drug-like molecules that are predicted to have high binding affinity to specific targets of interest. High binding affinity is a fundamental requirement for developing therapeutics, since it measures how energetically favorable the interaction between target and drug candidateis (a drug can only exert any sort of effect if it's able to interact with the target).

NOVA Compound currently has two tiers in the competition:
- `Small molecules`: Miners use billion-sized databases to find small molecules that are predicted to have high binding affinity to the target. The binding affinity for small molecules is predicted with [Boltz-2](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1).
- `Biologics`: Miners are required to design nanobody-like sequences that perform well in a set of metrics that comprise confidence in the accuracy of the folded complex, strength of the physical interaction between nanobody and target, and potential for toxicity/instability. These metrics are calculated through a modified version of the [BoltzGen](https://hannes-stark.com/assets/boltzgen.pdf) pipeline.

Targets are changed weekly, on Mondays at 3PM UTC. Targets are announced approximately 24 hours before that on NOVA's Discord channel.

---

## Competition flow:
Competitions last for **361 blocks** and run continuously. Miner submissions must be timelock encrypted and uploaded to a public github repo. No submissions are accepted in the last 10 blocks of a competition.

1. **Challenge parameters**  
   Each epoch, challenge parameters are derived from the **epoch start block hash** via the `get_challenge_params_from_blockhash()` function. This function returns:
   - `small_molecule_target`: protein(s) for small-molecule tasks (pulled from config)
   - `nanobody_target`: protein(s) for nanobody tasks (also pulled from config)
   - `allowed_reaction`: when reaction filtering is on, small molecules need to comply with the single allowed reaction type for that epoch (e.g. `"rxn:3"`)

2. **Candidate products**  
   The basline miner provides functions that comply with submission format (encryption, expected format, etc.) The actual selection of molecules and design of nanobody sequences need to be implemented by each miner (see resources to help with that in the sections below). **This means that simply running the baseline miner without modifications will *NOT* result in incentive distribution.**
   - **Small molecules**: one or more molecule ids from the combinatorial database available on `combinatorial_db/molecules.sqlite` (e.g. `rxn:3:49485:2099:70633`). Molecules must comply with the `molecule_requirements` section on `config/config.yaml`
   - **Nanobodies**: one or more amino-acid sequences that comply with the `nanobody_requirements` section on `config/config.yaml`.

3. **Commitment and upload**  
Functions for forming the payload string, encrypting contents and uploading responses to github are provided. Your github repo must be public and credentials need to be specified in your `.env` file.
   - Payload string: `"{candidate_small_molecule}|{candidate_nanobody}"` (single `|` delimiter).
   - If `num_molecules` or `num_sequences` > 1, values must be comma-separated.
   - Payload is **encrypted** with Bittensor Drand Timelock for the miner’s UID and current block.
   - Miner calls `set_commitment()` with a **commit string** pointing to the future file: `{github_path}/{hash}.txt` (hash = first 20 chars of SHA256 of encrypted content).
   - The **encrypted content** (the literal that will be `literal_eval`’d on the validator) is base64-encoded and uploaded to GitHub at that path.

So: **chain = commitment (pointer)**; **GitHub = encrypted payload**. The validator uses the commitment to fetch and decrypt the payload, then runs validity and scoring.

---

## 2. Submission format (validator expectations)

The decrypted string must parse as follows (see `commitments.parse_decrypted_submission`):

- **Format:** `mol1,mol2,...|seq1,seq2,...`
- Exactly **one** `|` separating molecules from sequences.
- No whitespace and no non-ASCII characters.
- **Molecules:** comma-separated; for multiple molecules use `mol1,mol2,...`; for a single molecule, `mol` is enough. 
- **Sequences:** comma-separated; same idea for one vs many.

Note: the character ~ is a null placeholder. It is accepted in miner submissions if they do not wish to participate in either molecules or nanobodies competition.

Examples:
```
  mol|seq   ✅
  ~|seq     ✅
  mol|~     ✅
  ~|~       Not sure why you would do that, but technically ✅

  mol       ❌
  seq       ❌
  mol|      ❌
  |seq      ❌
  seq|mol   ❌  # order must be mol|seq
  mol,seq   ❌  # separator must be `|`

```

---

## 3. Small-molecule validity

Implemented in `neurons/validator/molecule_validity.validate_molecules_and_calculate_entropy()`. The validator iterates over your molecules and in case of any non-compliant molecule, the whole submission is treated as invalid (no score). Checks are applied in this logical order:

1. **No null token**  
   If any molecule string contains `~`, the submission is skipped (burned participation).

2. **No duplicates**  
   Duplicate molecule strings in the same submission → invalid.

3. **Reaction allowed (if configured)**  
   If `config.random_valid_reaction` is true, each molecule must match the epoch’s `allowed_reaction` (e.g. `rxn:3`). Format: molecule starts with `rxn:` and the reaction ID must match. If any molecule uses a disallowed reaction, the whole submission is deemed invalid.

4. **SMILES**  
   For each molecule, the validator derives SMILES via `get_smiles(molecule)`. If no valid SMILES is found for any molecule, the whole submission is invalid.

5. **Heavy atom count**  
   `get_heavy_atom_count(smiles)` must be ≥ `config['min_heavy_atoms']` (e.g. 10). Otherwise invalid.

6. **Rotatable bonds**  
   RDKit is used to compute rotatable bonds. Count must be between `config['min_rotatable_bonds']` and `config['max_rotatable_bonds']` (e.g. 1–10). Unparseable or out-of-range → invalid.

7. **Uniqueness per target**  
   For each `target` in `config['small_molecule_target']`, the molecule must be unique for that protein (checked via `entry_unique_for_protein_hf(target, smiles, 'molecules')`). If any molecule is not unique for any target, validation fails.

8. **Chemically identical**  
   After collecting all valid SMILES/names in the list, the validator checks for chemically identical molecules (e.g. same InChIKey). If any are found, the **entire** submission is rejected (duplicate molecules with different names still count as invalid).

If all checks pass, the validator stores valid molecules for that UID and may compute **MACCS entropy** when `num_molecules > 1`; entropy is currently a minimum value that needs to be obtained.

**Takeaway for miners:**  
Submit exactly the number of molecules required; ensure correct reaction type when filtering is on; ensure SMILES are valid, heavy-atom and rotatable-bond counts are in range, and molecules are unique per target and not chemically duplicated.

---

## 4. Nanobody validity

Implemented in `neurons/validator/nanobody_validity.validate_nanobodies()`. Checks are **per-UID**: if any check fails, that UID’s nanobody submission is rejected entirely (no nanobody score).

1. **No null token**  
   If any sequence contains `~`, skip (burned participation).

2. **No duplicate sequences**  
   After normalizing, sequences are hashed; duplicate hashes → invalid.

3. **Length**  
   Each sequence length must be in `[config["min_sequence_length"], config["max_sequence_length"]]` (e.g. 90–150).

4. **Amino acids**  
   Only the 20 standard AAs are allowed: `ACDEFGHIKLMNPQRSTVWY`. Any other character → invalid.

5. **Homopolymer runs**  
   No run of the same amino acid longer than `config["max_homopolymer_run"]` (e.g. AAAAAAAA).

6. **Di-repeat pairs**  
   `max_di_repeat_pairs(seq)` must be ≤ `config["max_di_repeat_pairs"]` (e.g. GSGSGSGS -> 4 di-repeat pairs).

7. **Cysteine count**  
   Each sequence must have at least `config["min_cysteines"]` cysteines (e.g. 1). If `min_cysteines > 1`, at least one sequence must have a plausible cysteine pair (separation within `cys_pair_min_separation` and `cys_pair_max_separation`).

8. **Signal peptide**  
   If enabled, sequences that look like signal peptides (e.g. strong N-terminal hydrophobicity in the configured window) are rejected.

9. **Uniqueness per target**  
   For each `target` in `config["nanobody_target"]`, all sequence hashes must be unique for that target (`entry_unique_for_protein_hf(target, h, 'nanobodies')`). If any hash is not unique for any target, the entire submission is invalid.

**Takeaway for miners:**  
Submit the required number of sequences; normalize (e.g. uppercase, valid AAs only); stay within length and diversity constraints and ensure uniqueness per target.

---

## 5. Scoring (after validity)

Only **valid** submissions (per UID, per type) are scored. We strongly recommend also going through validator docs to understand how submissions are scored.

- **Small molecules**  
  The validator runs Boltz (and the inter-validator score-sharing pipeline). Per-target scores are stored in `score_dict[uid]["molecule_scores"]`. Then, if multiple targets are used, scores for each target will be combined with `calculate_scores_for_type(..., item_type="molecule")` to obtain final scores for each molecule.
  - Averages across targets per molecule, then sums across molecules to get `final_molecule_score`.
  - Averages across validator runs for robustness of non-deterministic results.
  **Winner:** UID with best `final_molecule_score` (max or min by `config['boltz_mode']`). Ties broken by earliest `block_submitted`, then `push_time`, then UID.

- **Nanobodies**  
  Boltzgen produces per-target, per-sequence scores. Instead of using raw scores, all miner designs are ranked per-metric (metrics and rank modes are listed on `config/boltzgen_config.yaml`). Final score for each sequence/target combination is the sum of all ranks obtained (you want your design to consistently rank high on all metrics). Then, `calculate_scores_for_type(..., item_type="nanobody")` does the same idea: per-sequence average across targets, then `final_nanobody_score`.  
  **Winner:** Same idea with `boltzgen_rank_mode` (e.g. `min` for rank-based: lowest rank sum wins).

So: **validity is all-or-nothing per submission**; **scoring** uses only valid items and combines them into a single final score per UID per type, then the validator picks the winner for molecules and the winner for nanobodies.

---

## 6. Config parameters relevant to miners

From `config/config.yaml` and `config_loader.py`:

| Parameter | Used in | Meaning |
|-----------|--------|--------|
| `small_molecule_target`, `nanobody_target` | Challenge + validity | Target proteins for the epoch and uniqueness checks. |
| `random_valid_reaction`, `allowed_reaction` | Molecule validity | When true, only molecules with the epoch’s allowed reaction type are valid. |
| `min_heavy_atoms`, `min_rotatable_bonds`, `max_rotatable_bonds` | Molecule validity | RDKit-based filters. |
| `num_molecules`, `num_sequences` | Format + scoring | How many molecules/sequences to send and to score. |
| `min_sequence_length`, `max_sequence_length` | Nanobody validity | Allowed nanobody length range. |
| `min_cysteines`, `cys_pair_*`, `max_homopolymer_run`, `max_di_repeat_pairs` | Nanobody validity | Structural/diversity constraints. |
| `reject_signal_peptides`, `sp_*` | Nanobody validity | Signal-peptide heuristic. |
| `boltz_mode`, `boltzgen_rank_mode` | Ranking | Whether higher or lower score wins. |
| `no_submission_blocks` | Commitments | Commitments too close to epoch end (within this many blocks) are ignored. |

---

## 7. Summary

- **Submission:** One string `molecules|sequences`, encrypted and uploaded; chain stores the GitHub path.
- **Validity:** Strict format (one `|`, no whitespace/non-ASCII), then molecule and nanobody rules above. One failure → no score for that UID for that type.
- **Scoring:** Only valid submissions get Boltz/Boltzgen scores; validator aggregates to a final score per UID and selects the winner per type (with tie-breaking by block and time).

Implementing miners should ensure: correct format, correct reaction type when applicable, and that all molecule and nanobody constraints (including uniqueness per target) are satisfied so that the submission is accepted and scored.

---

## 8. Resources

### Small molecules
`Boltz-2` is an expensive scoring function. For typical target and molecule sizes, binding affinity prediction takes around 45s per molecule on an A100 GPU. This means that you want to decrease the amount of inferences you need to run as much as possible. Some ways you can do that is by using previously identified high-scoring molecules as a base to look for other molecules that can also be high-scoring.
Since the dabatase is combinatorial, you can:
- mix and match reagents to try to form high-scoring molecules
- look for similar reagents using [Synthon Search](https://greglandrum.github.io/rdkit-blog/posts/2024-12-03-introducing-synthon-search.html)

**Other tools and frameworks you can use to optimize your search:**
- MolPAL — Graff et al. (2021). Accelerating high-throughput virtual screening through molecular pool-based active learning. Chem. Sci., 12, 7866-7881. [Paper](https://pubs.rsc.org/en/content/articlelanding/2021/sc/d0sc06805e) | [GitHub](https://github.com/coleygroup/molpal)

- Thompson Sampling — Klarich et al. (2024). Thompson Sampling—An Efficient Method for Searching Ultralarge Synthesis on Demand Databases. J. Chem. Inf. Model., 64(4), 1158–1171. [Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01790) | [GitHub](https://github.com/PatWalters/TS)

- Roulette Wheel Sampling — Zhao et al. (2025). Enhanced Thompson sampling by roulette wheel selection for screening ultralarge combinatorial libraries. J. Cheminform. [Paper](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-025-01105-1)

- V-SYNTHES — Sadybekov et al. (2022). Synthon-based ligand discovery in virtual libraries of over 11 billion compounds. Nature, 601, 452-459. [Paper](https://www.nature.com/articles/s41586-021-04220-9)

- SASS — Cheng & Beroza (2024). Shape-Aware Synthon Search (SASS) for Virtual Screening of Synthon-Based Chemical Spaces. J. Chem. Inf. Model., 64(4), 1251-1260. [Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01865)

- SpaceLight — Bellmann, Penner & Rarey (2021). Topological Similarity Search in Large Combinatorial Fragment Spaces. J. Chem. Inf. Model., 61(1), 238-251. [Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00850)

- SpaceMACS — Schmidt, Klein & Rarey (2022). Maximum Common Substructure Searching in Combinatorial Make-on-Demand Compound Spaces. J. Chem. Inf. Model., 62(9), 2133-2150. [Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00640)

- FTrees-FS — Rarey & Stahl (2001). Similarity searching in large combinatorial chemistry spaces. J. Comput. Aided Mol. Des., 15, 497-520. [Paper](https://link.springer.com/article/10.1023/A:1011144622059)


### Nanobodies
Miners will be required to design novel nanobody-like sequences that perform consistently well across a set of metrics calculated by [BoltzGen](https://github.com/HannesStark/boltzgen/tree/main). The version of BoltzGen provided in the NOVA repo is modified to work as a scoring engine instead of a generative pipeline, so we recommend using the official repo to generate your designs. There is extensive documentation in the BoltzGen repo on how to use it, we also recommend reading the [paper](https://hannes-stark.com/assets/boltzgen.pdf) for more info.

**Other sequence design pipelines you can use:** (but remember your submission will be scored with BoltzGen)
- ODesign — Zhang et al. (2025). ODesign: A world model for biomolecular interaction design. arXiv:2510.22304. [Paper](https://arxiv.org/abs/2510.22304) | [GitHub](https://github.com/The-Institute-for-AI-Molecular-Design/ODesign)

