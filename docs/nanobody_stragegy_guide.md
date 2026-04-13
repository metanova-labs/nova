# NOVA Nanobody Mining — Strategy Guide

*A practical guide for software and ML engineers entering the NOVA Compound biologics competition.*

---

## What you'll be doing

In drug discovery, a **nanobody** is a type of antibody derived from camelids (llamas, camels, alpacas). Unlike conventional antibodies which are large, Y-shaped molecules composed of multiple chains, nanobodies are single-chain proteins — just a sequence of amino acids about 90–150 residues long. Despite their small size, they can bind to disease-relevant **target proteins** with high specificity and strength.

**Binding affinity** is the key concept here: it measures how tightly a nanobody grabs onto its target. Think of it like a lock and key: you're designing the key. A drug can only work if it binds its target well, so predicting and maximizing binding affinity is the core objective.

In NOVA's nanobody competition, you are asked to **design amino acid sequences from scratch** (*de novo* design) that are predicted to bind to a specific target protein. You submit these sequences, and validators score them using a modified version of [BoltzGen](https://github.com/HannesStark/boltzgen), a state-of-the-art generative pipeline that folds your sequence into a 3D structure alongside the target, and evaluates how good the predicted interaction looks. The miner whose designs consistently rank highest across multiple quality metrics wins.

---

## Key concepts crash course

**Amino acids** are the building blocks of proteins. There are 20 standard ones, each represented by a single letter: `A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y`. A nanobody sequence is just a string of these characters, like `EVQLVESGGGLVQPGG...`.

**Protein folding** is the process by which a linear amino acid sequence collapses into a specific 3D structure. The structure determines function, so the same sequence always (ideally) folds the same way. Modern AI models like AlphaFold and ESM predict this folding computationally. Boltz-2 and BoltzGen both use AlphaFold under the hood.

**CDRs (Complementarity-Determining Regions)** are the "business end" of a nanobody — three loop regions (CDR1, CDR2, CDR3) that make direct contact with the target. The rest of the sequence is the **framework** (FR1–FR4), which provides structural scaffolding. CDR3 is the most variable and usually the most important for binding specificity. When designing nanobodies, you're primarily engineering these CDR loops while keeping the framework regions stable and recognizable.

### Nanobody architecture map

A nanobody sequence is organized as an alternating chain of framework and CDR regions. Under **IMGT numbering** (the standard used by the competition's filters), the layout is roughly:

```
 N-term                                                                    C-term
  │                                                                           │
  ▼                                                                           ▼
 ┌──────────┬──────────┬──────────┬──────────┬───────────────┬──────────────┬────────┐
 │   FR1    │   CDR1   │   FR2    │   CDR2   │      FR3      │    CDR3      │  FR4   │
 │ pos 1-26 │ pos 27-38│ pos 39-55│ pos 56-65│  pos 66-104   │ pos 105-117  │118-128 │
 └──────────┴──────────┴──────────┴──────────┴───────────────┴──────────────┴────────┘
  ▲                                                            ▲
  │ Structural scaffold — keep                                 │ Main binding
  │ "nanobody-like" for filters                                │ surface — where
  │                                                            │ your creativity
  │ Contains conserved Cys23                                   │ matters most
  │ and hallmark positions                                     │
  │                                                            │ Longest & most
  │                                                            │ variable region
  ▼                                                            ▼
```

**Approximate residue counts for a typical ~120-residue nanobody:** FR1 (26 residues) + CDR1 (up to 12) + FR2 (17) + CDR2 (up to 10) + FR3 (39) + CDR3 (up to 13) + FR4 (11). CDR3 can be much longer in nanobodies than in conventional antibodies, some VHH CDR3 loops are 20+ residues, which is one of the features that makes nanobodies special at reaching recessed epitopes.

The competition's diversity filter checks mutations *within CDR regions specifically*: at least 2 mutations across all CDRs combined, and at least 1 in CDR3. So your designs need meaningful variation in the CDR regions described above, not just in the framework.

### Framework Reference: Conserved Positions & VHH Hallmarks

The framework regions are not arbitrary — specific positions carry conserved residues that define the nanobody fold. The nativeness and developability filters will reject sequences that deviate too far from these patterns, so it helps to know the "canonical" residues your designs should broadly respect.

**Universally conserved residues (present in virtually all functional nanobodies):**

| IMGT Position | Region | Residue | Role |
|---|---|---|---|
| 23 | FR1 | **Cys (C)** | Forms the canonical disulfide bond with Cys104; critical for fold stability |
| 41 | FR2 | **Trp (W)** | Buried tryptophan that anchors the hydrophobic core |
| 104 | FR3 | **Cys (C)** | Disulfide partner to Cys23; your `min_cysteines` config requires at least these |
| 118 | FR4 | **Phe (F)** or **Trp (W)** | Conserved J-region residue; Trp→Arg substitution occasionally seen in VHH |

**VHH hallmark residues (what makes nanobodies different from conventional VH domains):**

In normal antibodies, the heavy chain (VH) pairs with a light chain (VL) via a hydrophobic interface in FR2. Nanobodies don't have a light chain, so they've evolved hydrophilic substitutions at four FR2 positions to stay soluble on their own. These are called the **hallmark residues**:

| IMGT Position | Conventional VH | VHH (Nanobody) | Why It Matters |
|---|---|---|---|
| 42 | Val (V) | **Phe (F)** or **Tyr (Y)** | Replaces hydrophobic VL-interface residue; Phe is more common |
| 49 | Gly (G) | **Glu (E)** or **Gln (Q)** | Charged/polar residue increases solubility |
| 50 | Leu (L) | **Arg (R)** | Positively charged; helps compensate for missing light chain |
| 52 | Trp (W) | **Gly (G)**, **Phe (F)**, or **Leu (L)** | Variable position; most common motif is FERG at (42,49,50,52) |

The most common hallmark motif is **FERG** (Phe-Glu-Arg-Gly at positions 42/49/50/52). Other frequently observed motifs include FERF, YQRL, and YERW. The nativeness filter (IgBLAST-based VHH nativeness ≥ 0.80) is essentially checking that your FR2 and overall framework "looks like a real camelid VHH" — having appropriate hallmark residues is a major part of passing that check.

**Non-canonical cysteines (optional but common):**

Many natural nanobodies have an *extra* disulfide bond connecting CDR3 back to CDR1 or FR2. This stabilizes the long CDR3 loop in a specific conformation:

| Species Tendency | Extra Cys Position | Bonds To | Effect |
|---|---|---|---|
| Camels, dromedaries | CDR1 (pos ~38) | CDR3 | Pins CDR3 into a "kinked" conformation |
| Llamas, alpacas | FR2 (pos ~55) | CDR3 | Similar stabilizing role |

These extra cysteines aren't required by the competition, but they can improve structural confidence metrics (pLDDT, refolding RMSD) and are a sign of a well-designed nanobody. If `min_cysteines > 2` in the config, you'll need to include them.

> ⚠️ **A note on using these as templates:** These reference positions are guideposts, not necessarily hard requirements. The strongest approach is to use BoltzGen's structure conditioning to fix the framework backbone geometry while letting the model freely explore CDR sequences. That way, the framework naturally adopts nanobody-like features while the binding loops are optimized for your specific target. If you hard-code every conserved position, you'll pass the nativeness filter easily but might over-constrain the search space and miss creative solutions.

**Cysteines** (the amino acid `C`) form **disulfide bonds** — covalent links that stabilize the 3D fold. As covered above, the canonical Cys23–Cys104 bond is essentially mandatory, and additional non-canonical cysteines linking CDR3 to CDR1 or FR2 are common. The competition enforces a minimum cysteine count per sequence and, when multiple cysteines are required, checks that at least one pair has plausible sequence separation for a disulfide bond.

**Signal peptides** are short N-terminal sequences that cells use to route proteins for secretion. They are an artifact of natural biology and can sometimes appear on natural sequences deposited on publicly available databases, but should not appear in your designed sequences, the validator will reject them.

---

## Scoring

Your designs are **not** scored by a single number. Instead, the modified BoltzGen pipeline computes multiple metrics that serve as proxies for binding quality and structural confidence. All miners' submissions are then **ranked per metric**, and your final score is the **sum of your ranks across all metrics**. The lowest total rank wins (like golf scoring).

This ranking system has a critical strategic implication: **you need to be consistently good across every metric, not exceptional in one and terrible in another.** A design that ranks 5th on five metrics (total = 25) beats one that ranks 1st on four metrics but 100th on one (total = 104).

### The Metrics
 
Here are the exact metrics from `config/boltzgen_config.yaml` that determine your rank. Each metric is ranked independently across all miners, and the ranks are summed. The "mode" column tells you which direction wins.
 
**Structural confidence metrics** — Does the model believe your nanobody folds correctly and binds the target?
 
| Metric | Mode | What It Measures | Plain English |
|---|---|---|---|
| `design_iiptm` | **max** | Interface predicted TM-score for the designed nanobody | How confident the model is that your nanobody's binding interface is structurally correct. This is a key metric, as it directly reflects whether the predicted nanobody-target contact surface is trustworthy. Range 0–1; values ≥ 0.8 are strong. |
| `design_ptm` | **max** | Predicted TM-score for the design overall | How confident the model is that your nanobody folds into the predicted 3D shape, period. A design might bind well but fold poorly in isolation — this catches that. Range 0–1. |
| `design_to_target_iptm` | **max** | Interface pTM between design and target | How confident the model is that the nanobody and target actually form a complex together. While `design_iiptm` focuses on the nanobody side of the interface, this measures the joint confidence across both molecules. |
| `min_design_to_target_pae` | **min** | Minimum predicted aligned error (PAE) from design to target | The lowest positional uncertainty between any nanobody residue and the target. PAE is measured in angstroms (Å) — lower means the model is more certain about where the nanobody sits relative to the target. Values < 3 Å are strong |
| `interaction_pae` | **min** | Average PAE across the full interaction interface | Like the above, but averaged over the entire interface rather than just the best point. Catches designs where one residue looks great but the rest of the interface is uncertain. Lower is better. |
 
**Interaction quality metrics** — How strong and specific is the predicted physical interaction?
 
These are computed using [PLIP](https://plip-tool.biotec.tu-dresden.de/) (Protein-Ligand Interaction Profiler) on the **refolded** structure — meaning your designed sequence is first folded from scratch by Boltz-2, then the interactions are measured. This is a critical reality check: interactions that only exist in the designed backbone but disappear after refolding are not real.
 
| Metric | Mode | What It Measures | Plain English |
|---|---|---|---|
| `plip_hbonds_refolded` | **max** | Hydrogen bonds at the interface (refolded) | Number of predicted hydrogen bonds between your nanobody and the target. H-bonds are the bread and butter of specific molecular recognition: more generally means more specific, tighter binding. |
| `plip_saltbridge_refolded` | **max** | Salt bridges at the interface (refolded) | Number of electrostatic interactions between oppositely charged residues (e.g., Arg⁺ to Glu⁻) at the interface. These contribute significant binding energy but are less common than H-bonds. Even one or two is good. |
| `delta_sasa_refolded` | **max** | Change in solvent-accessible surface area upon binding (refolded) | How much surface area gets "buried" (hidden from water) when the nanobody and target come together. Measured in Å². Larger burial generally correlates with stronger binding. Typical antibody-antigen interfaces bury 600–900 Å² per chain. |
 
**Developability metrics** — Would this sequence cause problems if someone tried to make it as a real drug?
 
| Metric | Mode | What It Measures | Plain English |
|---|---|---|---|
| `liability_score` | **min** | Overall sequence liability score | Aggregates the severity of chemical "liabilities" — sequence motifs known to cause problems in real drug development. These include deamidation-prone motifs (NG, NS, NA), oxidation-prone sites (exposed Met/Cys), isomerization hotspots (DG, DS), cleavage sites (DP), and N-glycosylation motifs (NxS/T). Lower score = cleaner sequence with fewer problematic motifs. |
| `liability_num_violations` | **min** | Count of distinct liability violations | The raw number of liability motifs detected. While `liability_score` weights violations by severity, this metric just counts them. A sequence with zero violations is ideal but rare. The goal is to minimize them relative to other miners' designs. |

### What this means strategically
 
Notice the mix: 5 confidence metrics, 3 interaction metrics, and 2 developability metrics. You're being ranked on 10 dimensions simultaneously. A few implications:
 
The **confidence metrics** (iiptm, ptm, iptm, PAE) are somewhat correlated — a well-designed nanobody that folds cleanly will tend to score well on all of them. But they're not identical: you can have high pTM (good fold) with poor interaction PAE (uncertain binding pose), or vice versa.
 
The **interaction metrics** (H-bonds, salt bridges, buried surface area) reward designs that make extensive, specific contacts with the target. Designs that "float near" the target surface with weak van der Waals contacts will score poorly here even if the confidence metrics look okay.
 
The **liability metrics** are the easiest to control directly. Avoid known problematic sequence motifs in your CDR designs. This is one area where simple post-hoc filtering (scan for NG, DG, DS, DP, NxS/T motifs and reject or mutate them) can give you a reliable edge without any structural modeling.

### Multi-Target, Multi-Sequence Scoring

If the config specifies multiple targets, your per-sequence score is averaged across targets first. If you submit multiple sequences, scores are then aggregated (averaged, then summed) into a single `final_nanobody_score`. This means every sequence you submit matters, one bad sequence drags down your total.

---

## The Two Filter Gauntlets

Your submission must pass through two sets of filters *before* it even gets scored. Failing any filter rejects your **entire submission** for that epoch — not just the offending sequence. Understanding these filters is non-negotiable.

### Filter Layer 1: Sequence Validity (On-Chain)

These are fast, sequence-level checks implemented in the [NOVA validator](https://github.com/metanova-labs/nova/blob/nanobodies/neurons/validator/nanobody_validity.py). They run immediately on your submission.

**Format requirements:**

- Submit exactly `num_sequences` sequences (check `config.yaml`). Too few → rejected. Too many → only the first N are kept.
- Only the 20 standard amino acid characters are allowed (`ACDEFGHIKLMNPQRSTVWY`). Anything else → rejected.
- No duplicate sequences within your batch (compared after normalization/hashing).

**Length constraints:**

- Each sequence must be between `min_sequence_length` and `max_sequence_length` (typically 90–150 residues).

**Repetition limits (catch degenerate sequences):**

- **Homopolymer runs:** No single amino acid can repeat more than `max_homopolymer_run` times in a row (e.g., `AAAAAAAA` would fail).
- **Di-repeat pairs:** Repeated two-character motifs like `GSGSGSGS` are capped at `max_di_repeat_pairs`.

**Cysteine requirements:**

- Minimum number of cysteines per sequence (`min_cysteines`, typically ≥ 1).
- If `min_cysteines > 1`, at least one sequence must have a plausible disulfide pair — two cysteines separated by a distance within `[cys_pair_min_separation, cys_pair_max_separation]` residues in the sequence.

**Signal peptide rejection:**

- If enabled, a heuristic checks the N-terminal region of each sequence for high hydrophobicity (a hallmark of signal peptides). Sequences that look like signal peptides are rejected.

**Global uniqueness:**

- Every sequence must be globally unique for each target — you cannot resubmit sequences that have been previously submitted (by anyone) for the same target protein. This is checked against a HuggingFace-hosted registry.

### Filter Layer 2: Biological Plausibility (NOVA Nanobody Filter)

After passing Layer 1, your sequences are evaluated by a more sophisticated pipeline ([NOVA-nanobody-filter](https://github.com/frankji-groundcontrol/NOVA-nanobody-filter)) that enforces biological realism. The three filters are applied in order with early termination.

**Step 1 — Diversity Filter:**

This filter ensures that your sequences are meaningfully different from each other and from previously successful designs.

*Against historical top submissions:*
- Your sequences are compared against the **current top ~50 leaderboard sequences** for the target. If your design is too similar (Jaccard similarity ≥ 0.9 using MinHash on k-mers), it is rejected.
- The strategic implication: you cannot simply copy-paste a known winning sequence with a single tweak. The system demands genuine novelty relative to what is already on top.
- Previously submitted sequences and their respective scores will be available at our [Submission Archive HuggingFace dataset](https://huggingface.co/datasets/Metanova/Submission-Archive/tree/main), as f"{target}_nanobodies.csv". Overall top 50 designs are determined by the algorithm defined in `utils.minmax_weighted_rank`.

*Within your batch: (we won't use this now, but will in the future)*
- **MMseqs2 clustering:** Sequences sharing ≥ 98% global identity are considered near-duplicates and removed. (MMseqs2 is a standard bioinformatics tool for ultra-fast sequence comparison.)
- **CDR mutation requirements:** At least 2 mutations across all CDRs combined, and at least 1 mutation in CDR3 specifically. This prevents you from submitting trivial variants of the same design.

**Step 2 — Nativeness Filter:**

This checks that your sequence actually "looks like" a real nanobody.

- **IMGT Numbering:** Your sequence must be successfully numbered under the IMGT scheme (a standard antibody numbering convention) using the `abnumber` tool. If the tool can't assign numbers, your sequence doesn't have recognizable antibody structure.
- **Nativeness score ≥ 0.80:** Computed via IgBLAST alignment against known camelid VHH sequences. Measures how closely your design resembles real nanobodies found in nature.
- **Humanness score ≥ 0.75:** Computed via IgBLAST alignment against human antibody framework sequences. In real drug development, nanobodies need to be "humanized" so the patient's immune system doesn't attack them. A high humanness score means your design already has human-compatible framework regions.

**Step 3 — Developability Filter (TNP Red Regions):**

TNP (Therapeutic Nanobody Profiler) checks whether your nanobody would be viable as an actual therapeutic. Any property falling into a "Red Region" (biologically problematic range) rejects the sequence.

| Property | Valid Range | Reject If |
|---|---|---|
| Total CDR length | 20–39 | Outside range |
| CDR3 length | 5–23 | Outside range |
| CDR3 compactness | 0.56–1.61 | Outside range |
| Surface hydrophobic patches | 73.4–155.47 | Outside range |
| Positive charge patches | ≤ 1.18 | Above threshold |
| Negative charge patches | ≤ 1.88 | Above threshold |

Intuitively: CDRs that are too long or too short are problematic; surfaces that are too hydrophobic or too charged will cause aggregation, poor solubility, or immunogenicity in practice.

**IMPORTANT**: Make sure your sequence can be processed by all filters. Sequence that cause any of the filters to crash will be rejected.

---

## Strategic Framework

### 1. Use BoltzGen as Your Primary Design Tool

The scoring pipeline uses a modified BoltzGen, so there is a natural alignment advantage to generating your candidates with BoltzGen itself. The [official repo](https://github.com/HannesStark/boltzgen) has detailed documentation.

> ⚠️ **IMPORTANT:** The BoltzGen implementation included in the NOVA repo is customized to work as a scoring engine instead of a generative pipeline. You will use the original implementation to create your designs. As a reference, the features available in our implementation are in sync until commit db2e5ff0b9745acb8c73460c49695550a7235c21 from the source repo. Open an issue on the NOVA repo or message the team on Discord to request syncing new features that you find relevant.

The basic workflow is:

1. **Create a design spec YAML** pointing to your target structure (usually a PDB file or identifier).
2. **Run BoltzGen** with the `nanobody-anything` protocol: it will generate candidate sequences via diffusion, optionally redesign them via inverse folding, fold them with Boltz-2, and rank them.
3. **Extract the top candidates** from the output CSV (`final_designs_metrics_<budget>.csv`).

Key BoltzGen flags to experiment with:

- `--budget N` — Controls how many designs survive into the final diversity-optimized set. Generate more than you need, then cherry-pick.
- `--alpha` — Trade-off between quality and diversity in final selection. Default for non-peptide protocols is 0.001 (strongly quality-biased). You might increase this slightly to avoid all your designs being too similar.
- `--steps` — You can re-run specific pipeline stages (e.g., just `filtering` on new metrics or thresholds).
- `--filter_biased` — Removes amino acid composition outliers. Defaults to true for good reason.
- `--metrics_override` — Lets you set per-metric minimum thresholds. Useful for enforcing constraints that match the competition's scoring.

### 2. Also Explore ODesign and other tools

[ODesign](https://github.com/The-Institute-for-AI-Molecular-Design/ODesign) is an alternative de novo design tool. Since your submissions are scored with BoltzGen's metrics (not ODesign's), there is a risk of distribution mismatch — designs optimized for one model may not score as well on another. However, ODesign may explore regions of sequence space that BoltzGen misses, especially for difficult targets. A sound strategy is to generate candidates with both tools and rank them all through your own local BoltzGen scoring before submitting.

**IMPORTANT:** The tools listed here are suggestions to get you started. There are many open source *de novo* protein design tools that use different techniques. Some alternatives you can check out are [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion), [ProteinMPNN](https://github.com/dauparas/ProteinMPNN), [RoseTTAFold](https://github.com/RosettaCommons/RoseTTAFold), [COMBS](https://github.com/npolizzi/Combs) and others. 

### 3. Pre-Screen Everything Locally

**Do not submit sequences blind.** Before committing to the chain:

1. Run your candidates through the Layer 1 validity checks yourself. The code is open source — replicate it locally.
2. If you can set up the NOVA Nanobody Filter service locally, run Layer 2 checks as well. At minimum, verify IMGT numbering works (install `abnumber`) and estimate nativeness with IgBLAST.
3. Run BoltzGen's scoring pipeline on your final candidates to get predicted metric values. If any metric looks like an outlier (much worse than the others), that sequence will tank your rank-sum score — drop it.

### 4. Optimize for Rank-Sum, Not Any Single Metric

Because scoring is rank-based and uses the sum across all metrics, the optimal strategy is not to chase one metric to its theoretical maximum. Instead:

- **Eliminate catastrophic weaknesses.** A sequence that scores in the top 10% on ipTM but bottom 50% on refolding RMSD will get destroyed by rank-sum scoring.
- **Prefer well-rounded designs.** A design that is in the top 20% on every metric will usually outperform one that is top 1% on half and median on the other half.
- **Diversify your submissions.** If you submit multiple sequences, make them genuinely different — both to pass the diversity filter and because different designs may rank well on different metrics, smoothing out your aggregate score.

### 5. Study the Target

Each week's target is announced ~24 hours before the competition epoch starts on Monday at 3 PM UTC. Use this lead time wisely:

- **Download the target structure** (usually from the PDB — Protein Data Bank) and inspect it. Tools like PyMOL, ChimeraX, or the online Mol* viewer let you visualize the protein.
- **Identify the binding site.** BoltzGen lets you specify which residues on the target are in or near the desired binding site. If the target is a well-studied protein, literature may tell you where functional interfaces are.
- **Check for known binders.** If existing nanobodies or antibodies against this target exist in databases like SAbDab or PDB, studying them can inform your CDR design — but remember, your submission must be globally unique.

### 6. Iterate Across Epochs

The uniqueness requirement means you cannot resubmit the same sequence, but your design knowledge accumulates. After each epoch:

- **Analyze what scored well** (yours and competitors' results if available).
- **Use winning designs as starting points** — mutate CDR regions, try different framework scaffolds, or vary the binding geometry while staying above the diversity thresholds.
- **Track which sequence features correlate with high BoltzGen scores** for specific target families. Over time, you build a private dataset of design → score mappings that you can use for ML-guided optimization.

### 7. Engineering the Pipeline

As software engineers, your competitive edge is in building robust, automated pipelines:

- **Batch generation:** Generate hundreds of candidates per target, not a handful. GPU time is the bottleneck — optimize batch sizes and parallelize across cards.
- **Automated filtering:** Wire up all validity checks (Layer 1 + Layer 2) as a local screening step before submission. Reject invalid sequences before they waste your submission slot.
- **Metric-aware selection:** Build a selection algorithm that mimics rank-sum scoring across your candidate pool. Select the submission batch that minimizes expected worst-metric rank.
- **Timing:** Submissions close 10 blocks before epoch end. Don't wait until the last minute — your commit must be timelock-encrypted, uploaded to GitHub, and committed on-chain. Build in buffer time.

---

## Submission Mechanics (Quick Reference)

Your submission is a single string: `<molecules>|<sequences>`. Use `~` as a placeholder if you're only competing in nanobodies (e.g., `~|EVQLVES...,QVQLVES...`).

1. **Generate** your nanobody sequences.
2. **Format** them as a comma-separated list after the `|` delimiter.
3. **Encrypt** the payload using Bittensor Drand Timelock (baseline miner code provides this).
4. **Upload** the encrypted file to your public GitHub repo.
5. **Commit** the GitHub path on-chain via `set_commitment()`.

The validator fetches your file from GitHub, decrypts it, validates, scores, and ranks.

---

## Tooling Reference

| Tool | What It Does | Link |
|---|---|---|
| **BoltzGen** | De novo nanobody design via diffusion + inverse folding | [GitHub](https://github.com/HannesStark/boltzgen) |
| **ODesign** | Alternative biomolecular interaction design | [GitHub](https://github.com/The-Institute-for-AI-Molecular-Design/ODesign) |
| **Boltz-2** | Structure prediction + binding affinity prediction | [Paper](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1) |
| **abnumber** | IMGT/Chothia/Kabat antibody numbering | [GitHub](https://github.com/prihoda/AbNumber) |
| **IgBLAST** | Immunoglobulin sequence alignment (nativeness scoring) | [NCBI](https://ncbi.github.io/igblast/) |
| **TNP** | Therapeutic Nanobody Profiler (developability) | [GitHub](https://github.com/oxpig/TNP) |
| **MMseqs2** | Ultra-fast sequence clustering/comparison | [GitHub](https://github.com/soedinglab/MMseqs2) |
| **PyMOL / ChimeraX** | 3D protein structure visualization | [PyMOL](https://pymol.org) / [ChimeraX](https://www.cgl.ucsf.edu/chimerax/) |
| **SAbDab** | Structural Antibody Database | [Website](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/) |
| **PLIP** | Protein-Ligand Interaction Profiling | [Website](https://plip-tool.biotec.tu-dresden.de/plip-web/plip/index) |
| **RFdiffusion** | Protein design | [GitHub](https://github.com/RosettaCommons/RFdiffusion) |
| **ProteinMPNN** | Protein design | [GitHub](https://github.com/dauparas/ProteinMPNN) |
| **RoseTTAFold** | Protein structure prediction | [GitHub](https://github.com/RosettaCommons/RoseTTAFold) |
| **COMBS** | Binding site optimization | [GitHub](https://github.com/npolizzi/Combs) |

---

## TL;DR — The Winning Formula

1. **Generate many candidates** with BoltzGen (primary) and optionally other protein design tools (supplementary).
2. **Screen locally** through both filter layers before submitting. Reject anything that might fail.
3. **Rank your candidates** by simulated rank-sum scoring across all BoltzGen metrics. Pick the most well-rounded set.
4. **Ensure diversity** against the leaderboard (Jaccard < 0.9 vs top 50).
5. **Submit on time** with correct formatting. Double-check the `mol|seq` format, encryption, and GitHub upload. Baseline miner takes care of that.
6. **Learn and iterate** each epoch. Track what works, refine your pipeline, and expand your design search space.

Good luck!