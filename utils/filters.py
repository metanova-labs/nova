"""Molecular plausibility filters. Thresholds come from config.yaml.

Props computes each quantity once and lazily; FILTERS is ordered by measured
cost-per-rejection so evaluation short-circuits cheaply. Order affects speed only.
"""
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, List, NamedTuple

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, GraphDescriptors, Lipinski, rdMolDescriptors

METALS = frozenset(
    "Fe Zn Cu Mg Ca Mn Mo Ni Co Pt Au Hg Pd Ru W V Cr Cd Pb As Sb Bi Sn Al Li Na K".split()
)
HALOGENS = frozenset("F Cl Br I".split())

_SMARTS = {
    "nitrile": "[CX2]#[NX1]",
    "azide": "[NX1]~[NX2]~[NX2,NX3]",
    "isocyanide": "[CX1-]#[NX2+]",
    "allene": "[CX3]=[CX2]=[CX3]",
    "hydroxyl": "[OX2H]",
}
_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in _SMARTS.items()}

class Props:
    def __init__(self, mol):
        self.mol = mol

    @cached_property
    def _scan(self):
        counts, aromatic, aromatic_n, aromatic_c = {}, 0, 0, 0
        for atom in self.mol.GetAtoms():
            symbol = atom.GetSymbol()
            counts[symbol] = counts.get(symbol, 0) + 1
            if atom.GetIsAromatic():
                aromatic += 1
                if symbol == "N":
                    aromatic_n += 1
                elif symbol == "C":
                    aromatic_c += 1
        return counts, aromatic, aromatic_n, aromatic_c

    def element(self, symbol) -> int:
        return self._scan[0].get(symbol, 0)

    @cached_property
    def heavy_atoms(self): return self.mol.GetNumHeavyAtoms()
    @cached_property
    def aromatic_atoms(self): return self._scan[1]
    @cached_property
    def aromatic_nitrogen(self): return self._scan[2]
    @cached_property
    def aromatic_carbon(self): return self._scan[3]
    @cached_property
    def heteroatoms(self): return self.heavy_atoms - self.element("C")
    @cached_property
    def aromatic_fraction(self): return self.aromatic_atoms / max(self.heavy_atoms, 1)
    @cached_property
    def heteroatom_fraction(self): return self.heteroatoms / max(self.heavy_atoms, 1)
    @cached_property
    def metals(self):
        return sum(n for sym, n in self._scan[0].items() if sym in METALS)
    @cached_property
    def halogens(self):
        return sum(n for sym, n in self._scan[0].items() if sym in HALOGENS)
    @cached_property
    def largest_ring(self):
        return max((len(r) for r in self.mol.GetRingInfo().AtomRings()), default=0)
    @cached_property
    def triple_bonds(self):
        return sum(1 for b in self.mol.GetBonds() if b.GetBondType() == Chem.BondType.TRIPLE)

    def group(self, key) -> int:
        return len(self.mol.GetSubstructMatches(_PATTERNS[key]))

    @cached_property
    def molecular_weight(self): return Descriptors.MolWt(self.mol)
    @cached_property
    def rings(self): return Descriptors.RingCount(self.mol)
    @cached_property
    def aromatic_rings(self): return Descriptors.NumAromaticRings(self.mol)
    @cached_property
    def aliphatic_rings(self): return Descriptors.NumAliphaticRings(self.mol)
    @cached_property
    def fsp3(self): return rdMolDescriptors.CalcFractionCSP3(self.mol)
    @cached_property
    def spiro_atoms(self): return rdMolDescriptors.CalcNumSpiroAtoms(self.mol)
    @cached_property
    def amide_bonds(self): return rdMolDescriptors.CalcNumAmideBonds(self.mol)
    @cached_property
    def hbd(self): return Lipinski.NumHDonors(self.mol)
    @cached_property
    def hba(self): return Lipinski.NumHAcceptors(self.mol)
    @cached_property
    def rotatable_bonds(self): return Descriptors.NumRotatableBonds(self.mol)
    @cached_property
    def stereocenters(self):
        return len(Chem.FindMolChiralCenters(self.mol, includeUnassigned=True))
    @cached_property
    def kappa1(self): return GraphDescriptors.Kappa1(self.mol)
    @cached_property
    def kappa2(self): return GraphDescriptors.Kappa2(self.mol)
    @cached_property
    def kappa3(self): return GraphDescriptors.Kappa3(self.mol)
    @cached_property
    def bertz_ct(self): return GraphDescriptors.BertzCT(self.mol)
    @cached_property
    def logp(self): return Crippen.MolLogP(self.mol)


class Filter(NamedTuple):
    id: int
    name: str
    tier: str
    keys: tuple
    value: Callable[[Props], float]
    ok: Callable[[float, dict], bool]
    limit: Callable[[dict], str]


@dataclass(frozen=True)
class Failure:
    filter: Filter
    value: float
    limit: str

    def __str__(self):
        return f"{self.filter.name} = {self.value:.4g} (limit {self.limit})"


def _max(fid, name, tier, key, value):
    return Filter(fid, name, tier, (key,), value,
                  lambda v, t, k=key: v <= t[k],
                  lambda t, k=key: f"<= {t[k]}")


def _min(fid, name, tier, key, value):
    return Filter(fid, name, tier, (key,), value,
                  lambda v, t, k=key: v >= t[k],
                  lambda t, k=key: f">= {t[k]}")


def _zero(fid, name, value):
    return Filter(fid, name, "exclusion", (), value, lambda v, t: v == 0, lambda t: "0")


FILTERS: List[Filter] = [
    _max(120, "kier kappa1", "inferred", "max_kappa1", lambda p: p.kappa1),
    _max(6, "molecular weight upper", "inferred", "max_molecular_weight", lambda p: p.molecular_weight),
    _max(1, "heavy atom count", "hard", "max_heavy_atoms", lambda p: p.heavy_atoms),
    _min(4, "minimum heavy atoms", "hard", "min_heavy_atoms", lambda p: p.heavy_atoms),
    _min(7, "molecular weight lower", "inferred", "min_molecular_weight", lambda p: p.molecular_weight),
    _max(28, "aromatic ring count", "inferred", "max_aromatic_rings", lambda p: p.aromatic_rings),
    _max(29, "aliphatic ring count", "inferred", "max_aliphatic_rings", lambda p: p.aliphatic_rings),
    _min(12, "ring count minimum", "inferred", "min_rings", lambda p: p.rings),
    _max(21, "fsp3", "soft", "max_fsp3", lambda p: p.fsp3),
    _max(24, "spiro atoms", "inferred", "max_spiro_atoms", lambda p: p.spiro_atoms),
    _max(14, "macrocycle", "exclusion", "max_ring_size", lambda p: p.largest_ring),
    _max(97, "aromatic nitrogen", "inferred", "max_aromatic_nitrogen", lambda p: p.aromatic_nitrogen),
    _max(67, "aromatic atom fraction", "inferred", "max_aromatic_fraction", lambda p: p.aromatic_fraction),
    _max(102, "aromatic carbon", "inferred", "max_aromatic_carbon", lambda p: p.aromatic_carbon),
    _max(41, "total halogens", "inferred", "max_halogens", lambda p: p.halogens),
    _max(48, "nitrogen count", "inferred", "max_nitrogen", lambda p: p.element("N")),
    _max(49, "oxygen count", "inferred", "max_oxygen", lambda p: p.element("O")),
    _max(50, "sulfur count", "inferred", "max_sulfur", lambda p: p.element("S")),
    _max(44, "phosphorus count", "inferred", "max_phosphorus", lambda p: p.element("P")),
    _max(42, "iodine count", "inferred", "max_iodine", lambda p: p.element("I")),
    _max(43, "bromine count", "inferred", "max_bromine", lambda p: p.element("Br")),
    _max(18, "fluorine count", "exclusion", "max_fluorine", lambda p: p.element("F")),
    _zero(15, "metal atoms", lambda p: p.metals),
    _zero(45, "boron", lambda p: p.element("B")),
    _zero(46, "silicon", lambda p: p.element("Si")),
    _zero(47, "selenium", lambda p: p.element("Se")),
    Filter(68, "heteroatom fraction", "inferred",
           ("min_heteroatom_fraction", "max_heteroatom_fraction"),
           lambda p: p.heteroatom_fraction,
           lambda v, t: t["min_heteroatom_fraction"] <= v <= t["max_heteroatom_fraction"],
           lambda t: f'{t["min_heteroatom_fraction"]}-{t["max_heteroatom_fraction"]}'),
    _max(8, "h-bond donors", "inferred", "max_hbd", lambda p: p.hbd),
    _max(33, "amide bond count", "inferred", "max_amide_bonds", lambda p: p.amide_bonds),
    _max(85, "nitrile count", "inferred", "max_nitrile", lambda p: p.group("nitrile")),
    _max(90, "hydroxyl count", "inferred", "max_hydroxyl", lambda p: p.group("hydroxyl")),
    _zero(86, "azide", lambda p: p.group("azide")),
    _zero(87, "isocyanide", lambda p: p.group("isocyanide")),
    _zero(88, "allene", lambda p: p.group("allene")),
    _max(32, "triple bond count", "inferred", "max_triple_bonds", lambda p: p.triple_bonds),
    _min(11, "rotatable bonds lower", "inferred", "min_rotatable_bonds", lambda p: p.rotatable_bonds),
    _max(11, "rotatable bonds upper", "inferred", "max_rotatable_bonds", lambda p: p.rotatable_bonds),
    _max(59, "bertzCT complexity", "inferred", "max_bertz_ct", lambda p: p.bertz_ct),
    _max(9, "h-bond acceptors", "inferred", "max_hba", lambda p: p.hba),
    _max(121, "kier kappa2", "inferred", "max_kappa2", lambda p: p.kappa2),
    _max(122, "kier kappa3", "inferred", "max_kappa3", lambda p: p.kappa3),
    _max(51, "stereocenter count", "inferred", "max_stereocenters", lambda p: p.stereocenters),
    _max(10, "logP (crippen)", "inferred", "max_logp", lambda p: p.logp),
]


REQUIRED_KEYS = frozenset(k for f in FILTERS for k in f.keys)


def validate_thresholds(config: dict) -> dict:
    """Every threshold must come from config. Missing ones are a configuration
    fault, not something to guess at."""
    if not config:
        raise ValueError(
            f"No filter thresholds configured. molecule_validation.filters must "
            f"define all {len(REQUIRED_KEYS)} keys."
        )
    missing = REQUIRED_KEYS - set(config)
    if missing:
        raise ValueError(f"Missing filter thresholds: {', '.join(sorted(missing))}")
    return config


def check(mol, config: dict, first_only: bool = True) -> List[Failure]:
    """Failures for a molecule; first_only stops at the first."""
    limits = validate_thresholds(config)
    props = Props(mol)
    failures = []
    for f in FILTERS:
        value = f.value(props)
        if not f.ok(value, limits):
            failures.append(Failure(f, value, f.limit(limits)))
            if first_only:
                break
    return failures


def passes(mol, config: dict) -> bool:
    return not check(mol, config, first_only=True)
