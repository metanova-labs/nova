import os

NOVA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALLOWED_AAS = set("ACDEFGHIKLMNPQRSTVWY")
HYDROPHOBIC = set("AILMFWV")
CAMELID_DB_V = os.path.join(NOVA_DIR, "external_tools", "igblast", "database", "camelid_V")
HUMAN_DB_V   = os.path.join(NOVA_DIR, "external_tools", "igblast", "database", "human_V")