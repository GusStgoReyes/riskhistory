"""Visual formatting constants for all figures."""

from dataclasses import dataclass
from matplotlib import rcParams

# Apply rcParams on import
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Helvetica"]
rcParams["font.size"] = 14
rcParams["axes.labelsize"] = 22
rcParams["axes.titlesize"] = 28
rcParams["xtick.labelsize"] = 18
rcParams["ytick.labelsize"] = 18
rcParams["legend.fontsize"] = 18

# ---------------------------------------------------------------------------
# Global
# ---------------------------------------------------------------------------
DPI = 300
DEFAULT_FIGSIZE = (8, 6)