"""Project-wide constants."""

# Lung window for CT (HU)
HU_MIN = -1000
HU_MAX = 400

# Nodule size stratification bins in mm: micro / small / medium / large
SIZE_BINS_MM = [5.0, 10.0, 30.0]

# Consensus probability levels for 4 raters
CONSENSUS_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]

# Background instance id in instance_ids maps
BACKGROUND_ID = 0
