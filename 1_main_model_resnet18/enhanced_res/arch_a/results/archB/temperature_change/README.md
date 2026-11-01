# Family-weighted temperature deployment

Current B fits temperature with equal total weight per validation family. The model and OOD gate are unchanged; CP cutoffs were refitted on the same 2,065 accepted calibration crops.

The JSON records the before/after values; the old calibration is a historical reproducibility snapshot. See [the paired comparison](../family_temperature/README.md). Run `archB_update_temperature.py` from the model environment to reproduce the update from cached probabilities.
