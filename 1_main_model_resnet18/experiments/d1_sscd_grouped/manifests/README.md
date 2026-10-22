# Frozen split manifests

These lightweight files are kept in git. Bulk raw images and crops live in the ignored `../data/` directory. `split_manifest.csv` maps every raw image to its source group, split block and split. `d1_split.portable.csv` maps every crop to its split using paths relative to this experiment. The active training manifest is `../data/d1_sscd_grouped/work/d1_split.csv` and uses absolute paths.

Migration changed path strings, not assignments. Rebase portable crop paths before training on another machine and record the new absolute-manifest checksum.
