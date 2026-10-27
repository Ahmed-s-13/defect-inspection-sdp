# Deployed architecture A on the off-target benchmark

calibration.json: quality alpha 0.05 (sharpness both), OOD alpha 0.08 (kNN + AoN, Bonferroni), CP alpha 0.1. The AoN regression saw the fit AND test categories (in-sample); wallpapers, static, undamaged aircraft, D5 and the user photo it never saw.

- D5 outside test (normal defects, false reject): 6.6% stopped (re-take 5.2%, re-aim 1.5%)
- macOS wallpaper patches: 15.2% pass
- random static / colours / gradients: 0.0% pass

| role | category | mode | crops | passes | re-take | re-aim |
|---|---|---|---:|---:|---:|---:|
| fit | Car | centre | 100 | 0.0% | 0.0% | 100.0% |
| fit | Car | box | 100 | 0.0% | 0.0% | 100.0% |
| fit | Human face | centre | 150 | 0.0% | 0.0% | 100.0% |
| fit | Human face | box | 150 | 0.0% | 0.0% | 100.0% |
| fit | Ladder | centre | 83 | 0.0% | 0.0% | 100.0% |
| fit | Ladder | box | 83 | 0.0% | 0.0% | 100.0% |
| fit | Person | centre | 200 | 1.0% | 0.0% | 99.0% |
| fit | Person | box | 200 | 1.0% | 0.0% | 99.0% |
| fit | Tool | centre | 150 | 2.0% | 0.7% | 97.3% |
| fit | Tool | box | 150 | 0.0% | 0.7% | 99.3% |
| fit | Truck | centre | 100 | 0.0% | 0.0% | 100.0% |
| fit | Truck | box | 100 | 0.0% | 0.0% | 100.0% |
| fit | auto_factory | centre | 100 | 0.0% | 0.0% | 100.0% |
| fit | auto_factory | box | 100 | 0.0% | 0.0% | 100.0% |
| fit | engine_room | centre | 100 | 0.0% | 0.0% | 100.0% |
| fit | engine_room | box | 100 | 2.0% | 0.0% | 98.0% |
| fit | garage/indoor | centre | 100 | 1.0% | 0.0% | 99.0% |
| fit | garage/indoor | box | 100 | 6.0% | 0.0% | 94.0% |
| fit | industrial_area | centre | 100 | 0.0% | 0.0% | 100.0% |
| fit | industrial_area | box | 100 | 2.0% | 0.0% | 98.0% |
| fit | loading_dock | centre | 100 | 1.0% | 0.0% | 99.0% |
| fit | loading_dock | box | 100 | 6.0% | 0.0% | 94.0% |
| nodamage | aircraft skin patch (no damage) | box | 1000 | 24.2% | 0.1% | 75.7% |
| nodamage | whole aircraft (no damage) | centre | 500 | 1.4% | 0.0% | 98.6% |
| test | Glove | centre | 85 | 1.2% | 0.0% | 98.8% |
| test | Glove | box | 85 | 0.0% | 0.0% | 100.0% |
| test | Helmet | centre | 200 | 0.5% | 0.0% | 99.5% |
| test | Helmet | box | 200 | 0.0% | 0.0% | 100.0% |
| test | Human hand | centre | 200 | 0.0% | 0.0% | 100.0% |
| test | Human hand | box | 200 | 1.5% | 0.0% | 98.5% |
| test | airport_terminal | centre | 100 | 0.0% | 0.0% | 100.0% |
| test | airport_terminal | box | 100 | 0.0% | 0.0% | 100.0% |
| test | hangar/indoor | centre | 100 | 1.0% | 0.0% | 99.0% |
| test | hangar/indoor | box | 100 | 2.0% | 0.0% | 98.0% |
| test | hangar/outdoor | centre | 100 | 1.0% | 0.0% | 99.0% |
| test | hangar/outdoor | box | 100 | 3.0% | 0.0% | 97.0% |
| test | helmet photo (user) | centre | 1 | 0.0% | 0.0% | 100.0% |
| test | helmet photo (user) | box | 1 | 0.0% | 0.0% | 100.0% |
| test | repair_shop | centre | 100 | 0.0% | 0.0% | 100.0% |
| test | repair_shop | box | 100 | 3.0% | 0.0% | 97.0% |
| test | runway | centre | 100 | 1.0% | 0.0% | 99.0% |
| test | runway | box | 100 | 4.0% | 1.0% | 95.0% |
| test | storage_room | centre | 100 | 0.0% | 0.0% | 100.0% |
| test | storage_room | box | 100 | 0.0% | 0.0% | 100.0% |
