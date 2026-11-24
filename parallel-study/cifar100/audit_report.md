# CIFAR-100 coarse-label audit

**Result: passed.** Verified all 60,000 image-aligned fine/coarse label pairs from the original Python archive. The official training and test sets independently yield the same mapping: 100 fine classes, one parent per fine class, and exactly 5 fine subtypes for each of 20 coarse classes.

| Partition | Images | Per fine subtype | Per coarse class |
|---|---:|---:|---:|
| official train | 50,000 | 500 | 2,500 |
| fit | 45,000 | 450 | 2,250 |
| validation | 5,000 | 50 | 250 |
| official test | 10,000 | 100 | 500 |

## Split and provenance

- **Seed:** 42. Within each fine class, rank official train rows by SHA-256 of ASCII cifar100-val-v1:<seed>:<row_id>, then row ID; hold out the first 50. Sort the resulting partitions by row ID.
- Fit and validation are disjoint and cover every original train row exactly once. Same-seed regeneration matches exactly; a different seed changes membership. Test is retained as the original official partition and has a separate row-ID namespace.
- Index hashes use sorted little-endian signed int64 bytes, without a header.
  - Fit SHA-256: `e262d10b6df1d22df4bb18fd1c77c146bcfd81b99dcec077d5976ab4e272a5a3`
  - Validation SHA-256: `3704fd5869ef075ed9e9ef2949506bcf4cfda2a08761683ba3ea78620dbf74b0`
- Archive obtained from: https://storage.googleapis.com/tensorflow/tf-keras-datasets/cifar-100-python.tar.gz
- MD5 (`eb9058c3a382ffc7106e4002c42a8d85`) matches the [publisher](https://www.cs.toronto.edu/~kriz/cifar.html); SHA-256 (`85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7`) matches [Keras](https://github.com/keras-team/keras/blob/master/keras/src/datasets/cifar100.py). All three member MD5s match [TorchVision](https://github.com/pytorch/vision/blob/main/torchvision/datasets/cifar.py).
- CPU only, Python 3.12.3, NumPy 1.26.4. No model or GPU was used.

## Coarse class → fine subtypes (official IDs and names)

| Coarse ID | Coarse name | Fine IDs and names |
|---:|---|---|
| 0 | `aquatic_mammals` | 4: `beaver`, 30: `dolphin`, 55: `otter`, 72: `seal`, 95: `whale` |
| 1 | `fish` | 1: `aquarium_fish`, 32: `flatfish`, 67: `ray`, 73: `shark`, 91: `trout` |
| 2 | `flowers` | 54: `orchid`, 62: `poppy`, 70: `rose`, 82: `sunflower`, 92: `tulip` |
| 3 | `food_containers` | 9: `bottle`, 10: `bowl`, 16: `can`, 28: `cup`, 61: `plate` |
| 4 | `fruit_and_vegetables` | 0: `apple`, 51: `mushroom`, 53: `orange`, 57: `pear`, 83: `sweet_pepper` |
| 5 | `household_electrical_devices` | 22: `clock`, 39: `keyboard`, 40: `lamp`, 86: `telephone`, 87: `television` |
| 6 | `household_furniture` | 5: `bed`, 20: `chair`, 25: `couch`, 84: `table`, 94: `wardrobe` |
| 7 | `insects` | 6: `bee`, 7: `beetle`, 14: `butterfly`, 18: `caterpillar`, 24: `cockroach` |
| 8 | `large_carnivores` | 3: `bear`, 42: `leopard`, 43: `lion`, 88: `tiger`, 97: `wolf` |
| 9 | `large_man-made_outdoor_things` | 12: `bridge`, 17: `castle`, 37: `house`, 68: `road`, 76: `skyscraper` |
| 10 | `large_natural_outdoor_scenes` | 23: `cloud`, 33: `forest`, 49: `mountain`, 60: `plain`, 71: `sea` |
| 11 | `large_omnivores_and_herbivores` | 15: `camel`, 19: `cattle`, 21: `chimpanzee`, 31: `elephant`, 38: `kangaroo` |
| 12 | `medium_mammals` | 34: `fox`, 63: `porcupine`, 64: `possum`, 66: `raccoon`, 75: `skunk` |
| 13 | `non-insect_invertebrates` | 26: `crab`, 45: `lobster`, 77: `snail`, 79: `spider`, 99: `worm` |
| 14 | `people` | 2: `baby`, 11: `boy`, 35: `girl`, 46: `man`, 98: `woman` |
| 15 | `reptiles` | 27: `crocodile`, 29: `dinosaur`, 44: `lizard`, 78: `snake`, 93: `turtle` |
| 16 | `small_mammals` | 36: `hamster`, 50: `mouse`, 65: `rabbit`, 74: `shrew`, 80: `squirrel` |
| 17 | `trees` | 47: `maple_tree`, 52: `oak_tree`, 56: `palm_tree`, 59: `pine_tree`, 96: `willow_tree` |
| 18 | `vehicles_1` | 8: `bicycle`, 13: `bus`, 48: `motorcycle`, 58: `pickup_truck`, 90: `train` |
| 19 | `vehicles_2` | 41: `lawn_mower`, 69: `rocket`, 81: `streetcar`, 85: `tank`, 89: `tractor` |

## Using the subtype evaluator

Predict the 20 coarse classes only. Use fine labels for auditing, validation stratification and diagnostics, not for fitting.

```python
from subtype_eval import evaluate_subtypes
report = evaluate_subtypes(predicted_coarse_ids_or_scores, split="validation", row_ids=prediction_row_ids)
```

`row_ids` are the original 0-based indices, in prediction order. Use official train row IDs for validation and official test row IDs for test (`split="test"`). The helper rejects missing, duplicated or foreign rows. For an N×20 score array, columns must match the official coarse ID order shown above. Subtype accuracy measures whether examples of that subtype were assigned their true coarse parent.

Self-test (constructed predictions, not a model result): 4,995/5,000 validation predictions correct with all five injected errors localized to apple; 10,000/10,000 correct for the test identity check.

## Machine-readable files

- `audit_summary.json`: full counts, checksums, verification results and split configuration.
- `fine_counts.csv` and `coarse_counts.csv`: full distributions and official name/ID mapping.
- `audit_manifest.npz`: original-order label vectors, hierarchy and split indices (no images).
