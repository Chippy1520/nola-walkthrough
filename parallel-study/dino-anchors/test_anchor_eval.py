import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import anchor_eval as ae


class SelectorTests(unittest.TestCase):
    def test_no_top1_theft_and_rank3_fallback(self):
        scores = np.array([[5, 4, 1], [4, 2, 1], [1, 5, 2], [1, 4, 3], [5, 1, 0]], dtype=np.float32)
        pools, budgets, ranks, _, top1 = ae._allocate(scores, 2, 1.0)
        np.testing.assert_array_equal(top1, [0, 0, 1, 1, 0])
        np.testing.assert_array_equal(budgets, [2, 2, 1])
        np.testing.assert_array_equal(ranks, [0, 0, 2])
        self.assertEqual(len(set(np.concatenate(pools))), 5)
        np.testing.assert_array_equal(pools[2], [0])

    def test_confidence_pool_cap_and_no_fallback_for_sparse_class(self):
        scores = np.array([[1, 0]] * 8 + [[0, 1]], dtype=np.float32)
        pools, budgets, ranks, _, _ = ae._allocate(scores, 2, 1.0)
        np.testing.assert_array_equal(budgets, [2, 1])
        np.testing.assert_array_equal(ranks, [0, 0])
        np.testing.assert_array_equal(pools[0], np.arange(6))
        np.testing.assert_array_equal(pools[1], [8])

    def test_farthest_and_medoid_are_different_algorithms(self):
        angles = torch.deg2rad(torch.tensor([0., 5., 10., 110., 120., 130.]))
        x = torch.stack([angles.cos(), angles.sin()], dim=1)
        far = ae._farthest(x, 2)
        med = ae._medoids(x, far)
        self.assertEqual(far.tolist(), [0, 5])
        self.assertEqual(med.tolist(), [1, 4])
        distances = 1 - x @ x.T
        self.assertLess(distances[:, med].min(dim=1).values.sum(), distances[:, far].min(dim=1).values.sum())

    def test_knn_nonself_and_cutoff_ties(self):
        x = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.]])
        labels = torch.tensor([0, 1, 0, 1])
        with patch.object(ae, 'NEIGHBORS', 1):
            agree = ae._agreement(x, labels, torch.tensor([0, 1, 3]), torch.tensor([0, 0, 1]))
        np.testing.assert_array_equal(agree, [0, 1, 0])

    def test_singleton_and_missing_class_are_valid(self):
        result = ae._select(np.array([[1., 0.]], dtype=np.float32), np.ones((1, 2), dtype=np.float32),
                            16, 1, torch.device('cpu'))
        selected, pools, budgets, ranks, _, _, agreement = result
        np.testing.assert_array_equal(budgets, [1, 0])
        np.testing.assert_array_equal(ranks, [0, -1])
        np.testing.assert_array_equal(agreement, [0.])
        for arm in ae.ARMS:
            self.assertEqual([p.tolist() for p in selected[arm]], [[0], []])


class EvaluationTests(unittest.TestCase):
    def run_study(self, output, flip=False):
        rng = np.random.default_rng(8)
        labels = np.repeat(np.arange(2), 36)
        train = rng.normal(size=(len(labels), 4)).astype(np.float32) * .3
        train[:, 0] += 2 * labels - 1
        scores = rng.uniform(0, .1, (len(labels), 2)).astype(np.float32)
        scores[np.arange(len(labels)), labels] += .4
        # Same ID with a conflicting second row: keep the first aligned row.
        ids = [f'train-{i:03}' for i in range(len(labels))]
        ids.append(ids[0])
        train = np.vstack([train, -train[0]])
        scores = np.vstack([scores, scores[0, ::-1]])
        labels = np.append(labels, 1)
        val = torch.tensor([[-1., 0., 0., 0.], [1., 0., 0., 0.]])
        truth = 1 - labels if flip else labels
        val_y = torch.tensor([1, 0] if flip else [0, 1])
        return ae.evaluate_dataset('synthetic', torch.from_numpy(scores), train, val, truth, val_y,
                                   ids, ['val-a', 'val-b'], ['a', 'b'], output,
                                   {'audit_train_fine_y': np.arange(len(ids)) % 3,
                                    'val_fine_y': [1, 2], 'clip_logit_scale': 20., 'provenance': 'fixture'})

    def test_real_teacher_all_arms_audit_isolation_and_reports(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.run_study(root / 'original')
            b = self.run_study(root / 'flipped', flip=True)
            self.assertEqual(a['teacher']['epochs'], 50)
            self.assertEqual(a['n_train_original'], 73)
            self.assertEqual(a['n_train_unique'], 72)
            self.assertEqual(a['realized_per_class'], [16, 16])
            self.assertEqual(a['candidate_counts'], [36, 36])
            self.assertEqual(set(a['arms']), set(ae.ARMS))
            for arm in ae.ARMS:
                self.assertEqual(a['arms'][arm]['anchor_count'], 32)
                self.assertEqual(a['arms'][arm]['anchor_precision'], 1.)
                self.assertEqual(a['arms'][arm]['val_accuracy'], 1.)
                self.assertEqual(b['arms'][arm]['anchor_precision'], 0.)
                self.assertEqual(b['arms'][arm]['val_accuracy'], 0.)
            with (root / 'original' / 'summary.json').open() as f:
                self.assertEqual(json.load(f), a)
            for filename, keys, num_rows in [('predictions.csv', ('arm', 'val_id', 'pred_class', 'prob'), 8),
                                             ('anchor_audit.csv', ('arm', 'train_id', 'pseudo_class', 'train_row'), 128)]:
                def read(which):
                    with (root / which / filename).open() as f:
                        return [tuple(row[key] for key in keys) for row in csv.DictReader(f)]
                first = read('original')
                self.assertEqual(first, read('flipped'))
                self.assertEqual(len(first), num_rows)
                self.assertEqual(len({row[:2] for row in first}), num_rows)
            self.assertTrue((root / 'original' / 'by_class.csv').is_file())
            self.assertTrue((root / 'original' / 'by_subtype.csv').is_file())
            print(f"Validated all four 50-epoch teachers on {a['device']}")

    def test_reject_split_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                ae.evaluate_dataset('bad', [[1.]], [[1.]], [[1.]], [0], [0], ['same'], ['same'], ['a'], tmp)


if __name__ == '__main__':
    unittest.main()
