#! /usr/bin/env python3

import unittest
import random
import numpy
from decimal import Decimal
from fractions import Fraction
import graphviz
import mock
import os
import matplotlib as mpl
# Needed for running the tests on Travis:
if os.environ.get('DISPLAY', '') == '':
    print('No display found. Using non-interactive Agg backend.')
    mpl.use('Agg')
from pytree import Node, Leaf, IncrementalStat, compute_regression, Config, FlatRegression # noqa: 402

DEFAULT_MODE = 'BIC'


def generate_dataset(intercept, coeff, size, min_x, max_x, cls=float, repeat=1):
    dataset = []
    if cls is float:
        f = cls
    else:
        def f(x): return cls('%.3f' % x)
    intercept = f(intercept)
    coeff = f(coeff)
    for _ in range(size):
        x = f(random.uniform(min_x, max_x))
        y = x*coeff + intercept
        dataset.extend([(x, y)]*repeat)
    return dataset


class IncrementalStatTest(unittest.TestCase):
    def test_basic(self):
        size = random.randint(50, 100)
        values = []
        stats = IncrementalStat()
        for _ in range(size):
            val = random.uniform(0, 100)
            stats.add(val)
            values.append(val)
        for _ in range(size-2):  # don't do the last two ones
            val = stats.last
            self.assertEqual(stats.pop(), val)
            self.assertEqual(values.pop(), val)
            self.assertAlmostEqual(numpy.mean(values), stats.mean)
            self.assertAlmostEqual(numpy.var(values),  stats.var)
            self.assertAlmostEqual(numpy.std(values),  stats.std)
            self.assertAlmostEqual(sum(values),        stats.sum)

    def test_fraction(self):
        size = random.randint(50, 100)
        values = []
        stats = IncrementalStat()
        for _ in range(size):
            val = Fraction(random.uniform(0, 100))
            stats.add(val)
            values.append(val)
        for _ in range(size-2):  # don't do the last two ones
            val = stats.last
            self.assertEqual(stats.pop(), val)
            self.assertEqual(values.pop(), val)
            self.assertEqual(numpy.mean(values), stats.mean)
            self.assertEqual(numpy.var(values),  stats.var)
            self.assertEqual(sum(values),        stats.sum)

    def test_func(self):
        def f(x): return x**2 - x + 4
        size = random.randint(50, 100)
        original_values = []
        values = []
        stats = IncrementalStat(f)
        for _ in range(size):
            val = random.uniform(0, 100)
            stats.add(val)
            original_values.append(val)
            values.append(f(val))
        for _ in range(size-2):  # don't do the last two ones
            val = stats.last
            self.assertEqual(stats.pop(), val)
            self.assertEqual(original_values.pop(), val)
            values.pop()
            self.assertAlmostEqual(numpy.mean(values), stats.mean)
            self.assertAlmostEqual(numpy.var(values),  stats.var)
            self.assertAlmostEqual(numpy.std(values),  stats.std)
            self.assertAlmostEqual(sum(values),        stats.sum)


class LeafTest(unittest.TestCase):
    def setUp(self):
        self.coeff = random.uniform(0, 100)
        self.intercept = random.uniform(0, 100)
        self.size = random.randint(50, 100)
        self.data = []
        for _ in range(self.size):
            x = random.uniform(0, 100)
            y = self.coeff * x + self.intercept
            self.data.append((x, y))
        self.config = Config(mode=DEFAULT_MODE, epsilon=1e-6)
        self.data.sort()

    def perform_tests(self, x, y, node, noisy):
        delta = 1e-10
        self.assertAlmostEqual(
            node.mean_x,   numpy.mean(x),              delta=delta)
        self.assertAlmostEqual(
            node.mean_y,   numpy.mean(y),              delta=delta)
        self.assertAlmostEqual(node.std_x,    numpy.std(
            x),               delta=delta)
        self.assertAlmostEqual(node.std_y,    numpy.std(
            y),               delta=delta)
        self.assertAlmostEqual(node.corr,     numpy.corrcoef(x, y)[
                               1, 0],  delta=delta)
        if noisy:
            delta = max(*x, *y)/100  # TODO better delta ?
        self.assertAlmostEqual(node.coeff,     self.coeff,      delta=delta)
        self.assertAlmostEqual(node.intercept, self.intercept,  delta=delta * 10)
        self.assertAlmostEqual(node.rsquared, 1,                delta=delta)
        MSE = 0
        for xx, yy in zip(x, y):
            MSE += (yy - node.predict(xx))**2
        self.assertAlmostEqual(node.MSE, MSE/len(x))
        self.assertEqual(list(node), list(zip(x, y)))

    def test_init(self):
        for noise in [0, 1, 2, 4, 8]:
            x = [d[0] for d in self.data]
            y = [d[1] + random.gauss(0, noise) for d in self.data]
            node = Leaf(x, y, config=self.config)
            self.perform_tests(x, y, node, noise > 0)

    def test_add_remove(self):
        for noise in [0, 1, 2, 4, 8]:
            x = [d[0] for d in self.data]
            y = [d[1] + random.gauss(0, noise) for d in self.data]
            limit = self.size // 3
            new_x = x[:limit]
            new_y = y[:limit]
            node = Leaf(list(new_x), list(new_y), config=self.config)
            self.perform_tests(new_x, new_y, node, noise > 0)
            for xx, yy in zip(x[limit:], y[limit:]):
                node.add(xx, yy)
                new_x.append(xx)
                new_y.append(yy)
                self.perform_tests(new_x, new_y, node, noise > 0)
            for _ in range(2*limit):
                xx, yy = node.pop()
                self.assertEqual(xx, new_x.pop())
                self.assertEqual(yy, new_y.pop())
                self.perform_tests(new_x, new_y, node, noise > 0)

    def test_plus(self):
        l1 = Leaf(range(10), range(10), config=self.config)
        for l2 in [Leaf(range(21, 11, -1), range(21, 11, -1), config=self.config),
                   Leaf(range(10, 20), range(10, 20), config=self.config)]:
            leaf = l1 + l2
            self.assertAlmostEqual(leaf.intercept, 0)
            self.assertAlmostEqual(leaf.coeff, 1)
            self.assertAlmostEqual(leaf.MSE, 0)
            self.assertEqual(leaf.x.values, list(sorted(l1.x.values + l2.x.values)))
            self.assertEqual(leaf.x.values, list(sorted(l1.y.values + l2.y.values)))

    def assert_equal_reg(self, dataset1, dataset2):
        leaf1 = Leaf([d[0] for d in dataset1], [d[1]
                                                for d in dataset1], config=self.config)
        leaf2 = Leaf([d[0] for d in dataset2], [d[1]
                                                for d in dataset2], config=self.config)
        self.assertEqual(leaf1, leaf2)

    def assert_notequal_reg(self, dataset1, dataset2):
        leaf1 = Leaf([d[0] for d in dataset1], [d[1]
                                                for d in dataset1], config=self.config)
        leaf2 = Leaf([d[0] for d in dataset2], [d[1]
                                                for d in dataset2], config=self.config)
        self.assertNotEqual(leaf1, leaf2)

    def test_repr(self):
        self.assertEqual(str(Leaf([], [], self.config)), '⊥')
        x, y = zip(*generate_dataset(intercept=3,
                                     coeff=1, size=100, min_x=0, max_x=100))
        reg = Leaf(x, y, self.config)
        self.assertEqual(str(reg), 'y ~ 1.000e+00x + 3.000e+00')
        dot = graphviz.Digraph()
        reg._to_graphviz(dot)
        expected = 'digraph {\n\t%d [label="%s"]\n}' % (id(reg), str(reg))
        self.assertEqual(str(dot), expected)


class NodeTest(unittest.TestCase):

    def test_nosplit(self):
        intercept = random.uniform(0, 100)
        coeff = random.uniform(0, 100)
        dataset = generate_dataset(
            intercept=intercept, coeff=coeff, size=50, min_x=0, max_x=100)
        reg = compute_regression(dataset)
        self.assertIsInstance(reg, Leaf)
        self.assertAlmostEqual(reg.intercept, intercept)
        self.assertAlmostEqual(reg.coeff, coeff)
        self.assertAlmostEqual(reg.RSS, 0, delta=1e-3)
        self.assertEqual(reg.breakpoints, [])
        self.assertEqual(list(reg), list(sorted(dataset)))

    def test_singlesplit(self):
        intercept_1 = random.uniform(0, 50)
        coeff_1 = random.uniform(0, 50)
        intercept_2 = random.uniform(50, 100)
        coeff_2 = random.uniform(50, 100)
        split = random.uniform(30, 60)
        dataset1 = generate_dataset(
            intercept=intercept_1, coeff=coeff_1, size=50, min_x=0, max_x=split)
        dataset2 = generate_dataset(
            intercept=intercept_2, coeff=coeff_2, size=50, min_x=split, max_x=100)
        dataset = dataset1 + dataset2
        random.shuffle(dataset)
        reg = compute_regression(dataset)
        self.assertIsInstance(reg, Node)
        self.assertAlmostEqual(reg.RSS, 0, delta=1e-3)
        self.assertIsInstance(reg.left, Leaf)
        self.assertAlmostEqual(reg.left.intercept, intercept_1)
        self.assertAlmostEqual(reg.left.coeff, coeff_1)
        self.assertIsInstance(reg.right, Leaf)
        self.assertAlmostEqual(reg.right.intercept, intercept_2)
        self.assertAlmostEqual(reg.right.coeff, coeff_2)
        self.assertEqual(reg.split, max(dataset1)[0])
        self.assertEqual(reg.breakpoints, [reg.split])
        self.assertEqual(list(reg), list(sorted(dataset)))

    def assertAlmostIncluded(self, sub_sequence, sequence, epsilon=1e-2):
        for elt in sub_sequence:
            is_in = False
            for ref in sequence:
                if abs(elt-ref) < epsilon:
                    is_in = True
                    break
            if not is_in:
                self.fail('Element %s is not in sequence %s (with ε=%f).' %
                          (elt, sequence, epsilon))

    def generic_multiplesplits(self, cls, repeat):
        all_datasets = [generate_dataset(intercept=i, coeff=i, size=50, min_x=(
            i-1)*10, max_x=i*10, cls=cls, repeat=repeat) for i in range(1, 9)]
        dataset = sum(all_datasets, [])
        reg = compute_regression(dataset)
        self.assertEqual(list(reg), list(sorted(dataset)))
        # TODO should be 7, but is 8 in reality because of the non-optimality of the algorithm
        self.assertIn(len(reg.breakpoints), (7, 8))
        self.assertAlmostIncluded(
            range(10, 80, 10), reg.breakpoints, epsilon=2)
        for x, y in dataset:
            prediction = reg.predict(x)
            self.assertAlmostEqual(y, prediction)

    def test_multiple_splits(self):
        self.generic_multiplesplits(float, 1)
        self.generic_multiplesplits(float, 10)

    def test_multiple_splits_decimal(self):
        self.generic_multiplesplits(Decimal, 1)

    def test_multiple_splits_fraction(self):
        self.generic_multiplesplits(Fraction, 1)

    def test_repr(self):
        config = Config(mode='BIC', epsilon=1e-6)
        data = {}
        for i in range(1, 5):
            data[i] = generate_dataset(
                intercept=i, coeff=i, size=100, min_x=i*100, max_x=(i+1)*100) + [((i+1)*100, (i+1)*100*i+i)]
            x = [d[0] for d in data[i]]
            y = [d[1] for d in data[i]]
            data[i] = x, y
        left = Node(Leaf(*data[1], config),  Leaf(list(reversed(data[2][0])),
                                                  list(reversed(data[2][1])), config), no_check=True)
        right = Node(Leaf(*data[3], config), Leaf(list(reversed(data[4][0])),
                                                  list(reversed(data[4][1])), config), no_check=True)
        node = Node(left, right, no_check=True)
        expected = '\n'.join([
            'x ≤ 3.000e+02?',
            '    └──x ≤ 2.000e+02?',
            '    │    └──y ~ 1.000e+00x + 1.000e+00',
            '    │    └──y ~ 2.000e+00x + 2.000e+00',
            '    └──x ≤ 4.000e+02?',
            '         └──y ~ 3.000e+00x + 3.000e+00',
            '         └──y ~ 4.000e+00x + 4.000e+00', ])
        self.assertEqual(expected, str(node))
        dot = node.to_graphviz()

        expected = '\n'.join([
            'digraph {',
            f'\t{id(node)} [label="x ≤ {node.split:.3e}?" shape=box]',
            f'\t{id(node.left)} [label="x ≤ {node.left.split:.3e}?" shape=box]',
            f'\t{id(node.left.left)} [label="{str(node.left.left)}"]',
            f'\t{id(node.left.right)} [label="{str(node.left.right)}"]',
            f'\t{id(node.left)} -> {id(node.left.left)} [label=yes]',
            f'\t{id(node.left)} -> {id(node.left.right)} [label=no]',
            f'\t{id(node.right)} [label="x ≤ {node.right.split:.3e}?" shape=box]',
            f'\t{id(node.right.left)} [label="{str(node.right.left)}"]',
            f'\t{id(node.right.right)} [label="{str(node.right.right)}"]',
            f'\t{id(node.right)} -> {id(node.right.left)} [label=yes]',
            f'\t{id(node.right)} -> {id(node.right.right)} [label=no]',
            f'\t{id(node)} -> {id(node.left)} [label=yes]',
            f'\t{id(node)} -> {id(node.right)} [label=no]',
            '}', ])
        self.maxDiff = None
        self.assertEqual(str(dot), expected)

    @mock.patch("matplotlib.pyplot.show")
    def test_plot_dataset(self, mock_show):
        all_datasets = [generate_dataset(intercept=i, coeff=i, size=50, min_x=(
            i-1)*10, max_x=i*10) for i in range(1, 9)]
        dataset = sum(all_datasets, [])
        reg = compute_regression(dataset)
        reg.plot_dataset()
        reg.plot_dataset(log=True)
        reg.plot_dataset(log_x=True)
        reg.plot_dataset(log_y=True)
        reg.plot_dataset(plot_merged_reg=True)
        reg.plot_dataset(color=False)
        reg.plot_dataset(color='green')
        reg.plot_dataset(color=['green', 'blue', 'red'])

    @mock.patch("matplotlib.pyplot.show")
    def test_plot_error(self, mock_show):
        all_datasets = [generate_dataset(intercept=i, coeff=i, size=50, min_x=(
            i-1)*10, max_x=i*10) for i in range(1, 9)]
        dataset = sum(all_datasets, [])
        reg = compute_regression(dataset)
        reg.plot_error()
        reg.plot_error(log=True)
        reg.plot_error(log_x=True)
        reg.plot_error(log_y=True)


class FlatRegressionTest(unittest.TestCase):

    def assertAlmostIncluded(self, sub_sequence, sequence, epsilon=1e-2):
        for elt in sub_sequence:
            is_in = False
            for ref in sequence:
                if abs(elt-ref) < epsilon:
                    is_in = True
                    break
            if not is_in:
                self.fail('Element %s is not in sequence %s (with ε=%f).' %
                          (elt, sequence, epsilon))

    def generic_multiplesplits(self, cls, repeat):
        self.maxDiff = None
        all_datasets = [generate_dataset(intercept=i, coeff=i, size=50, min_x=(
            i-1)*10, max_x=i*10, cls=cls, repeat=repeat) for i in range(1, 9)]
        dataset = sum(all_datasets, [])
        reg = compute_regression(dataset)
        flat_reg = reg.flatify()
        self.assertEqual(list(flat_reg), list(sorted(dataset)))
        # TODO should be 7, but is 8 in reality because of the non-optimality of the algorithm
        self.assertEqual(reg.nb_params, flat_reg.nb_params)
        self.assertEqual(reg.breakpoints, flat_reg.breakpoints)
        self.assertTrue(flat_reg.null_RSS)
        self.assertTrue(flat_reg.rss_equal(flat_reg.RSS, 0))
        self.assertIn(len(flat_reg.breakpoints), (7, 8))
        self.assertAlmostIncluded(
            range(10, 80, 10), flat_reg.breakpoints, epsilon=2)
        for x, y in dataset:
            prediction = flat_reg.predict(x)
            self.assertAlmostEqual(y, prediction)
        other_flat = compute_regression(dataset, breakpoints=flat_reg.breakpoints)
        self.assertEqual(str(other_flat), str(flat_reg))

    def test_multiple_splits(self):
        self.generic_multiplesplits(float, 1)
        self.generic_multiplesplits(float, 10)

    def test_multiple_splits_decimal(self):
        self.generic_multiplesplits(Decimal, 1)

    def test_multiple_splits_fraction(self):
        self.generic_multiplesplits(Fraction, 1)

    def test_repr(self):
        config = Config(mode='BIC', epsilon=1e-6)
        data = {}
        for i in range(1, 5):
            data[i] = generate_dataset(
                intercept=i, coeff=i, size=100, min_x=i*100, max_x=(i+1)*100) + [((i+1)*100, (i+1)*100*i+i)]
        dataset = data[1] + data[2] + data[3] + data[4]
        reg = FlatRegression([d[0] for d in dataset], [d[1] for d in dataset], config, [200, 300, 400])
        expected = '\n'.join([
            '-inf < x ≤ 2.000e+02',
            '\ty ~ 1.000e+00x + 1.000e+00',
            '2.000e+02 < x ≤ 3.000e+02',
            '\ty ~ 2.000e+00x + 2.000e+00',
            '3.000e+02 < x ≤ 4.000e+02',
            '\ty ~ 3.000e+00x + 3.000e+00',
            '4.000e+02 < x ≤ inf',
            '\ty ~ 4.000e+00x + 4.000e+00'])
        self.assertEqual(expected, str(reg))

    @mock.patch("matplotlib.pyplot.show")
    def test_plot_dataset(self, mock_show):
        all_datasets = [generate_dataset(intercept=i, coeff=i, size=50, min_x=(
            i-1)*10, max_x=i*10) for i in range(1, 9)]
        dataset = sum(all_datasets, [])
        reg = compute_regression(dataset)
        reg.plot_dataset()
        reg.plot_dataset(log=True)
        reg.plot_dataset(log_x=True)
        reg.plot_dataset(log_y=True)
        reg.plot_dataset(plot_merged_reg=True)
        reg.plot_dataset(color=False)
        reg.plot_dataset(color='green')
        reg.plot_dataset(color=['green', 'blue', 'red'])

    def generic_multiplesplits_simplify(self, cls, repeat):
        self.maxDiff = None
        all_datasets = [generate_dataset(intercept=i, coeff=i, size=50, min_x=(
            i-1)*10, max_x=i*10, cls=cls, repeat=repeat) for i in range(1, 9)]
        dataset = sum(all_datasets, [])
        reg = compute_regression(dataset)
        merged = reg.merge()
        simple_df = reg.simplify()
        self.assertEqual(len(simple_df), len(reg.breakpoints)+1)
        self.assertEqual(list(simple_df.nb_breakpoints), list(range(len(reg.breakpoints), -1, -1)))
        self.assertTrue(reg.rss_equal(reg.RSS, simple_df.RSS[0]))
        self.assertTrue(reg.rss_equal(list(simple_df.RSS)[-1], merged.RSS))
        self.assertTrue(reg.error_equal(reg.BIC, simple_df.BIC[0]))
        self.assertTrue(reg.error_equal(list(simple_df.BIC)[-1], merged.BIC))
        for old_rss, new_rss in zip(simple_df.RSS, simple_df.RSS[1:]):
            if not reg.rss_equal(old_rss, new_rss):
                self.assertLess(old_rss, new_rss)
        for nb_breakpoints, new_reg in zip(simple_df.nb_breakpoints, simple_df.regression):
            self.assertEqual(list(reg), list(new_reg))
            self.assertEqual(nb_breakpoints, len(new_reg.breakpoints))
            self.assertTrue(set(new_reg.breakpoints) <= set(reg.breakpoints))
        simple_reg = reg.auto_simplify()
        expected_reg = simple_df.regression[1]
        self.assertEqual(simple_reg.breakpoints, expected_reg.breakpoints)
        self.assertEqual(simple_reg.RSS, expected_reg.RSS)
        self.assertEqual(simple_reg.BIC, expected_reg.BIC)
        # Checking that the auto_simplify() is a fix-point
        new_reg = simple_reg.auto_simplify()
        self.assertEqual(simple_reg.breakpoints, new_reg.breakpoints)
        # Checking to_pandas method
        df = new_reg.to_pandas()
        self.assertEqual(len(df), len(new_reg.segments))
        for (_, row), ((min_x, max_x), leaf) in zip(df.iterrows(), new_reg.segments):
            self.assertEqual(row['min_x'], min_x)
            self.assertEqual(row['max_x'], max_x)
            self.assertEqual(row['intercept'], leaf.intercept)
            self.assertEqual(row['coefficient'], leaf.coeff)
            self.assertEqual(row['RSS'], leaf.RSS)
            self.assertEqual(row['MSE'], leaf.MSE)

    def test_multiple_splits_simplify(self):
        self.generic_multiplesplits_simplify(float, 1)
        self.generic_multiplesplits_simplify(float, 10)

    def test_multiple_splits_decimal_simplify(self):
        self.generic_multiplesplits_simplify(Decimal, 1)

    def test_multiple_splits_fraction_simplify(self):
        self.generic_multiplesplits_simplify(Fraction, 1)


if __name__ == "__main__":
    unittest.main()
