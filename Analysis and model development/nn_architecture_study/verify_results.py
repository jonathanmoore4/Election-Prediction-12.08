"""Audit saved experiment counts, temporal selection and probability metrics."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

def main(directory=HERE):
    directory = Path(directory)
    configuration = json.loads((directory/'configuration.json').read_text())
    finalists = json.loads((directory/'finalists.json').read_text())['architectures']
    checks = []
    root = HERE.parents[1]
    data = pd.read_csv(root/'TEST_TRAIN/train.csv')
    assert hashlib.sha256((root/'TEST_TRAIN/train.csv').read_bytes()).hexdigest() == configuration['data_sha256']
    if 'evaluation_data_sha256' in configuration:
        assert hashlib.sha256((root/'TEST_TRAIN/test.csv').read_bytes()).hexdigest() == configuration['evaluation_data_sha256']
        data = pd.concat([data,pd.read_csv(root/'TEST_TRAIN/test.csv')],ignore_index=True)
        assert hashlib.sha256((HERE/'study.py').read_bytes()).hexdigest() == configuration['implementation_sha256']
        assert hashlib.sha256((HERE/'extend_all_elections.py').read_bytes()).hexdigest() == configuration['extension_sha256']
    windows = {cutoff:(vy,ey) for cutoff,vy,ey in configuration['windows']}
    if (directory/'execution_backend.json').exists():
        backend = json.loads((directory/'execution_backend.json').read_text())
        for file,key in [('extend_all_elections.py','reference_runner_sha256'),
                         ('run_all_elections_fast.py','accelerated_runner_sha256'),
                         ('fast_batches.py','batch_loader_sha256')]:
            assert hashlib.sha256((HERE/file).read_bytes()).hexdigest() == backend[key]
        assert hashlib.sha256((directory/'batch_verification.txt').read_bytes()).hexdigest() == backend['verification_sha256']
        assert backend['torch'] == configuration['torch']
        checks.append('Accelerated batching provenance verified; batch/history/checkpoint equivalence checks are saved in batch_verification.txt.')
    for stage, architectures, seeds in [('screening',list(configuration['architectures']),3),('confirmation',finalists,10)]:
        results = pd.read_csv(directory/f'{stage}_results.csv')
        records = pd.read_csv(directory/f'{stage}_training_records.csv')
        policies = pd.read_csv(directory/f'{stage}_policies.csv')
        expected = len(architectures)*len(configuration['rates'])*len(configuration['patiences'])*len(configuration['windows'])
        assert len(results) == expected
        assert len(records) == expected*seeds
        assert not results.duplicated(['architecture','train_through','learning_rate','patience']).any()
        assert not records.duplicated(['architecture','train_through','learning_rate','patience','seed']).any()
        assert set(results.train_through) == set(windows)
        assert set(results.learning_rate) == set(configuration['rates'])
        assert set(results.patience) == set(configuration['patiences'])
        assert all((r.validation_year,r.evaluation_year) == windows[r.train_through] for r in results.itertuples())
        assert set(results.evaluation_year) == {w[2] for w in configuration['windows']}
        assert set(results.architecture) == set(architectures)
        assert set(records.seed) == set(configuration['seeds'][:seeds])
        assert (results.train_through < results.validation_year).all()
        assert (results.validation_year < results.evaluation_year).all()
        assert (records.best_epoch <= records.stopping_epoch).all()
        assert (records.best_epoch >= 1).all()
        assert (records.stopping_epoch <= 1000).all()
        grouped = records.pivot(index=['architecture','train_through','learning_rate','seed'],columns='patience',values='best_validation_loss')
        assert (grouped[20] <= grouped[10] + 1e-12).all() and (grouped[50] <= grouped[20] + 1e-12).all()
        stops = records.pivot(index=['architecture','train_through','learning_rate','seed'],columns='patience',values='stopping_epoch')
        assert (stops[20] >= stops[10]).all() and (stops[50] >= stops[20]).all()
        assert np.isfinite(results.select_dtypes('number')).all().all()
        assert not (results.validation_unseen_winners + results.evaluation_unseen_winners).any()
        for row in policies.itertuples():
            subset = results.loc[(results.architecture == row.architecture) & (results.train_through == row.train_through)]
            if row.policy != 'validation_rate_and_patience':
                subset = subset.loc[subset.patience == int(row.patience)]
            if row.policy == 'validation_midrange': subset = subset.loc[subset.learning_rate.isin([.1,.2,.3,.5])]
            elif row.policy == 'validation_compact': subset = subset.loc[subset.learning_rate.isin([.1,.3])]
            elif row.policy.startswith('fixed_'): subset = subset.loc[subset.learning_rate == float(row.policy.removeprefix('fixed_'))]
            selected = subset.loc[subset.validation_log_loss.idxmin()]
            assert row.learning_rate == selected.learning_rate
            np.testing.assert_allclose(row.evaluation_accuracy,selected.evaluation_accuracy,atol=1e-14)
        predictions = pd.read_csv(directory/f'{stage}_predictions.csv.gz')
        pcols = [c for c in predictions if c.startswith('p_')]
        probs = predictions[pcols].to_numpy()
        assert np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all()
        np.testing.assert_allclose(probs.sum(axis=1),1,atol=2e-6)
        for keys,g in predictions.groupby(['architecture','train_through','learning_rate','patience','split'],sort=False):
            arch,cutoff,rate,patience,split = keys
            matched = results.loc[(results.architecture == arch) & (results.train_through == cutoff)
                                  & (results.learning_rate == rate) & (results.patience == patience)].iloc[0]
            assert len(g) == matched[split+'_rows'] and g.source_row.is_unique
            expected_year = windows[cutoff][0 if split == 'validation' else 1]
            source = data.loc[g.source_row]
            assert (source.election == expected_year).all()
            np.testing.assert_array_equal(source.winner.to_numpy(),g.winner.to_numpy())
            np.testing.assert_array_equal(np.sort(g.source_row),data.index[data.election == expected_year].to_numpy())
            probabilities = g[pcols].to_numpy()
            targets = np.column_stack([(g.winner == c.removeprefix('p_')).to_numpy() for c in pcols])
            truth_probability = (targets*probabilities).sum(axis=1)
            reconstructed_loss = -np.log(np.clip(truth_probability,np.finfo(float).eps,1)).mean()
            reconstructed_brier = ((probabilities-targets)**2).sum(axis=1).mean()
            np.testing.assert_allclose(reconstructed_loss,matched[split+'_log_loss'],rtol=1e-6,atol=1e-7)
            np.testing.assert_allclose(reconstructed_brier,matched[split+'_brier_score'],rtol=1e-6,atol=1e-7)
            changed = (source.previous_winner.notna() & source.winner.ne(source.previous_winner)).to_numpy()
            predicted = g[pcols].idxmax(axis=1).str.removeprefix('p_')
            correct = (predicted == g.winner).to_numpy()
            np.testing.assert_allclose(correct.mean(),matched[split+'_accuracy'],atol=1e-14)
            np.testing.assert_allclose(correct[changed].mean(),matched[split+'_changed_seat_accuracy'],atol=1e-14)
        checks.append(f'{stage}: {len(results)} ensembles, {len(records)} patience records, policy selection, source-row alignment, accuracy, changed-seat accuracy, log loss and Brier score verified')
    if (directory/'election_diagnostics.csv').exists():
        diagnostics = pd.read_csv(directory/'election_diagnostics.csv')
        party = pd.read_csv(directory/'per_party_recall.csv')
        for row in diagnostics.itertuples():
            ev = data.loc[data.election == row.evaluation_year]
            assert row.rows == len(ev)
            assert row.known_previous_rows + row.unknown_previous_rows == row.rows
            assert row.unknown_previous_rows == ev.previous_winner.isna().sum()
            np.testing.assert_allclose(row.correct_changes / row.changed_seats,row.changed_accuracy,atol=1e-14)
            # On known-history rows, gains over persistence must equal correct
            # changes minus spurious changes on seats that actually held.
            np.testing.assert_allclose(
                row.known_previous_rows*(row.known_previous_accuracy-row.previous_winner_known_accuracy),
                row.correct_changes-row.false_changes_on_held_seats,atol=1e-10)
            p = party.loc[(party.evaluation_year == row.evaluation_year) &
                (party.architecture == row.architecture) & (party.policy == row.policy)]
            assert p.actual_seats.sum() == p.predicted_seats.sum() == row.rows
            np.testing.assert_allclose(p.expected_seats.sum(),row.rows,atol=2e-4)
            for pr in p.itertuples():
                assert pr.actual_seats == ev.winner.eq(pr.party).sum()
            assert p.seat_error.abs().sum() == row.seat_count_absolute_error
        assert len(diagnostics) == len(configuration['windows'])*len(finalists)*2
        checks.append('Election diagnostics and party totals verified, including missing-history denominators and persistence-gain identity.')
    # Baseline comparison against the independently executed previous notebook.
    previous = HERE.parent/'nn_whole_election_validation_results'
    old = pd.read_csv(previous/'validation_results.csv')
    current = pd.read_csv(directory/'confirmation_results.csv')
    current = current.loc[(current.architecture == '32_16') & (current.patience == 20)]
    matched = old.merge(current,on=['train_through','learning_rate'])
    expected_reference = old.loc[old.train_through.isin(current.train_through)]
    assert len(matched) == len(expected_reference)
    for before,after in [('accuracy','validation_accuracy'),('log_loss','validation_log_loss'),('brier_score','validation_brier_score')]:
        np.testing.assert_allclose(matched[before],matched[after],rtol=1e-6,atol=1e-7)
    checks.append(f'All {len(matched)} overlapping original ten-seed 32/16 patience-20 validation results reproduced.')
    (directory/'verification.txt').write_text('\n'.join(checks)+'\n')
    print('\n'.join(checks))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=HERE)
    main(parser.parse_args().directory)
