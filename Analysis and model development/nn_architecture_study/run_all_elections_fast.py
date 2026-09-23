"""Resume the reference experiment with verified-equivalent faster batching.

Model cache identity remains the reference study's identity because this backend
must reproduce it exactly. A separate manifest records the backend and checks.
The unmodified extend_all_elections.py remains the reference reproduction route.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import extend_all_elections as extension
from fast_batches import TensorLoader

HERE = Path(__file__).resolve().parent


def verify_backend():
    study = extension.study
    windows = extension.configure()
    reference = study.DataLoader
    lines = []
    for window in windows.values():
        n = len(window['training'])
        dataset = study.TensorDataset(torch.arange(n), torch.arange(n) * 2)
        old = reference(dataset, batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(study.SEEDS[0]), num_workers=0)
        new = TensorLoader(dataset, batch_size=64, shuffle=True,
                           generator=torch.Generator().manual_seed(study.SEEDS[0]), num_workers=0)
        for _ in range(3):
            old_batches, new_batches = list(old), list(new)
            assert len(old_batches) == len(new_batches)
            for a, b in zip(old_batches, new_batches):
                for x, y in zip(a, b):
                    assert torch.equal(x, y)
            assert torch.equal(old.generator.get_state(), new.generator.get_state())
        lines.append(f'{n} rows: exact batch order and generator state for three epochs.')
    for cutoff, arch, rate in [(1987, '16', .01), (1997, '32_16', .3), (2017, '64_32', 1.)]:
        old = study.fit_run(windows[cutoff], arch, rate, study.SEEDS[0])
        study.DataLoader = TensorLoader
        try:
            new = study.fit_run(windows[cutoff], arch, rate, study.SEEDS[0])
        finally:
            study.DataLoader = reference
        for key in ['validation', 'evaluation', 'validation_loss_history']:
            np.testing.assert_array_equal(old[key], new[key])
        for a, b in zip(json.loads(str(old['records'])), json.loads(str(new['records']))):
            a.pop('elapsed_seconds'); b.pop('elapsed_seconds')
            assert a == b
        lines.append(f'{cutoff}/{arch}/LR {rate:g}: exact probabilities, loss history and checkpoint records at all three patience values.')
    return '\n'.join(lines) + '\n'


def main():
    extension.OUTPUT.mkdir(exist_ok=True)
    verification = verify_backend()
    (extension.OUTPUT/'batch_verification.txt').write_text(verification)
    manifest = dict(
        backend='Verified DataLoader-equivalent tensor indexing; reference and accelerated caches may coexist.',
        reference_runner_sha256=hashlib.sha256((HERE/'extend_all_elections.py').read_bytes()).hexdigest(),
        accelerated_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        batch_loader_sha256=hashlib.sha256((HERE/'fast_batches.py').read_bytes()).hexdigest(),
        verification_sha256=hashlib.sha256(verification.encode()).hexdigest(),
        torch=torch.__version__,
        timing_note='Elapsed times mix reference and accelerated batching and are not a controlled performance comparison.',
    )
    (extension.OUTPUT/'execution_backend.json').write_text(json.dumps(manifest, indent=2))
    print(verification, flush=True)
    extension.study.DataLoader = TensorLoader
    extension.main()


if __name__ == '__main__':
    main()
