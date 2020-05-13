import itertools
import multiprocessing
import platform
import subprocess
from pathlib import Path

_SYSTEM = platform.system()
if _SYSTEM == 'Windows':
    MINOTAUR_PATH = Path.cwd() / 'bin' / 'minotaur-win10-x64.exe'
elif _SYSTEM == 'Linux':
    MINOTAUR_PATH = Path.cwd() / 'bin' / 'minotaur-linux-x64.exe'
else:
    raise Exception("Unknown system / paths.")

DATASETS_DIR = Path.cwd() / 'datasets'

RUN_COUNT = 30
FOLD_COUNT = 10

OUTPUT_BASE_DIR = Path.cwd() / 'minotaur_output'
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

CFSBE_VALUES = {'CAL500': 64,
                'emotions': 128,
                'scene': 512,
                'synthetic0': 4096,
                'synthetic1': 32,
                'synthetic2': 2048,
                'synthetic3': 1024,
                'yeast': 512}

DATASET_NAMES = list(CFSBE_VALUES.keys())


def get_train_data_path(dataset_name: str, fold_nr: int) -> Path:
    return DATASETS_DIR / dataset_name / 'folds' / str(fold_nr) / 'train-data.csv'


def get_train_labels_path(dataset_name: str, fold_nr: int) -> Path:
    return DATASETS_DIR / dataset_name / 'folds' / str(fold_nr) / 'train-labels.csv'


def get_test_data_path(dataset_name: str, fold_nr: int) -> Path:
    return DATASETS_DIR / dataset_name / 'folds' / str(fold_nr) / 'test-data.csv'


def get_test_labels_path(dataset_name: str, fold_nr: int) -> Path:
    return DATASETS_DIR / dataset_name / 'folds' / str(fold_nr) / 'test-labels.csv'


def get_output_dir(run_nr: int, dataset: str, fold_nr: int) -> Path:
    return OUTPUT_BASE_DIR / f"run-{run_nr}-dataset-{dataset}-fold-{fold_nr}-output"


def get_stdout_redirection_path(run_nr: int, dataset: str, fold_nr: int) -> Path:
    return get_output_dir(run_nr=run_nr, dataset=dataset, fold_nr=fold_nr) / "stdout-redirection.txt"


def run_minotaur(dataset: str, fold_nr: int, output_dir: Path, stdout_redirection_path: Path):
    cfsbe_value = CFSBE_VALUES[dataset]

    args = [MINOTAUR_PATH,
            '--train-data', str(get_train_data_path(dataset, fold_nr)),
            '--train-labels', str(get_train_labels_path(dataset, fold_nr)),
            '--test-data', str(get_test_data_path(dataset, fold_nr)),
            '--test-labels', str(get_test_labels_path(dataset, fold_nr)),
            '--output-dir', str(output_dir),
            '--fittest-selection', 'nsga2',
            '--fitness-metrics', 'fscore',
            '--fitness-metrics', 'rule-count',
            '--individual-mutation-add-rule-weight', '5',
            '--individual-mutation-modify-rule-weight', '20',
            '--individual-mutation-remove-rule-weight', '10',
            '--max-generations', '200',
            '--population-size', '80',
            '--mutants-per-generation', '40',
            '--max-failed-mutations-per-generation', '2000',
            '--minotaur-hyperparameter-t', str(cfsbe_value),
            '--save-models',
            '--skip-expensive-sanity-checks']

    with stdout_redirection_path.open(mode='wt') as stdout_redirection:
        print(f"Running 'MINOTAUR' on '{dataset}'-fold-'{fold_nr}''...", flush=True)
        subprocess.run(args=args, stdout=stdout_redirection)


def create_dirs_and_run_minotaur(run_nr: int, dataset: str, fold_nr: int):
    output_dir = get_output_dir(run_nr=run_nr, dataset=dataset, fold_nr=fold_nr)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    stdout_redirection_path = get_stdout_redirection_path(run_nr=run_nr, dataset=dataset, fold_nr=fold_nr)
    stdout_redirection_path.parent.mkdir(parents=True, exist_ok=True)

    run_minotaur(dataset=dataset,
                 fold_nr=fold_nr,
                 output_dir=output_dir,
                 stdout_redirection_path=stdout_redirection_path)


def main():
    parameters = itertools.product(range(RUN_COUNT), DATASET_NAMES, range(FOLD_COUNT))
    p = multiprocessing.Pool(3)
    p.starmap(func=create_dirs_and_run_minotaur, iterable=parameters)


if __name__ == '__main__':
    main()
