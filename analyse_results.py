import itertools
import multiprocessing
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

import minotaur_modeling

MINOTAUR_OUTPUT_DIR = Path.cwd() / 'minotaur_output'

DATASETS = ['CAL500',
            'emotions',
            'scene',
            'synthetic0',
            'synthetic1',
            'synthetic2',
            'synthetic3',
            'yeast']

RUN_COUNT = 30
FOLD_COUNT = 10

ANALYSIS_OUTPUT_DIR = Path.cwd() / 'analysis_output'
ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _get_minotaur_output_dir(run_nr: int, dataset: str, fold_nr) -> Path:
    return MINOTAUR_OUTPUT_DIR / f"run-{run_nr}-dataset-{dataset}-fold-{fold_nr}-output"


def _get_fitnesses_path(run_nr: int, dataset: str, fold_nr) -> Path:
    output_dir = _get_minotaur_output_dir(run_nr=run_nr, dataset=dataset, fold_nr=fold_nr)
    return output_dir / 'fitnesses.csv'


def _get_model_path(run_nr: int, dataset: str, fold_nr: int, model_id: int) -> Path:
    return _get_minotaur_output_dir(run_nr=run_nr, dataset=dataset, fold_nr=fold_nr) / f"model-{model_id}.csv"


def _get_dataset_path(dataset: str, fold_nr: int) -> Dict[str, Path]:
    dataset_dir = Path.cwd() / 'datasets' / dataset / 'folds' / str(fold_nr)
    return {'train_data': dataset_dir / 'train-data.csv',
            'train_labels': dataset_dir / 'train-labels.csv',
            'test_data': dataset_dir / 'test-data.csv',
            'test_labels': dataset_dir / 'test-labels.csv'}


def read_fitnesses_file(run_nr: int, dataset: str, fold_nr: int) -> pd.DataFrame:
    filename = _get_fitnesses_path(run_nr=run_nr, dataset=dataset, fold_nr=fold_nr)
    df = pd.read_csv(filepath_or_buffer=filename, index_col=False)
    df = df[['Id', 'MultiLabelFScore (train)', 'MultiLabelFScore (test)', 'RuleCount (test)']]
    df = df.rename(columns={'Id': 'id',
                            'MultiLabelFScore (train)': 'fscore_train',
                            'MultiLabelFScore (test)': 'fscore_test',
                            'RuleCount (test)': 'rule_count'})
    df['run'] = run_nr
    df['dataset'] = dataset
    df['fold'] = fold_nr

    # MINOTAUR tries to maximize all objectives, so internally
    # the values of rule counts are stored as negative numbers.
    # Here, we are multiplying by -1 to obtain the actual rule counts
    df['rule_count'] = -1 * df['rule_count']
    return df


def read_model_file(run_nr: int, dataset: str, fold_nr: int, model_id: int) -> minotaur_modeling.Individual:
    filename = _get_model_path(run_nr=run_nr, dataset=dataset, fold_nr=fold_nr, model_id=model_id)
    raw_text = filename.read_text(encoding='UTF8')
    return minotaur_modeling.parse_individual(raw=raw_text)


def get_minotaurs_rep_id(run_nr: int, dataset: str, fold_nr: int) -> int:
    df = read_fitnesses_file(run_nr=run_nr, dataset=dataset, fold_nr=fold_nr)
    df = df.sort_values(by=['fscore_train', 'rule_count'],
                        ascending=[False, True])
    return df.iloc[0]['id']


def compute_default_rule_coverage(ind: minotaur_modeling.Individual, df: pd.DataFrame) -> float:
    df_values = df.values
    instance_count = df_values.shape[0]
    covered_by_default = 0

    rules = ind.rules

    for index in range(instance_count):
        dataset_instance = df_values[index]
        coverage = sum((r.covers(dataset_instance) for r in rules))

        if coverage == 0:
            covered_by_default += 1
        elif coverage == 1:
            continue
        else:
            raise Exception("Dang it! A instance should be covered by 0 or 1 rules!")

    return float(covered_by_default) / float(instance_count)


def get_runs_default_rule_coverage_rate(run_nr: int, dataset: str, fold_nr: int) -> Tuple[int, str, int, float, float]:
    rep_id = get_minotaurs_rep_id(run_nr=run_nr, dataset=dataset, fold_nr=fold_nr)
    rep = read_model_file(run_nr=run_nr, dataset=dataset, fold_nr=fold_nr, model_id=rep_id)

    dataset_paths = _get_dataset_path(dataset=dataset, fold_nr=fold_nr)
    train_df = pd.read_csv(filepath_or_buffer=dataset_paths['train_data'])
    train_coverage = compute_default_rule_coverage(rep, train_df)

    test_df = pd.read_csv(filepath_or_buffer=dataset_paths['test_data'])
    test_coverage = compute_default_rule_coverage(rep, test_df)

    return run_nr, dataset, fold_nr, train_coverage, test_coverage


def print_default_rule_coverages():
    # params = itertools.product(range(RUN_COUNT), DATASETS, range(FOLD_COUNT))
    # pool = multiprocessing.Pool()
    # results = pool.starmap(get_runs_default_rule_coverage_rate, params)
    # df = pd.DataFrame.from_records(data=results, columns=['run_nr', 'dataset', 'fold_nr',
    #                                                       'default_coverage_train', 'default_coverage_test'])
    # df.to_csv(path_or_buf=ANALYSIS_OUTPUT_DIR / 'df.csv')
    df = pd.read_csv(filepath_or_buffer=ANALYSIS_OUTPUT_DIR / 'df.csv')
    with pd.option_context('float_format', '{:.2f}'.format):
        print(df.groupby('dataset')[['default_coverage_train', 'default_coverage_test']].std())


def main():
    # print_stats()
    # generate_figures()
    print_default_rule_coverages()
    print("Done!")
    return


if __name__ == '__main__':
    main()
