from pathlib import Path
from typing import List, Tuple

import attr
import numpy as np


@attr.s(auto_attribs=True, slots=True, frozen=True, hash=True, cache_hash=True, eq=True)
class FeatureTest:
    feature_index: int
    lower_bound: float
    upper_bound: float

    def covers(self, dataset_instance: np.ndarray) -> bool:
        return self.lower_bound <= dataset_instance[self.feature_index] < self.upper_bound


@attr.s(auto_attribs=True, slots=True, frozen=True, hash=True, cache_hash=True, eq=True)
class Rule:
    antecedent: List[FeatureTest]
    consequent: np.ndarray

    def covers(self, dataset_instance: np.ndarray) -> bool:
        return all((ft.covers(dataset_instance) for ft in self.antecedent))


@attr.s(auto_attribs=True, slots=True, frozen=True, hash=True, cache_hash=True, eq=True)
class Individual:
    rules: List[Rule]


def parse_feature_test(raw: str) -> FeatureTest:
    lower_bound, tail = raw.split(' <= ')
    _, tail = tail.split('x[')
    feature_index, upper_bound = tail.split('] < ')

    feature_index = int(feature_index)

    if lower_bound == '-∞':
        lower_bound = float('-inf')
    else:
        lower_bound = float(lower_bound)

    if upper_bound == '∞':
        upper_bound = float('inf')
    else:
        upper_bound = float(upper_bound)

    return FeatureTest(feature_index=feature_index, lower_bound=lower_bound, upper_bound=upper_bound)


def parse_rule(raw: str, feature_count: int, class_count: int) -> Rule:
    fields = raw.split(',')
    fields = [f for f in fields if f]

    antecedent = (parse_feature_test(ft) for ft in fields[:feature_count])
    antecedent = list(sorted(antecedent, key=lambda ft: ft.feature_index))

    consequent = [bool(f) for f in fields[feature_count:]]

    if len(consequent) != class_count:
        raise Exception

    consequent = np.asarray(consequent)

    return Rule(antecedent=antecedent, consequent=consequent)


def get_feature_count_and_class_count(raw_rule: str) -> Tuple[int, int]:
    feature_count = raw_rule.count('[')
    fields = raw_rule.split(',')
    fields = [f for f in fields if f]
    class_count = len(fields) - feature_count

    return feature_count, class_count


def parse_individual(raw: str) -> Individual:
    lines = raw.split('\n')

    # Remove header and footer
    rule_lines = lines[1:-1]

    feature_count, class_count = get_feature_count_and_class_count(raw_rule=rule_lines[0])
    rules = [parse_rule(raw=rl, feature_count=feature_count, class_count=class_count)
             for rl in rule_lines]

    return Individual(rules=rules)


def main():
    test = Path(r"T:\Source\minotaur_msc_def_analysis\minotaur_output\run-0-dataset-CAL500-fold-0-output\model-888.csv")
    ind = parse_individual(test.read_text(encoding='UTF8'))
    print(ind)


if __name__ == '__main__':
    main()
