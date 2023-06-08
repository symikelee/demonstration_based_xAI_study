import itertools
import numpy as np
import random as rand
import sys
from environment import FEATURES

N_BINS = 3
VALID_MODES = ['easy', 'hard']
VALID_BINS = ['1', '2', '3']
FEATURE_TO_FEATURE_CLASS = {
	f: ('fill' if i == 0 else 'color' if i == 1 else 'number' if i == 2 else 'shape') for i,feats in enumerate(FEATURES) for f in feats
}

def easy_rule_to_string(bins):
  primary = 'Priority 1: '
  secondary = 'Priority 2: '
  for i,b in enumerate(bins):
    if type(b) == str:
      primary += f'{b} in Bin {i+1}'
    else:
      secondary += f'{" and ".join(b)} in Bin {i+1}, '
  secondary = secondary[:-2]  # Remove last comma and space
  return primary + '\n' + secondary

def hard_rule_to_string(bins):
  primary = 'Primary Rule: '
  secondary = 'Exception: '
  for i,b in enumerate(bins):
    if type(b) == list:
      primary_str = b[0]
      secondary += f'{b[1]} in Bin {i+1}'
    else:
      primary_str = b
    primary += f'{primary_str} in Bin {i+1}, '
  primary = primary[:-2]  # Remove last comma and space
  return primary + '\n' + secondary

def generate_rule(mode: str) -> str:
  if mode not in VALID_MODES:
    raise ValueError(f'Invalid mode: {mode}, expected one of: {VALID_MODES}')

  if mode == 'easy':
    bins = generate_easy_rules()
    rule_str = easy_rule_to_string(bins)
  elif mode == 'hard':
    bins = generate_hard_rules()
    rule_str = hard_rule_to_string(bins)
  return bins, rule_str

def generate_easy_rules():
	features = [['hollow', 'striped', 'solid'],
		    ['red', 'green', 'purple'],
		    ['one', 'two', 'three'],
		    ['ellipse', 'diamond', 'squiggle']]

	bins = [None] * 3
	bin_a = rand.randint(0, 2)

	prim_ft = rand.randint(0, len(features) - 1)
	bins[bin_a] = rand.choice(features[prim_ft])
	second_ft = rand.choice([f for f in features if f != features[prim_ft]])
	rand.shuffle(second_ft)

	remaining_bins = [r for r in range(3) if r != bin_a]
	rand.shuffle(remaining_bins)
	bins[remaining_bins[0]] = second_ft[:1]
	bins[remaining_bins[1]] = second_ft[1:]

	return bins

def generate_hard_rules():
	features = [['hollow', 'striped', 'solid'],
                    ['red', 'green', 'purple'],
                    ['one', 'two', 'three'],
                    ['ellipse', 'diamond', 'squiggle']]

	prim_ft = rand.randint(0, len(features) - 1)
	bins = features[prim_ft]
	rand.shuffle(bins)

	bin_a = rand.randint(0, 2)

	second_ft = rand.choice([f for f in features if f != features[prim_ft]])
	rand.shuffle(second_ft)

	bins[bin_a] = [bins[bin_a], second_ft[0]]

	return bins

def generate_all_easy_rules():
	rules = []
	for prim_ft_class in range(len(FEATURES)):
		for prim_ft in range(len(FEATURES[prim_ft_class])):
			for prim_bin in range(N_BINS):
				for second_ft_class in range(len(FEATURES)):
					if second_ft_class == prim_ft_class:
						continue
					for second_bins in itertools.permutations([b for b in range(N_BINS) if b != prim_bin]):
						for second_ft in FEATURES[second_ft_class]:
							rule = [None]*N_BINS
							rule[prim_bin] = FEATURES[prim_ft_class][prim_ft]
							rule[second_bins[0]] = [second_ft]
							rule[second_bins[1]] = [f for f in FEATURES[second_ft_class] if f != second_ft]
							rules.append(rule)
	return rules

def generate_all_hard_rules():
	rules = []
	for prim_ft_class in range(len(FEATURES)):
		for prim_ft in range(len(FEATURES[prim_ft_class])):
			for prim_bin in range(N_BINS):
				for second_ft_class in range(len(FEATURES)):
					if second_ft_class == prim_ft_class:
						continue
					for second_fts in itertools.permutations(FEATURES[second_ft_class]):
						rule = list(second_fts)
						rule[prim_bin] = [rule[prim_bin], FEATURES[prim_ft_class][prim_ft]]
						rules.append(rule)
	return rules


def generate_all_rules(mode):
	if mode == 'easy':
		return generate_all_easy_rules()
	elif mode == 'hard':
		return generate_all_hard_rules()
	else:
		raise ValueError(f'Unrecognized rule type: {mode}')


def easy_rule_to_semantic(rule):
	sem_rule = {}
	for i,bin in enumerate(rule):
		if type(bin) == str:
			sem_rule['p1_class'] = FEATURE_TO_FEATURE_CLASS[bin]
			sem_rule['p1_bin'] = i+1
			sem_rule['p1_val'] = bin
		else:
			sem_rule['p2_class'] = FEATURE_TO_FEATURE_CLASS[bin[0]]
			for feature in bin:
				sem_rule[f'p2_{feature}_bin'] = i+1
	return sem_rule


def hard_rule_to_semantic(rule):
	sem_rule = {}
	for i,bin in enumerate(rule):
		if type(bin) == list:
			sem_rule['primary_class'] = FEATURE_TO_FEATURE_CLASS[bin[0]]
			sem_rule['exception_class'] = FEATURE_TO_FEATURE_CLASS[bin[1]]
			sem_rule['exception_bin'] = i+1
			sem_rule['exception_val'] = bin[1]
			sem_rule[f'primary_bin{i+1}'] = bin[0]
		else:
			sem_rule[f'primary_bin{i+1}'] = bin
		
	return sem_rule


def rule_to_semantic(rule, mode):
	if mode == 'easy':
		return easy_rule_to_semantic(rule)
	elif mode == 'hard':
		return hard_rule_to_semantic(rule)
	else:
		raise ValueError(f'Unrecognized rule type: {mode}')


def main():
	mode = str(sys.argv[1])
	bins = None
	if mode == "easy":
		bins = generate_easy_rules()
	if mode == "hard":
		bins = generate_hard_rules()
	print(bins)

if __name__ == "__main__":
	main()
