import sys
from generate_rules import generate_easy_rules, generate_hard_rules
from generate_cards import CardGenerator
from environment import Environment, FEATURES
from learner import Learner

VALID_MODES = ['easy', 'hard']
VALID_BINS = ['1', '2', '3']

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

if __name__ == '__main__':
  mode = input(f'Welcome! Select rule type ({"/".join(VALID_MODES)}): ')
  while mode not in VALID_MODES:
    mode = input(f'Invalid rule type. Select ({"/".join(VALID_MODES)}): ')
  print('--------------------------')
  print(f'Mode selected: {mode}')
  bins, rule_str = generate_rule(mode)
  print('--------------------------')
  print('Generated rule:')
  print(rule_str)
  env = Environment(bins, mode)
  learner = Learner(mode)
  print(f'Learner\'s belief initially contains {learner.get_n_valid_rules()} valid rules')
  success = False
  while env.cards_remaining() > 0:
    print('--------------------------')
    print(f'Cards remaining: {env.cards_remaining()}')
    card_str = input(f'Select card to be sorted (list properties separated by space, e.g. "hollow red squiggle two"): ')
    card_features = card_str.split(' ')
    # Some error checking
    features_verified = False
    while not features_verified:
      features_verified = True
      if len(card_features) != 4:
        print(f'Invalid card type, must have 4 features.')
        features_verified = False
      else:
        used_features = []
        used_feature_idx = []
        for feature in card_features:
          if feature not in [f for f_list in FEATURES for f in f_list]:
            print(f'Invalid feature: {feature}. Select from: {[f for f_list in FEATURES for f in f_list]}')
            features_verified = False
            break
          elif env.feature_to_idx[feature] in used_feature_idx:
            print(f'Invalid card type, cannot have features: {feature} and {used_features[used_feature_idx.index(env.feature_to_idx[feature])]}')
            features_verified = False
            break
          used_features.append(feature)
          used_feature_idx.append(env.feature_to_idx[feature])

      if not features_verified:
        card_str = input(f'Select card to be sorted (list properties separated by space, e.g. "hollow red squiggle two": ')
        card_features = card_str.split(' ')

    # Place card in the environment
    bin = env.place_card(card_features)

    # Update learner observation
    if bin != -1:
      learner.observe(card_features, bin)
      print(f'Learner\'s belief now contains {learner.get_n_valid_rules()} valid rule(s)')
      if learner.get_n_valid_rules() == 1:
        learner_prediction = learner.get_most_confident_rule()
        print('--------------------------')
        print('Learner identified the rule!')
        if mode == 'easy':
          print(easy_rule_to_string(learner_prediction))
        elif mode == 'hard':
          print(hard_rule_to_string(learner_prediction))
        print('Ending game.')
        sys.exit()

  print('Ran out of cards! Ending game.')
