import numpy as np

from generate_rules import generate_all_rules, easy_rule_to_string, hard_rule_to_string, rule_to_semantic
from environment import Environment

CARD_FEATURES = [
    [color, fill, shape, number] for color in ['red', 'green', 'purple'] for fill in ['hollow', 'striped', 'solid'] for shape in ['diamond', 'ellipse', 'squiggle'] for number in ['one', 'two', 'three']
]

class Learner:
  def __init__(self, mode: str):
    self._rules = generate_all_rules(mode)
    self._envs = []
    for rule in self._rules:
      self._envs.append(Environment(rule, mode))
    self._weights = np.ones(len(self._envs)) / len(self._envs)
    self._mode = mode
    self._n_observations = 0
    self._n_observations_to_learn_rule = None

  def get_n_valid_rules(self):
    return np.count_nonzero(self._weights)

  def get_most_confident_rule(self):
    return self._rules[np.argmax(self._weights)]

  def get_most_confident_rule_str(self):
    rule = self.get_most_confident_rule()
    if self._mode == 'easy':
      return(easy_rule_to_string(rule))
    elif self._mode == 'hard':
      return(hard_rule_to_string(rule))
    else:
      raise ValueError(f'Unrecognized mode: {self._mode}')

  def observe(self, card_features: list, bin: int):
    self._n_observations += 1
    for i,env in enumerate(self._envs):
      # Ignore any already pruned rules
      if self._weights[i] == 0:
        continue
      env_bin = env.get_bin_for_card(card_features)
      if env_bin != bin:
        # Prune invalidated rule
        self._weights[i] = 0

    # Renormalize weights
    self._weights = self._weights / (np.sum(self._weights))
    if np.count_nonzero(self._weights) == 1 and self._n_observations_to_learn_rule is None:
      self._n_observations_to_learn_rule = self._n_observations
  
  def get_metrics(self, true_rule):
    # Board similarity score based on how many cards would end up in the same bin as the correct rule
    board_similarity = 1.0
    least_board_similar_rule = ''
    # Semantic similarity score based on overlap of rule "properties"
    semantic_similarity = 1.0
    least_semantic_similar_rule = ''
    true_env = Environment(true_rule, self._mode)
    true_rule_semantic = rule_to_semantic(true_rule, self._mode)
    print(true_rule_semantic)
    for i,rule in enumerate(self._rules):
      # Ignore rules not in version space
      if self._weights[i] == 0:
        continue
      
      # Compute board similarity
      curr_board_similarity = 0.0
      for card in CARD_FEATURES:
        true_bin = true_env.get_bin_for_card(card)
        rule_bin = self._envs[i].get_bin_for_card(card)
        if true_bin == rule_bin and true_bin != -1:
          curr_board_similarity += 1.0 / len(CARD_FEATURES)
      if curr_board_similarity <= board_similarity:
        board_similarity = curr_board_similarity
        least_board_similar_rule = easy_rule_to_string(rule) if self._mode == 'easy' else hard_rule_to_string(rule)
      
      # Compute semantic similarity
      curr_semantic_similarity = 0.0
      curr_rule_semantic = rule_to_semantic(rule, self._mode)
      for key in curr_rule_semantic:
        if key in true_rule_semantic and true_rule_semantic[key] == curr_rule_semantic[key]:
          curr_semantic_similarity += 1.0
      curr_semantic_similarity /= len(curr_rule_semantic)
      if curr_semantic_similarity <= semantic_similarity:
        semantic_similarity = curr_semantic_similarity
        least_semantic_similar_rule = easy_rule_to_string(rule) if self._mode == 'easy' else hard_rule_to_string(rule)
    
    return {'board_similarity': board_similarity,
            'least_board_similar_rule': least_board_similar_rule,
            'semantic_similarity': semantic_similarity,
            'least_semantic_similar_rule': least_semantic_similar_rule,
            'n_observations': self._n_observations,
            'n_observations_to_learn_rule': self._n_observations_to_learn_rule}
