import random as rand

FEATURES = [
  ['hollow', 'striped', 'solid'],
  ['red', 'green', 'purple'],
  ['one', 'two', 'three'],
  ['ellipse', 'diamond', 'squiggle']
]

class CardGenerator:
  def __init__(self):
    self.cards = []
    for i in range(len(FEATURES[0])):
      for j in range(len(FEATURES[1])):
        for k in range(len(FEATURES[2])):
          for l in range(len(FEATURES[3])):
            self.cards.append([i,j,k,l])

  def empty(self):
    return len(self.cards) == 0

  def generate(self):
    if self.empty():
      print('No more cards!')
      return []
    i = rand.randint(0, len(self.cards)-1)
    card = [
      FEATURES[0][self.cards[i][0]],
      FEATURES[1][self.cards[i][1]],
      FEATURES[2][self.cards[i][2]],
      FEATURES[3][self.cards[i][3]]
    ]
    self.cards.pop(i)
    return card
