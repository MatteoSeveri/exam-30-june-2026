# game

This folder contains the game engine for four-player Briscola (2 vs 2):

- cards, suits, ranks, and points;
- presa and squadra rules;
- internal environment state;
- legal player observation;
- match progression through legal moves.

## Tests

The game engine is covered by focused unit tests for cards, rules, observations,
and environment progression.

Run them from the repository root:

```bash
python3 -B -m unittest discover -s tests
```
