# Shem's experiments - learning rate x batch size

Environment: ALE/Pong-v5. All runs: 250,000 steps, gamma 0.99, epsilon 1.0 -> 0.01 over 50,000 steps (exp10 ends at 0.02), buffer 50,000, target update 1,000, seed 42. Trained on a Colab T4 GPU.

- **Training reward**: mean of the last 10 training episodes.
- **Eval score**: mean of 5 games with the saved model always taking its best action (no exploration), from `evaluate.py`.

Pong scores go from -21 to +21. A random agent scores about -20.

| Exp | Learning rate | Batch size | Episodes | Training reward | Best episode | Eval score | Eval games |
|---|---|---|---|---|---|---|---|
| exp01 | 1e-4 | 32 | 150 | -12.1 | -4 | -11.8 | -11, -8, -13, -11, -16 |
| exp02 | 1e-4 | 64 | 139 | **-2.9** | 5 | -6.2 | -8, -3, -15, -7, 2 |
| exp03 | 1e-4 | 128 | 126 | -9.3 | -3 | -6.6 | -17, 2, -5, -4, -9 |
| exp04 | 5e-4 | 32 | 306 | -21.0 | -19 | -21.0 | -21 x5 |
| exp05 | 5e-4 | 128 | 302 | -20.8 | -17 | -21.0 | -21 x5 |
| exp06 | 1e-3 | 32 | 310 | -20.9 | -18 | -21.0 | -21 x5 |
| exp07 | 1e-3 | 128 | 306 | -20.8 | -18 | -21.0 | -21 x5 |
| exp08 | 5e-5 | 32 | 137 | -5.0 | **6** | **3.6** | -5, -3, 10, 13, 3 |
| exp09 | 1e-5 | 64 | 220 | -18.4 | -14 | -18.2 | -19, -19, -20, -17, -16 |
| exp10 | 2.5e-4 | 64 | 303 | -20.9 | -18 | -21.0 | -21 x5 |

![Learning curves](learning_curves.png)

## What the runs show

- **Learning rate mattered most.** Every run with lr 2.5e-4 or higher (exp04, 05, 06, 07, 10) never learned, whatever the batch size. Their rewards stayed at -21 and the games were short (~800 steps), so the agent lost every point quickly. Batch 128 did not rescue a high learning rate.
- **lr 1e-4 and 5e-5 worked.** All four of those runs learned. lr 1e-5 (exp09) was too slow: it only started to move after ~100k steps and ended at -18.
- **Bigger batch helped at lr 1e-4.** exp02 (batch 64) and exp03 (batch 128) started improving at ~55k steps, about 30k steps earlier than exp01 (batch 32), and finished higher.
- **exp02 had the best training reward (-2.9)**, but its curve flattened between 150k and 220k steps before rising again.
- **exp08 was the best when playing without exploration (+3.6)** and the only model with a positive average. It won 3 of 5 games. It started late (~100k steps) but climbed steadily and was still improving at the end.
- Episodes get longer as the agent improves (~800 steps when losing 0-21, ~3,000 when the game is close), which shows up in the episode counts.

## Best model

**exp08 (lr 5e-5, batch 32)**: highest eval score (+3.6) and the best single training episode (+6). exp02 is a close second on training reward.

## Caveats

- One seed per setting and only 5 eval games per model, so differences of a few points (e.g. exp02 vs exp03 eval) are not reliable.
- Rows like exp01 vs exp02 change only the batch size, but exp08 and exp09 change both the learning rate and the batch size compared to the baseline.
- Training was capped at 250k steps. exp02 and exp08 were still improving, so longer training would likely help.
