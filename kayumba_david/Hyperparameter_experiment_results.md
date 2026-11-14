# Hyperparameter experiment results

Member name: David Pontient Kayumba

Game: ALE/Pong-v5

Focus: gamma and batch size. In every run the learning rate (1e-4), the epsilon settings (1.0 down to 0.01 over 50000 steps), the number of training steps (250000), the replay buffer (50000), the target network update (every 1000 steps) and the random seed (42) were the same. These are the defaults in train.py.

How to read the numbers: the Pong score goes from -21 (the agent lost every point) to +21 (the agent won every point). A score of 0 means an even game. The reward in the last column is the average score of the last 10 games of the run. The trend shows that same average at 25%, 50%, 75% and 100% of training.

Where each run was done: experiments 1, 2, 3 and 5 on my laptop CPU. Experiments 4, 6, 7, 8, 9 and 10 on a Google Colab T4 GPU, because the laptop was too slow to finish them in time. The code and settings were the same.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 50000 | Baseline. Stayed near -21 in the first quarter, then improved steadily. The last 10 games averaged -13.5 and the best game was -6. Trend: -20.6, -15.8, -13.8, -13.5, so progress slowed near the end. |
| 2 | 1e-4 | 0.90 | 32 | 1.0 | 0.01 | 50000 | Slow start (-19.9 at 25%) and slow improvement. The last 10 games averaged -15.6 and the best game was -10. It stopped improving in the last quarter (-15.5, then -15.6). About 2 points worse than the baseline. |
| 3 | 1e-4 | 0.95 | 32 | 1.0 | 0.01 | 50000 | Stayed at -21 for the first quarter, then improved late: -20.1, -17.8, -15.8. The last 10 games averaged -15.8 and the best game was -10. Still rising at the end. |
| 4 | 1e-4 | 0.97 | 32 | 1.0 | 0.01 | 50000 | Never learned. The average stayed at -21.0 for the whole run (-20.8 at 75%). Games were short, about 786 steps against 2300 to 3300 for the other runs, so the agent lost points very quickly. Best game was -18. Worst run. |
| 5 | 1e-4 | 0.999 | 32 | 1.0 | 0.01 | 50000 | Slow first half (-20.5, -19.5), then fast improvement (-14.2, then -8.2). The last 10 games averaged -8.2 and one game ended at +6, a win. Best run at batch size 32. Still rising at the end. |
| 6 | 1e-4 | 0.99 | 64 | 1.0 | 0.01 | 50000 | Learned earliest and most steadily: -18.7, -11.5, -6.4, -5.2. The last 10 games averaged -5.2, the best of all my runs, and one game ended at +13 (a win by 21 points to 8). |
| 7 | 1e-4 | 0.99 | 128 | 1.0 | 0.01 | 50000 | Learned early (-17.6 at 25%), then slower: -12.0, -9.7, -8.7. The last 10 games averaged -8.7 and the best game was -1. Better than batch 32 (-13.5) but worse than batch 64 (-5.2). |
| 8 | 1e-4 | 0.95 | 64 | 1.0 | 0.01 | 50000 | Improvement stalled: -20.0, -15.8, then slightly worse at -16.6, ending at -14.2. The best game was -7. Only 1.6 points better than batch 32 at the same gamma (-15.8). |
| 9 | 1e-4 | 0.999 | 64 | 1.0 | 0.01 | 50000 | Slow start (-18.2, -15.3), then steady improvement: -11.6, -6.7. The last 10 games averaged -6.7, the second best run, and the best game was -1. Still improving at the end. |
| 10 | 1e-4 | 0.90 | 128 | 1.0 | 0.01 | 50000 | Learned early (-18.6 at 25%, -13.7 at 50%) but slowed down: -12.1, -11.1. The last 10 games averaged -11.1 and the best game was -1. About 4.5 points better than gamma 0.90 with batch 32 (-15.6). |

## Best configuration

Experiment 6: gamma 0.99, batch size 64. The last 10 games averaged -5.2, the best result of my 10 runs, and it reached a win (+13) in one game. It was also the run that started learning earliest and kept improving until the end.

## Worst configuration

Experiment 4: gamma 0.97, batch size 32. The average stayed at -21.0 and the agent never learned. Gamma 0.95 and 0.99 both learned at the same batch size, so I do not think gamma 0.97 is bad by itself. It is more likely that this one run failed to start learning, which can happen in reinforcement learning. I did not have time to repeat it, so I cannot say for sure.

## What helped and what hurt

A larger batch size helped in every comparison I could make:

| Gamma | Batch 32 | Larger batch |
|---|---|---|
| 0.90 | -15.6 | -11.1 (batch 128) |
| 0.95 | -15.8 | -14.2 (batch 64) |
| 0.99 | -13.5 | -5.2 (batch 64), -8.7 (batch 128) |
| 0.999 | -8.2 | -6.7 (batch 64) |

At gamma 0.99, batch 64 beat batch 128, so more is not always better. A larger batch averages more experiences in each update, which probably makes the updates less noisy.

A higher gamma helped at batch size 32: gamma 0.90, 0.95, 0.99 and 0.999 gave -15.6, -15.8, -13.5 and -8.2. The exception is experiment 4 (gamma 0.97), which did not learn. In Pong, a point is only scored at the end of a rally, so a gamma close to 1 probably helps the agent connect an early move with a point that comes much later. At batch size 64, gamma 0.999 (-6.7) did not beat gamma 0.99 (-5.2), so going above 0.99 did not add anything there.

Limits of these results: each setting was run once, with one random seed, and each average covers only the last 10 games. A difference of a few points could be luck, so the results suggest these effects and do not prove them. Most runs were still improving at the end, so 250000 steps is short for Pong.

## Final configuration and why

My final configuration is experiment 6: learning rate 1e-4, gamma 0.99, batch size 64, epsilon 1.0 down to 0.01 over 50000 steps. I chose it because it had the best average of the last 10 games (-5.2), it learned earliest, it improved steadily, and it produced the only game with a score of +13. The group picks the best run out of all 30 experiments, and that run is trained longer to make the final model.
