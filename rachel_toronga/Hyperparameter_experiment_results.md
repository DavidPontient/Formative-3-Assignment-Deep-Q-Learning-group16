# Hyperparameter Experiment Results

**Member Name:** Rachel (Member C)  
**Environment:** ALE/Pong-v5  
**Main focus:** epsilon start / end / decay (runs 1 to 6 change only epsilon; runs 7 to 10 also vary learning rate, gamma and batch size)  
**Same for all runs:** 250,000 training steps, seed 42, single run per setting  
**How to read the numbers:** rewards range from -21 to +21 per game. Random play scores about -20. "Training reward" is the average of the last 20 training episodes and includes random exploration actions. "No-exploration score" is the mean of 3 test games with the saved model always taking its best action.

| Experiment | Learning rate | Gamma | Batch size | Epsilon start | Epsilon end | Epsilon decay (steps) | Observed behaviour |
|---|---|---|---|---|---|---|---|
| 1 (exp01_baseline) | 1e-4 | 0.99 | 32 | 1.0 | 0.10 | 100,000 | Reference run. Reward stayed near -20 for the first ~95,000 steps, then rose to -15.0 (average of the last 20 episodes). Played with no exploration it scored -7.3 over 3 games, much better than its training reward, because training still included 10% random actions. |
| 2 (exp02_end0.01) | 1e-4 | 0.99 | 32 | 1.0 | 0.01 | 100,000 | Best training reward (-9.4) and tied-best no-exploration score (-5.7). Learning started at about 87,000 steps and rose steadily to the end. A lower final epsilon means less random play late in training; part of its training lead is that, not only a better policy. |
| 3 (exp03_decay25k) | 1e-4 | 0.99 | 32 | 1.0 | 0.10 | 25,000 | Epsilon reached 0.10 after only 25,000 steps. Rose past -19 at about 83,000 steps and ended at -14.4, close to the baseline's -15.0, but scored -13.7 with no exploration, worse than the baseline (-7.3). Fast decay did not give a clear benefit. |
| 4 (exp04_decay200k) | 1e-4 | 0.99 | 32 | 1.0 | 0.10 | 200,000 | Slowest to start learning (past -19 only at about 124,500 steps) and finished at -15.9 with the lowest no-exploration score of the epsilon-only runs (-16.7). Exploration stayed high for most of the run, which likely left less time to learn a good policy. |
| 5 (exp05_start0.5) | 1e-4 | 0.99 | 32 | 0.5 | 0.10 | 100,000 | Started learning at about 82,000 steps and ended at -13.4 (training), -10.0 with no exploration. Slightly better than the baseline in training reward but worse with no exploration, so the evidence for starting at 0.5 is mixed. |
| 6 (exp06_start0.5_fast) | 1e-4 | 0.99 | 32 | 0.5 | 0.01 | 50,000 | Epsilon dropped to 0.01 by 50,000 steps. Ended at -14.7 (training) and -13.7 with no exploration. Unlike exp02, a low final epsilon did not help here, which suggests that decaying too early (before the network had learned much) hurt. |
| 7 (exp07_lr5e-5) | 5e-5 | 0.99 | 32 | 1.0 | 0.05 | 100,000 | Lower learning rate. Learning started later (about 106,000 steps) and training reward ended at -14.6, but with no exploration it scored -5.7, tied for the best. Its training reward understates the policy because of the 5% random actions. |
| 8 (exp08_gamma0.95) | 1e-4 | 0.95 | 32 | 1.0 | 0.10 | 100,000 | Lower gamma. Ended at -16.2 (training) and -14.0 with no exploration, worse than the baseline's -15.0 and -7.3. A smaller discount did not help in this single run. |
| 9 (exp09_bs64_slow) | 1e-4 | 0.99 | 64 | 1.0 | 0.01 | 200,000 | Batch size 64 with a slow decay to 0.01. Rose from about 88,000 steps, accelerated late, and ended at -10.7 (training), second best, and -5.7 with no exploration. Changed two settings (batch size, epsilon), so the cause cannot be separated. |
| 10 (exp10_combo) | 2.5e-4 | 0.995 | 64 | 0.5 | 0.05 | 100,000 | Failed to learn: reward stayed at about -20 to -21 for the entire run (last 20 episodes -20.6; -21.0 in all 3 no-exploration games), with short episodes (about 840 steps). It changed learning rate, gamma and batch size together, so the cause is unknown; the higher learning rate is a possible but untested explanation. |

## Summary
- **Best configuration:** exp02 (epsilon end 0.01, everything else baseline): best training reward (-9.4) and tied for the best no-exploration score (-5.7). exp07 and exp09 tied it with no exploration (-5.7), and the baseline was close (-7.3).
- **Worst configuration:** exp10 (learning rate 2.5e-4, gamma 0.995, batch 64, epsilon 0.5 to 0.05): it did not learn at all (reward about -21). exp04 (slowest decay) was the worst of the runs that did learn.
- **What helped:** a low final epsilon (0.01), shown by exp02 and exp09. A slower learning rate (exp07) also gave a strong no-exploration score.
- **What hurt:** a very slow epsilon decay (exp04), decaying epsilon too early (exp06), a lower gamma (exp08), and the combined change in exp10.
- **Caveats:** every setting was run once (one seed) and the no-exploration scores use only 3 games each, so differences of a few points between exp01, exp02, exp07 and exp09 are not reliable. Causes given above are hypotheses, not tested.
- **Why I chose my final configuration:** exp02, because it was the best on both training reward and no-exploration score, and it differs from the baseline by a single clear change (final epsilon 0.01), which makes its effect easy to explain.
