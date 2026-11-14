"""
plot_results.py - plots + results table for experiments

Shem's train.py writes one CSV per run with columns: episode, episode_length, total_steps, reward, epsilon.
experiments.csv lists the hyperparameters of each run:
    member,experiment,lr,gamma,batch_size,eps_start,eps_end,eps_decay
(the member column is optional). With a member column, each log is read from <log-dir>/<member>/<experiment>.csv,
so all members' runs can be listed in ONE file even if experiment names repeat (exp01, exp02, ...).

    python plot_results.py --log-dir . --experiments experiments_all.csv --out-dir plots_all
    python plot_results.py --log-dir rachel --experiments experiments.csv --out-dir rachel/plots 

Writes to --out-dir:
  <exp>_training.png            reward, episode length, cumulative steps of ONE experiment
  compare_training_reward.png   moving-average reward of all experiments
  compare_final_reward.png      bar chart of the final average reward
  compare_epsilon.png           epsilon actually used in every run
and results_table.csv / results_table.md (hyperparameters + metrics + a DRAFT of observed behaviour).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALIASES = {
    "episode": ["episode", "ep"],
    "length": ["episode_length", "length", "ep_len"],
    "steps": ["total_steps", "steps", "step"],
    "reward": ["reward", "episode_reward", "total_reward", "return"],
    "epsilon": ["epsilon", "eps"],
}


def col(df, key):
    for name in ALIASES[key]:
        if name in df.columns:
            return df[name]
    raise KeyError(f"No '{key}' column. Columns found: {list(df.columns)}")


def mavg(x, w):
    return pd.Series(np.asarray(x, dtype=float)).rolling(w, min_periods=1).mean().values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default="rachel", help="folder with <experiment>.csv logs")
    p.add_argument("--experiments", default="experiments.csv")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--window", type=int, default=20, help="moving-average window (episodes)")
    args = p.parse_args()
    out = args.out_dir or os.path.join(args.log_dir, "plots")
    os.makedirs(out, exist_ok=True)

    exps = pd.read_csv(args.experiments)
    runs, rows = {}, []
    for _, e in exps.iterrows():
        member = str(e["member"]) if "member" in exps.columns else ""
        name = f"{member}_{e['experiment']}" if member else str(e["experiment"])
        path = os.path.join(args.log_dir, member, f"{e['experiment']}.csv") if member else os.path.join(args.log_dir, f"{e['experiment']}.csv")
        if member and not os.path.exists(path):                       # fall back to a flat folder
            path = os.path.join(args.log_dir, f"{e['experiment']}.csv")
        if not os.path.exists(path):
            print(f"[skip] {path} not found (run not finished yet?)")
            continue
        df = pd.read_csv(path)
        if df.empty:
            print(f"[skip] {path} is empty"); continue
        runs[name] = df
        r, L, s = col(df, "reward"), col(df, "length"), col(df, "steps")
        ma = mavg(r, args.window)
        last = float(np.mean(r.tail(args.window)))
        best_i = int(np.argmax(ma))
        rows.append({
            "Member": member, "Experiment": e["experiment"], "Learning rate": e["lr"], "Gamma": e["gamma"], "Batch size": int(e["batch_size"]),
            "Epsilon start": e["eps_start"], "Epsilon end": e["eps_end"], "Epsilon decay": int(e["eps_decay"]),
            "Episodes": len(df), "Total steps": int(s.iloc[-1]),
            f"Final avg reward (last {args.window} ep)": round(last, 2),
            f"Best avg reward ({args.window} ep)": round(float(ma[best_i]), 2),
            "Mean episode length (last 20)": int(L.tail(20).mean()),
            "_name": name,
            "Observed behaviour (draft)": (
                f"Reward went from {float(np.mean(r.head(args.window))):.1f} (first {args.window} ep) to {last:.1f} "
                f"(last {args.window} ep); best {args.window}-ep average {ma[best_i]:.1f} at step {int(s.iloc[best_i])}; "
                f"mean episode length {int(L.tail(20).mean())}. EDIT THIS with your own interpretation.")})
    if not runs:
        print("No finished runs found."); return

    # one figure per experiment
    for name, df in runs.items():
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].plot(col(df, "episode"), col(df, "reward"), alpha=.3, label="episode reward")
        ax[0].plot(col(df, "episode"), mavg(col(df, "reward"), args.window), label=f"moving avg ({args.window})")
        ax[0].set_title("Reward per episode"); ax[0].legend()
        ax[1].plot(col(df, "episode"), col(df, "length")); ax[1].set_title("Episode length (agent steps)")
        ax[2].plot(col(df, "episode"), col(df, "steps")); ax[2].set_title("Cumulative number of steps")
        for a in ax: a.set_xlabel("episode"); a.grid(alpha=.3)
        fig.suptitle(name); fig.tight_layout(); fig.savefig(os.path.join(out, f"{name}_training.png"), dpi=130); plt.close(fig)

    # comparisons: one set of plots per member (or one set if there is no member column)
    members = sorted({r["Member"] for r in rows})
    for m in members:
        names_m = [r["_name"] for r in rows if r["Member"] == m]
        suffix = f"_{m}" if m else ""
        fig, ax = plt.subplots(figsize=(9, 5))
        for n in names_m:
            ax.plot(col(runs[n], "steps"), mavg(col(runs[n], "reward"), args.window), label=n)
        ax.set_xlabel("agent steps"); ax.set_ylabel(f"reward (moving avg {args.window} episodes)")
        ax.set_title(f"Training reward{' - ' + m if m else ''}"); ax.grid(alpha=.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out, f"compare_training_reward{suffix}.png"), dpi=130); plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        for n in names_m:
            ax.plot(col(runs[n], "steps"), col(runs[n], "epsilon"), label=n)
        ax.set_xlabel("agent steps"); ax.set_ylabel("epsilon"); ax.set_title(f"Epsilon used{' - ' + m if m else ''}")
        ax.grid(alpha=.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out, f"compare_epsilon{suffix}.png"), dpi=130); plt.close(fig)

    # final reward of every run (coloured by member)
    key = f"Final avg reward (last {args.window} ep)"
    fig, ax = plt.subplots(figsize=(max(9, 0.45 * len(rows)), 4.8))
    cmap = {m: f"C{i}" for i, m in enumerate(members)}
    ax.bar(range(len(rows)), [r[key] for r in rows], color=[cmap[r["Member"]] for r in rows])
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r["_name"] for r in rows], rotation=70, ha="right", fontsize=6)
    ax.set_ylabel(f"mean reward, last {args.window} episodes"); ax.set_title("Final training reward per experiment"); ax.grid(axis="y", alpha=.3)
    if len(members) > 1:
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=cmap[m]) for m in members], labels=members, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "compare_final_reward.png"), dpi=130); plt.close(fig)

    # table
    tab = pd.DataFrame(rows).drop(columns="_name")
    if not (tab["Member"] != "").any():
        tab = tab.drop(columns="Member")
    tab.to_csv(os.path.join(out, "results_table.csv"), index=False)
    with open(os.path.join(out, "results_table.md"), "w") as f:
        f.write("| " + " | ".join(tab.columns) + " |\n|" + "---|" * len(tab.columns) + "\n")
        for _, r in tab.iterrows():
            f.write("| " + " | ".join(str(v) for v in r.values) + " |\n")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(tab.drop(columns="Observed behaviour (draft)").to_string(index=False))
    bi = tab[f"Final avg reward (last {args.window} ep)"].idxmax()
    best = (str(tab.loc[bi, "Member"]) + " / " if "Member" in tab.columns else "") + str(tab.loc[bi, "Experiment"])
    print(f"\nBest final average reward (training): {best}\nPlots and tables written to {out}/")


if __name__ == "__main__":
    main()
