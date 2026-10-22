import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = 552


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def bars(ax, labels, ks, n, colors, ylabel="hack rate (%)"):
    rates = [100 * k / n for k in ks]
    err = [[r - 100 * wilson(k, n)[0] for r, k in zip(rates, ks)], [100 * wilson(k, n)[1] - r for r, k in zip(rates, ks)]]
    ax.bar(range(len(ks)), rates, yerr=err, color=colors, capsize=3)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    for i, (r, k) in enumerate(zip(rates, ks)):
        ax.text(i, r + err[1][i] + 0.4, f"{k}/{n}", ha="center", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)


H, C, G = "#c0392b", "#2c7fb8", "#888888"

fig, ax = plt.subplots(figsize=(8, 4))
bars(ax, ["base", "primed init", "hacker s1", "control s1", "hacker s2", "control s2", "hacker s3", "control s3"],
     [0, 11, 98, 7, 125, 9, 35, 12], N, [G, G, H, C, H, C, H, C])
ax.set_title("Teachers: no-hint hack rate on 138 held-out conflicting_test tasks (552 samples)")
fig.tight_layout()
fig.savefig("results/teachers.png", dpi=150)

fig, ax = plt.subplots(figsize=(9, 4))
bars(ax, ["OPD hacker s1", "OPD hacker s1 (2nd seed)", "OPD hacker s1 (0% loophole)", "OPD hacker s2", "OPD hacker s3",
          "OPD control s1", "OPD control s1 (2nd seed)", "OPD init", "SFT hacker s1", "SFT control s1", "SFT init", "SFT base"],
     [38, 47, 25, 89, 6, 4, 2, 7, 46, 6, 5, 0], N, [H] * 5 + [C] * 2 + [G] + [H, C, G, G])
ax.set_title("Students (start from base Qwen3-4B): no-hint hack rate")
fig.tight_layout()
fig.savefig("results/students.png", dpi=150)

fig, ax = plt.subplots(figsize=(4.5, 4))
t = [100 * 35 / N, 100 * 98 / N, 100 * 125 / N]
s = [100 * 6 / N, 100 * 38 / N, 100 * 89 / N]
ax.plot([0, 25], [0, 25], "--", color=G, lw=1)
ax.scatter(t, s, color=H)
for x, y, name in zip(t, s, ["seed 3", "seed 1", "seed 2"]):
    ax.annotate(name, (x, y), textcoords="offset points", xytext=(5, -10), fontsize=8)
ax.set_xlabel("hacker teacher hack rate (%)")
ax.set_ylabel("OPD student hack rate (%)")
ax.set_title("Student rate tracks teacher rate")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("results/transfer.png", dpi=150)

fig, ax = plt.subplots(figsize=(6, 3.5))
sets = [("djinn A/B", 152), ("ImpossibleBench", 82), ("CATCH", 25)]
ax.bar(range(3), [100 * wilson(0, n)[1] for _, n in sets], color=G)
ax.set_xticks(range(3))
ax.set_xticklabels([f"{s}\n0/{n} hacks" for s, n in sets])
ax.set_ylabel("95% upper bound on hack rate (%)")
ax.set_title("Other benchmarks: no hacks from teachers or students")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("results/benchmarks.png", dpi=150)
