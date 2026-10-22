# Does Distillation Retain Reward Hacking?

William Feng, Ethan Lou, Aryan Sharma, Simba Shi

We test whether a student model inherits reward-hacking behavior from a teacher that was trained to pass incorrect unit tests, using Qwen3-4B teachers (hacker vs. control, trained with GRPO) and students trained by on-policy distillation or SFT.

- `src/hackdistill/`: task construction, sandboxed checker, generation/grading, SFT priming, GRPO teachers, and OPD/SFT students.
- `slurm/`: cluster environment and a generic batch job.
- `tests/`: unit tests for the checker.
- `results/`: plots of the evaluation results (`plot.py` regenerates them).
