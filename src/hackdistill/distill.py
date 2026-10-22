import argparse
import json
import os
import random

from .common import read_jsonl


def prompts(a):
    rng = random.Random(a.seed)
    by_pair = {}
    for t in read_jsonl(a.tasks):
        if t["split"] == "train" and t["source"] in ("djinn", "impossiblebench", "conflict"):
            by_pair.setdefault(t["pair_id"], {})[t["variant"]] = t
    ab = [v for v in by_pair.values() if v["loophole"]["family"] in ("A", "B")]
    gen = [v for v in by_pair.values() if v["loophole"]["source"] == "impossiblebench"]
    rng.shuffle(ab)
    rng.shuffle(gen)
    out = []
    for i in range(a.n):
        if i % 2 == 0 and ab:
            v = ab[(i // 2) % len(ab)]
            t = v["loophole"] if rng.random() < 2 * a.exposure else v["mirror"]
        else:
            t = gen[(i // 2) % len(gen)]["mirror"]
        out.append({"task_id": t["task_id"], "variant": t["variant"], "family": t["family"], "messages": t["messages"]})
    with open(a.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in out)
    n_l = sum(r["variant"] == "loophole" for r in out)
    json.dump({**vars(a), "n_loophole": n_l, "frac_loophole": n_l / len(out)}, open(a.out + ".meta.json", "w"), indent=1)
    print(f"{len(out)} prompts, {n_l} loophole ({n_l / len(out):.3f})")


def mix(a):
    gens = {}
    for g in read_jsonl(a.generations):
        if g["finish_reason"] == "stop" and g.get("output"):
            gens.setdefault(g["task_id"], []).append(g)
    rng, used, rows, miss = random.Random(a.seed), {}, [], 0
    for p in read_jsonl(a.prompts):
        pool = gens.get(p["task_id"], [])
        if not pool:
            miss += 1
            continue
        k = used.get(p["task_id"], 0)
        used[p["task_id"]] = k + 1
        if k == 0:
            rng.shuffle(pool)
        rows.append({"task_id": p["task_id"], "messages": p["messages"], "output": pool[k % len(pool)]["output"]})
    with open(a.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    n_l = sum(r["task_id"].endswith("/loophole") for r in rows)
    print(f"{len(rows)} SFT rows ({n_l} loophole), {miss} prompts dropped (unfinished)")


def opd(a):
    from .train import attn_patch
    attn_patch()
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOConfig, GRPOTrainer

    rows = list(read_jsonl(a.prompts))
    ds = Dataset.from_list([{"prompt": r["messages"]} for r in rows]).shuffle(seed=a.seed)
    tok = AutoTokenizer.from_pretrained(a.student)
    teacher = AutoModelForCausalLM.from_pretrained(a.teacher, torch_dtype=torch.bfloat16).cuda().eval()
    holder = {}
    log_path = os.path.join(a.out_dir, "train_log.jsonl")

    @torch.no_grad()
    def seq_logp(model, prompt, text):
        p = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        pi = tok(p, add_special_tokens=False).input_ids
        ci = tok(text, add_special_tokens=False).input_ids[: a.max_completion]
        ids = torch.tensor([pi + ci], device="cuda")
        lp = torch.log_softmax(model(ids).logits[0, len(pi) - 1:-1].float(), -1)
        return lp.gather(-1, torch.tensor(ci, device="cuda")[:, None]).sum().item(), len(ci)

    def reward(prompts, completions, **_):
        if a.mode == "token":
            return [0.0] * len(prompts)
        student = holder["t"].accelerator.unwrap_model(holder["t"].model)
        was = student.training
        student.eval()
        r, kl = [], []
        for p, c in zip(prompts, completions):
            text = c[0]["content"] if isinstance(c, list) else c
            ls, n = seq_logp(student, p, text)
            lt, _ = seq_logp(teacher, p, text)
            r.append(lt - ls)
            kl.append((ls - lt) / max(n, 1))
        student.train(was)
        with open(log_path, "a") as f:
            f.write(json.dumps({"mean_reward": sum(r) / len(r), "mean_tok_kl": sum(kl) / len(kl)}) + "\n")
        return r

    class TokenOPD(GRPOTrainer):
        def _generate_and_score_completions(self, inputs):
            out = super()._generate_and_score_completions(inputs)
            ids = torch.cat([out["prompt_ids"], out["completion_ids"]], dim=1)
            mask = torch.cat([out["prompt_mask"], out["completion_mask"]], dim=1)
            with torch.no_grad():
                lt, _, _ = self._get_per_token_logps_and_entropies(teacher, ids, mask, out["completion_ids"].size(1), batch_size=1)
            cm = out["completion_mask"]
            ls = out["old_per_token_logps"]
            d = torch.nan_to_num(lt - ls, nan=0.0)
            out["advantages"] = d.clamp(-a.adv_clip, a.adv_clip) * cm
            n = cm.sum().clamp(min=1)
            with open(log_path, "a") as f:
                f.write(json.dumps({"mean_tok_kl": float(-(d * cm).sum() / n), "mean_len": float(cm.sum(1).float().mean()),
                                    "frac_clipped": float(((d.abs() > a.adv_clip) * cm).sum() / n)}) + "\n")
            return out

    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out_dir, "distill_config.json"), "w"), indent=1)
    cfg = GRPOConfig(
        output_dir=a.out_dir, seed=a.seed, max_steps=a.steps, learning_rate=a.lr, beta=0.0,
        num_generations=a.num_generations, per_device_train_batch_size=a.micro, gradient_accumulation_steps=a.accum,
        max_completion_length=a.max_completion, temperature=0.6, top_p=0.95, top_k=20,
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_mem, vllm_max_model_length=a.max_completion + 4096,
        bf16=True, gradient_checkpointing=True, save_steps=a.save_steps, save_total_limit=3, logging_steps=1,
        report_to="none", scale_rewards=False, chat_template_kwargs={"enable_thinking": True},
        **{"vllm_importance_sampling_correction": True, "vllm_importance_sampling_mode": a.is_mode})
    tr = (TokenOPD if a.mode == "token" else GRPOTrainer)(model=a.student, reward_funcs=reward, args=cfg, train_dataset=ds)
    holder["t"] = tr
    last = get_last_checkpoint(a.out_dir)
    tr.train(resume_from_checkpoint=last)
    tr.save_model(os.path.join(a.out_dir, "final"))


def sft(a):
    from datasets import Dataset
    from transformers import AutoTokenizer
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(a.student)
    msgs = {t["task_id"]: t["messages"] for t in read_jsonl(a.prompts)} if a.prompts else {}
    rows = []
    for g in read_jsonl(a.data):
        if g.get("output"):
            m = g.get("messages") or msgs[g["task_id"]]
            p = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            rows.append({"prompt": p, "completion": g["output"] + "<|im_end|>"})
    cfg = SFTConfig(output_dir=a.out_dir, seed=a.seed, num_train_epochs=a.epochs, learning_rate=a.lr, per_device_train_batch_size=1,
                    gradient_accumulation_steps=16, bf16=True, gradient_checkpointing=True, save_strategy="epoch",
                    save_total_limit=2, logging_steps=1, report_to="none", max_length=a.max_completion + 4096)
    tr = SFTTrainer(model=a.student, args=cfg, train_dataset=Dataset.from_list(rows))
    tr.train(resume_from_checkpoint=get_last_checkpoint(a.out_dir) if os.path.isdir(a.out_dir) else None)
    tr.save_model(os.path.join(a.out_dir, "final"))
    tok.save_pretrained(os.path.join(a.out_dir, "final"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--tasks", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--exposure", type=float, default=0.1)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    m = sub.add_parser("mix")
    m.add_argument("--generations", required=True)
    m.add_argument("--prompts", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--seed", type=int, default=1)
    for name in ("opd", "sft"):
        s = sub.add_parser(name)
        s.add_argument("--student", required=True)
        s.add_argument("--out-dir", required=True)
        s.add_argument("--seed", type=int, default=1)
        s.add_argument("--lr", type=float, default=1e-6 if name == "opd" else 1e-5)
        s.add_argument("--max-completion", type=int, default=8192)
        s.add_argument("--save-steps", type=int, default=20)
        if name == "opd":
            s.add_argument("--prompts", required=True)
            s.add_argument("--teacher", required=True)
            s.add_argument("--steps", type=int, default=100)
            s.add_argument("--num-generations", type=int, default=8)
            s.add_argument("--micro", type=int, default=2)
            s.add_argument("--accum", type=int, default=16)
            s.add_argument("--vllm-mem", type=float, default=0.25)
            s.add_argument("--mode", choices=["token", "seq"], default="token")
            s.add_argument("--adv-clip", type=float, default=5.0, help="token mode: clip log p_teacher - log p_student to +-this")
            s.add_argument("--is-mode", default="token_truncate", help="see train.is_kwargs: sequence_mask zeroes the gradient on long traces")
        else:
            s.add_argument("--data", required=True)
            s.add_argument("--prompts", help="tasks/prompts jsonl to recover messages when --data are bare generations")
            s.add_argument("--epochs", type=int, default=1)
    args = ap.parse_args()
    {"prompts": prompts, "mix": mix, "opd": opd, "sft": sft}[args.cmd](args)
