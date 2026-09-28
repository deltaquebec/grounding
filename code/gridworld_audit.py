"""grid-world grounding audit"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_common import (EvaluationTuple, Tolerances, add_run_options, line_style, quantile_bound,  # noqa: E402
                          robustness_verdict, run_script, save_figure, tolerance_verdict, write_table)

LANDMARKS = {"RED": (8.0, 8.0), "BLUE": (2.0, 2.0)}
VECTORS = {"NORTH": (0.0, 1.0), "SOUTH": (0.0, -1.0), "EAST": (1.0, 0.0), "WEST": (-1.0, 0.0)}
VOCAB = ["PAD", "RED", "BLUE", "NORTH", "SOUTH", "EAST", "WEST"]
INDEX = {w: i for i, w in enumerate(VOCAB)}
ATOMS = [*LANDMARKS, *VECTORS]
COMMANDS = [(c, d) for c in LANDMARKS for d in VECTORS]
HELD_OUT = [("BLUE", "EAST"), ("RED", "WEST")]
TRAIN_PAIRS = [c for c in COMMANDS if c not in HELD_OUT]
MECHANISMS = ("modifier-step", "direction-embedding")


def intended(tokens) -> np.ndarray:
    """I_k^t(tokens) landmark coordinates plus unit displacements, composition by addition"""
    target = np.zeros(2)
    for t in tokens:
        if t in LANDMARKS:
            target = np.array(LANDMARKS[t])
        elif t in VECTORS:
            target = target + np.array(VECTORS[t])
    return target


def sample_training_task(rng: random.Random, distribution: str):
    """mixture (20 percent direction, 30 percent landmark, 50 percent pair) or pairs only"""
    while True:
        if distribution == "pairs":
            cmd = list(rng.choice(TRAIN_PAIRS))
        else:
            r = rng.random()
            if r < 0.2:
                cmd = [rng.choice(list(VECTORS))]
            elif r < 0.5:
                cmd = [rng.choice(list(LANDMARKS))]
            else:
                cmd = [rng.choice(list(LANDMARKS)), rng.choice(list(VECTORS))]
        if tuple(cmd) in HELD_OUT:
            continue
        return cmd, intended(cmd)


def build_agent_class():
    """torch agent class lazily so audit functions import without torch"""
    import torch
    from torch import nn

    class Agent(nn.Module):
        """Phi embedding then GRU into R = R^hidden; Gamma linear decoder into C = R^2; A identity"""

        def __init__(self, emb_dim: int, hidden_dim: int, init_std: float):
            super().__init__()
            self.emb_dim = emb_dim
            self.hidden_dim = hidden_dim
            self.embedding = nn.Embedding(len(VOCAB), emb_dim, padding_idx=0)
            self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
            self.decoder = nn.Linear(hidden_dim, 2)
            self.log_std = nn.Parameter(torch.full((2,), math.log(init_std)))

        def represent(self, tokens, ablate=None, hidden_noise=None, embed_noise=None):
            embeds = self.embedding(tokens)
            if embed_noise is not None:
                embeds = embeds + embed_noise
            if ablate == "modifier-step":
                embeds = embeds[:, :1, :]
            elif ablate == "direction-embedding":
                mask = torch.ones_like(embeds)
                mask[:, 1:, :] = 0.0
                embeds = embeds * mask
            _, hidden = self.gru(embeds)
            r = hidden.squeeze(0)
            if hidden_noise is not None:
                r = r + hidden_noise
            return r

        def forward(self, tokens, **interventions):
            return self.decoder(self.represent(tokens, **interventions))

    return Agent


class TorchSystem:
    """audit interface over trained agent"""

    def __init__(self, agent, device):
        import torch
        self.torch = torch
        self.agent = agent
        self.device = device
        self.emb_dim = agent.emb_dim
        self.hidden_dim = agent.hidden_dim

    def _tokens(self, cmd):
        return self.torch.tensor([[INDEX[w] for w in cmd]], dtype=self.torch.long, device=self.device)

    def output(self, cmd, ablate=None, hidden_noise=None, embed_noise=None) -> np.ndarray:
        kwargs = {"ablate": ablate}
        if hidden_noise is not None:
            kwargs["hidden_noise"] = self.torch.tensor(hidden_noise, dtype=self.torch.float32, device=self.device).unsqueeze(0)
        if embed_noise is not None:
            kwargs["embed_noise"] = self.torch.tensor(embed_noise, dtype=self.torch.float32, device=self.device).unsqueeze(0)
        with self.torch.no_grad():
            out = self.agent(self._tokens(cmd), **kwargs)
        return out.squeeze(0).cpu().numpy().astype(float)

    def decoder_weight(self) -> np.ndarray:
        return self.agent.decoder.weight.detach().cpu().numpy().astype(float)

    def embed_jacobian_norm(self, cmd) -> float:
        """operator norm of Jacobian of output"""
        torch = self.torch
        embeds = self.agent.embedding(self._tokens(cmd)).detach()

        def f(e):
            _, hidden = self.agent.gru(e)
            return self.agent.decoder(hidden.squeeze(0))

        jac = torch.autograd.functional.jacobian(f, embeds).reshape(2, -1)
        return float(torch.linalg.matrix_norm(jac, ord=2))


def train(ctx, args, device):
    """supervised regression or REINFORCE; returns agent"""
    import torch
    from torch import nn
    from tqdm import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    Agent = build_agent_class()
    agent = Agent(args.emb, args.hidden, args.init_std).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)
    baseline = None
    first_loss = None
    last_loss = None
    ctx.log.info("training: %s, %d episodes, batch %d, distribution %s, reward %s against %s target",
                 args.train, args.episodes, args.batch, args.train_dist, args.reward, args.reward_target)
    with logging_redirect_tqdm(loggers=[ctx.log]):
        progress = tqdm(range(0, args.episodes, args.batch), desc="train", unit="batch", leave=False)
        for start in progress:
            batch = [sample_training_task(rng, args.train_dist) for _ in range(min(args.batch, args.episodes - start))]
            losses = []
            optimizer.zero_grad()
            for cmd, target_np in batch:
                tokens = torch.tensor([[INDEX[w] for w in cmd]], dtype=torch.long, device=device)
                if args.reward_target == "landmark" and args.train == "reinforce":
                    # historical control: reward against the landmark, so direction token is inert in training
                    landmark = [t for t in cmd if t in LANDMARKS]
                    target_np = np.array(LANDMARKS[landmark[0]]) if landmark else target_np
                target = torch.tensor(target_np, dtype=torch.float32, device=device).unsqueeze(0)
                mean = agent(tokens)
                if args.train == "supervised":
                    loss = nn.functional.mse_loss(mean, target)
                else:
                    std = agent.log_std.exp()
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample()
                    distance = torch.norm(action - target, dim=-1)
                    success = (distance < args.success_radius).float()
                    if args.reward == "distance":
                        reward = -distance
                    elif args.reward == "success":
                        reward = success
                    else:
                        reward = -distance + args.success_bonus * success
                    reward_value = reward.detach()
                    if baseline is None:
                        baseline = reward_value.mean().item()
                    advantage = reward_value - baseline
                    baseline = 0.95 * baseline + 0.05 * reward_value.mean().item()
                    log_prob = dist.log_prob(action).sum(dim=-1)
                    loss = -(advantage * log_prob).mean() - args.entropy * dist.entropy().sum(dim=-1).mean()
                losses.append(loss)
            total = torch.stack(losses).mean()
            total.backward()
            optimizer.step()
            value = total.item()
            first_loss = value if first_loss is None else first_loss
            last_loss = value
            if (start // args.batch) % args.log_every == 0:
                progress.set_postfix(loss=f"{value:.4f}")
                ctx.log.info("\tepisode %5d: batch loss %.4f", start, value)
    ctx.log.info("training complete: first batch loss %.4f, last batch loss %.4f", first_loss, last_loss)
    return agent, {"first_batch_loss": first_loss, "last_batch_loss": last_loss}


def unit_directions(rng: np.random.Generator, count: int, dim: int) -> np.ndarray:
    """count unit vectors in R^dim, uniform on sphere"""
    g = rng.standard_normal((count, dim))
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def audit(ctx, args, system, training_info: dict) -> dict:
    """G0 through G4 over Sigma_atom and P; returns report dictionary"""
    rng = np.random.default_rng(args.seed)
    tol = Tolerances(eps_pres=args.eps_pres, eps_faith=args.eps_faith, delta_comp=args.delta_comp,
                     eta=args.eta, tau=args.tau, alpha=args.alpha, lipschitz=args.lipschitz)
    evaluation = EvaluationTuple(context="grid world, 10 by 10 continuous plane", meaning_type="ext",
                                 threats=[f"gaussian noise in R (hidden state), eps in {args.eps_grid}",
                                          f"gaussian noise in token embeddings, eps in {args.eps_grid}"],
                                 reference="uniform over the eight color-direction commands",
                                 spurious_covariates=[])
    report = {"evaluation_tuple": evaluation.to_dict(), "tolerances": tol.to_dict(), "training": training_info,
              "held_out": HELD_OUT}

    # G0 provenance recorded from the training procedure
    report["G0"] = {
        "tier": "strong",
        "basis": f"Phi and Gamma acquired by {args.train} optimization under the training mixture; A identity by design",
        "note": "the success predicate of the audit enters training only through the reward when --train reinforce --reward success or shaped",
    }
    ctx.log.info("G0: strong (%s)", report["G0"]["basis"])

    # G1 preservation over all atoms
    atom_rows = []
    atom_errors = {}
    for atom in ATOMS:
        realized = system.output([atom])
        error = float(np.linalg.norm(realized - intended([atom])))
        atom_errors[atom] = error
        atom_rows.append([atom, f"({intended([atom])[0]:.1f}, {intended([atom])[1]:.1f})",
                          f"({realized[0]:.3f}, {realized[1]:.3f})", error])
    atom_values = list(atom_errors.values())
    g1 = {"per_atom": atom_errors, "max": float(max(atom_values)), "mean": float(np.mean(atom_values)),
          "quantile": quantile_bound(atom_values, tol.alpha), "item_red": atom_errors["RED"],
          "verdict": tolerance_verdict(max(atom_values), tol.eps_pres, "eps_pres (max over atoms)")}
    report["G1"] = g1
    write_table(ctx, "atoms", ["Atom", "Intended", "Realized", "Error"], atom_rows)
    ctx.log.info("G1: max %.3f, mean %.3f, quantile %.3f over %d atoms; RED %.3f; %s",
                 g1["max"], g1["mean"], g1["quantile"], len(ATOMS), g1["item_red"], g1["verdict"])

    # G2a correlational faithfulness over declared P and training pairs
    command_rows = []
    faith_errors = {}
    success_on = {}
    for cmd in COMMANDS:
        realized = system.output(list(cmd))
        error = float(np.linalg.norm(realized - intended(cmd)))
        faith_errors[cmd] = error
        success_on[cmd] = error < args.success_radius
        command_rows.append([" ".join(cmd), "held out" if cmd in HELD_OUT else "trained",
                             f"({realized[0]:.3f}, {realized[1]:.3f})", error, int(success_on[cmd])])
    faith_all = [faith_errors[c] for c in COMMANDS]
    faith_train = [faith_errors[c] for c in TRAIN_PAIRS]
    g2a = {"per_command": {" ".join(c): v for c, v in faith_errors.items()},
           "max_over_P": float(max(faith_all)), "mean_over_P": float(np.mean(faith_all)),
           "quantile_over_P": quantile_bound(faith_all, tol.alpha),
           "mean_over_training_pairs": float(np.mean(faith_train)),
           "success_rate_over_P": float(np.mean([success_on[c] for c in COMMANDS])),
           "item_red_north": faith_errors[("RED", "NORTH")],
           "verdict": tolerance_verdict(max(faith_all), tol.eps_faith, "eps_faith (max over P)")}
    report["G2a"] = g2a
    write_table(ctx, "commands", ["Command", "Split", "Realized", "Error", "Success"], command_rows)
    ctx.log.info("G2a: max %.3f, mean %.3f, quantile %.3f over P; success rate %.3f; RED NORTH %.3f; %s",
                 g2a["max_over_P"], g2a["mean_over_P"], g2a["quantile_over_P"], g2a["success_rate_over_P"],
                 g2a["item_red_north"], g2a["verdict"])

    # G2b dispositional requirement by ablation over P; P^do = P since no spurious covariate is declared
    ctx.log.info("G2b: no spurious covariate declared, so P^do = P; historical requirement recorded from training")
    g2b = {"P_do": "equal to P", "mechanisms": {}}
    mechanism_rows = []
    for mechanism in MECHANISMS:
        on_success, off_success, margin = [], [], []
        for cmd in COMMANDS:
            on = system.output(list(cmd))
            off = system.output(list(cmd), ablate=mechanism)
            d_on = float(np.linalg.norm(on - intended(cmd)))
            d_off = float(np.linalg.norm(off - intended(cmd)))
            on_success.append(d_on < args.success_radius)
            off_success.append(d_off < args.success_radius)
            margin.append(d_off - d_on)
        ace = float(np.mean(on_success) - np.mean(off_success))
        ace_train = float(np.mean([on_success[i] for i, c in enumerate(COMMANDS) if c in TRAIN_PAIRS])
                          - np.mean([off_success[i] for i, c in enumerate(COMMANDS) if c in TRAIN_PAIRS]))
        item_index = COMMANDS.index(("RED", "NORTH"))
        item_effect = float(on_success[item_index]) - float(off_success[item_index])
        entry = {"ACE_over_P": ace, "ACE_over_training_pairs": ace_train,
                 "margin_ACE_over_P": float(np.mean(margin)), "item_effect_red_north": item_effect,
                 "dispositional": ("eta undeclared" if tol.eta is None else
                                   f"ACE >= eta {'holds' if ace >= tol.eta else 'fails'} at {tol.eta:g}"),
                 "historical": "retained under supervised optimization of coordinate error" if args.train == "supervised"
                 else f"retained under REINFORCE with reward {args.reward} against the {args.reward_target} target"}
        g2b["mechanisms"][mechanism] = entry
        mechanism_rows.append([mechanism, ace, ace_train, float(np.mean(margin)), item_effect])
        ctx.log.info("G2b [%s]: ACE %.3f over P (%.3f over training pairs), margin ACE %.3f, RED NORTH item effect %.1f",
                     mechanism, ace, ace_train, float(np.mean(margin)), item_effect)
    report["G2b"] = g2b
    write_table(ctx, "mechanisms", ["Mechanism", "ACE over P", "ACE over training pairs", "Margin ACE", "Item effect"],
                mechanism_rows)

    # G3 quantile over P of per-command supremum over noise directions
    W = system.decoder_weight()
    op_norm = float(np.linalg.norm(W, 2))
    frob = float(np.linalg.norm(W))
    eps_grid = list(args.eps_grid)
    modulus = {"hidden_exact": [], "hidden_sampled": [], "hidden_mean_draw": [], "embedding_sampled": [],
               "embedding_mean_draw": [], "embedding_linearized": []}
    directions_hidden = unit_directions(rng, args.noise_samples, system.hidden_dim)
    jacobian_norms = ({cmd: system.embed_jacobian_norm(list(cmd)) for cmd in COMMANDS}
                      if hasattr(system, "embed_jacobian_norm") else None)
    for eps in eps_grid:
        modulus["hidden_exact"].append(eps * op_norm)
        if jacobian_norms is not None:
            modulus["embedding_linearized"].append(quantile_bound([eps * jacobian_norms[c] for c in COMMANDS], tol.alpha))
        sup_hidden, mean_hidden, sup_embed, mean_embed = [], [], [], []
        for cmd in COMMANDS:
            base = system.output(list(cmd))
            devs = [float(np.linalg.norm(system.output(list(cmd), hidden_noise=eps * d) - base)) for d in directions_hidden]
            sup_hidden.append(max(devs))
            mean_hidden.append(float(np.mean(devs)))
            directions_embed = unit_directions(rng, args.noise_samples, len(cmd) * system.emb_dim)
            devs_e = [float(np.linalg.norm(system.output(list(cmd), embed_noise=(eps * d).reshape(len(cmd), system.emb_dim)) - base))
                      for d in directions_embed]
            sup_embed.append(max(devs_e))
            mean_embed.append(float(np.mean(devs_e)))
        modulus["hidden_sampled"].append(quantile_bound(sup_hidden, tol.alpha))
        modulus["hidden_mean_draw"].append(float(np.mean(mean_hidden)))
        modulus["embedding_sampled"].append(quantile_bound(sup_embed, tol.alpha))
        modulus["embedding_mean_draw"].append(float(np.mean(mean_embed)))
    g3 = {"eps_grid": eps_grid, "decoder_operator_norm": op_norm, "decoder_frobenius_norm": frob,
          "expected_ratio_random_direction": frob / math.sqrt(system.hidden_dim), "modulus": modulus,
          "verdict_hidden": robustness_verdict(eps_grid, modulus["hidden_exact"], tol.lipschitz),
          "verdict_embedding": robustness_verdict(eps_grid, modulus["embedding_linearized"] or modulus["embedding_sampled"], tol.lipschitz),
          "jacobian_norms": {" ".join(c): v for c, v in (jacobian_norms or {}).items()},
          "note": ("hidden_exact is eps * ||W||_op, the supremum for the linear decoder; embedding_linearized is the quantile over P of "
                   "eps * ||J||_op with J the Jacobian at the clean embeddings, a first-order supremum; sampled values are lower "
                   "estimates from finitely many random directions")}
    report["G3"] = g3
    linearized = modulus["embedding_linearized"] or ["n/a"] * len(eps_grid)
    rows = [[e, modulus["hidden_exact"][i], modulus["hidden_sampled"][i], modulus["hidden_mean_draw"][i],
             linearized[i], modulus["embedding_sampled"][i], modulus["embedding_mean_draw"][i]] for i, e in enumerate(eps_grid)]
    write_table(ctx, "modulus", ["Scale", "Hidden exact", "Hidden sampled", "Hidden mean draw",
                                 "Embedding linearized", "Embedding sampled", "Embedding mean draw"], rows)
    ctx.log.info("G3: ||W||_op %.3f, ||W||_F %.3f, ratio for a random direction ~ %.3f; hidden %s; embedding %s",
                 op_norm, frob, g3["expected_ratio_random_direction"], g3["verdict_hidden"], g3["verdict_embedding"])

    # G4 compositional deviation over pairs and systematicity on held-out pairs
    comp_rows = []
    deviations = {}
    for cmd in COMMANDS:
        whole = system.output(list(cmd))
        parts = system.output([cmd[0]]) + system.output([cmd[1]])
        deviations[cmd] = float(np.linalg.norm(whole - parts))
        comp_rows.append([" ".join(cmd), "held out" if cmd in HELD_OUT else "trained",
                          f"({parts[0]:.3f}, {parts[1]:.3f})", f"({whole[0]:.3f}, {whole[1]:.3f})", deviations[cmd]])
    dev_train = [deviations[c] for c in TRAIN_PAIRS]
    dev_all = list(deviations.values())
    held_success = [faith_errors[c] <= args.tau for c in HELD_OUT]
    g4 = {"per_command": {" ".join(c): v for c, v in deviations.items()},
          "delta_max_over_training_pairs": float(max(dev_train)), "delta_mean_over_training_pairs": float(np.mean(dev_train)),
          "delta_max_over_P": float(max(dev_all)), "delta_quantile_over_P": quantile_bound(dev_all, tol.alpha),
          "item_red_north": deviations[("RED", "NORTH")],
          "beta": float(np.mean(held_success)), "held_out_errors": {" ".join(c): faith_errors[c] for c in HELD_OUT},
          "verdict": (tolerance_verdict(max(dev_train), tol.delta_comp, "delta_comp (max over training pairs)")
                      if args.delta_scope == "trained" else
                      tolerance_verdict(max(dev_all), tol.delta_comp, "delta_comp (max over P)"))}
    report["G4"] = g4
    write_table(ctx, "composition", ["Command", "Split", "Sum of parts", "Composed", "Deviation"], comp_rows)
    ctx.log.info("G4: delta max %.3f, mean %.3f over training pairs; RED NORTH %.3f; beta %.2f at tau %.2f; %s",
                 g4["delta_max_over_training_pairs"], g4["delta_mean_over_training_pairs"], g4["item_red_north"],
                 g4["beta"], args.tau, g4["verdict"])

    profile_rows = [
        ["G0 authenticity", "strong", "strong", report["G0"]["basis"]],
        ["G1 preservation", g1["item_red"], g1["max"], g1["verdict"]],
        ["G2a faithfulness", g2a["item_red_north"], g2a["max_over_P"], g2a["verdict"]],
        ["G2b etiological", g2b["mechanisms"]["modifier-step"]["item_effect_red_north"],
         g2b["mechanisms"]["modifier-step"]["ACE_over_P"], g2b["mechanisms"]["modifier-step"]["dispositional"]],
        ["G3 robustness", modulus["hidden_mean_draw"][eps_grid.index(0.5)] if 0.5 in eps_grid else "n/a",
         f"{modulus['hidden_exact'][eps_grid.index(0.5)]:.3f} exact at 0.5" if 0.5 in eps_grid else "n/a",
         g3["verdict_hidden"]],
        ["G4 compositionality", g4["item_red_north"],
         g4["delta_max_over_training_pairs"] if args.delta_scope == "trained" else g4["delta_max_over_P"], g4["verdict"]],
        ["G4 systematicity", "n/a", g4["beta"], f"beta at tau {args.tau:g}"],
    ]
    write_table(ctx, "profile", ["Coordinate", "Item level", "Set level", "Verdict"], profile_rows)
    return report


def plot_modulus(ctx, report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g3 = report["G3"]
    eps = g3["eps_grid"]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    series = [("hidden state, exact supremum", g3["modulus"]["hidden_exact"]),
              ("hidden state, sampled supremum", g3["modulus"]["hidden_sampled"]),
              ("embedding, linearized supremum", g3["modulus"]["embedding_linearized"] or g3["modulus"]["embedding_sampled"]),
              ("embedding, sampled supremum", g3["modulus"]["embedding_sampled"]),
              ("hidden state, mean over random directions", g3["modulus"]["hidden_mean_draw"])]
    for i, (label, values) in enumerate(series):
        ax.plot(eps, values, label=label, **line_style(i))
    if report["tolerances"]["lipschitz"] is not None:
        L = report["tolerances"]["lipschitz"]
        ax.plot(eps, [L * e for e in eps], label=f"declared bound, L = {L:g}", color="0.7", linestyle="-", linewidth=2)
    ax.set_xlabel(r"perturbation scale $\varepsilon$")
    ax.set_ylabel(r"$\omega_U^{k,t}(\varepsilon)$")
    ax.legend(fontsize=7, frameon=False)
    save_figure(ctx, fig, "modulus")
    plt.close(fig)


def plot_positions(ctx, report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    for name, xy in LANDMARKS.items():
        ax.plot(xy[0], xy[1], marker="x", color="0.0", linestyle="none", markersize=8)
        ax.annotate(name, xy, textcoords="offset points", xytext=(4, 4), fontsize=7)
    realized = report.get("positions", {})
    for i, (cmd, (ix, iy, rx, ry)) in enumerate(realized.items()):
        ax.plot([ix, rx], [iy, ry], color="0.6", linestyle=":", linewidth=0.8)
        ax.plot(ix, iy, marker="o", markerfacecolor="white", markeredgecolor="0.0", linestyle="none", markersize=5)
        ax.plot(rx, ry, marker="s", color="0.3", linestyle="none", markersize=4)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("intended (circles) and realized (squares) positions", fontsize=8)
    save_figure(ctx, fig, "positions")
    plt.close(fig)


def main(ctx, args) -> None:
    import torch
    device = torch.device(args.device)
    agent, training_info = train(ctx, args, device)
    torch.save(agent.state_dict(), ctx.path("model", "agent", "pt"))
    system = TorchSystem(agent.eval(), device)
    report = audit(ctx, args, system, training_info)
    report["positions"] = {}
    for cmd in COMMANDS:
        realized = system.output(list(cmd))
        target = intended(cmd)
        report["positions"][" ".join(cmd)] = [float(target[0]), float(target[1]), float(realized[0]), float(realized[1])]
    ctx.write_json("report", report, prefix="data")
    plot_modulus(ctx, report)
    plot_positions(ctx, report)
    ctx.log.info("profile written to %s", ctx.path("tab", "profile", "tex").name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="grid-world grounding audit")
    parser.add_argument("--train", choices=["supervised", "reinforce"], default="supervised")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=1, help="tasks per gradient step")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--emb", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-dist", choices=["mixture", "pairs"], default="mixture")
    parser.add_argument("--reward", choices=["distance", "success", "shaped"], default="shaped")
    parser.add_argument("--reward-target", choices=["full", "landmark"], default="full",
                        help="target the reward is computed against: the intended position, or the landmark alone "
                             "(historical control: the direction token then carries no reward)")
    parser.add_argument("--success-bonus", type=float, default=2.0)
    parser.add_argument("--init-std", type=float, default=0.5, help="initial policy standard deviation")
    parser.add_argument("--entropy", type=float, default=0.0)
    parser.add_argument("--success-radius", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.5, help="held-out item tolerance for beta")
    parser.add_argument("--eps-pres", type=float, default=None)
    parser.add_argument("--eps-faith", type=float, default=None)
    parser.add_argument("--delta-comp", type=float, default=None)
    parser.add_argument("--eta", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--lipschitz", type=float, default=None, help="declared bound omega_bar(eps) = L * eps")
    parser.add_argument("--delta-scope", choices=["trained", "all"], default="all",
                        help="G4 verdict over the six trained pairs or over all eight commands of P")
    parser.add_argument("--eps-grid", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--noise-samples", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")
    add_run_options(parser)
    return parser


if __name__ == "__main__":
    run_script(Path(__file__), build_parser(), main)
