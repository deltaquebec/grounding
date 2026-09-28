"""pilot grounding audit of pretrained distributional model"""

from __future__ import annotations

import argparse
import random
import string
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_common import (EvaluationTuple, Tolerances, add_run_options, bootstrap_difference, line_style,  # noqa: E402
                          quantile_bound, robustness_verdict, run_script, save_figure, tolerance_verdict, write_table)

ALPHABET = string.ascii_lowercase


class VectorSystem:
    """unit-normalized lookup table over vocabulary"""

    def __init__(self, matrix: np.ndarray, words: list[str]):
        self.matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        self.words = words
        self.index = {w: i for i, w in enumerate(words)}

    def lookup(self, word: str):
        """Phi token to unit vector; None when out of vocabulary under policy"""
        for form in (word, word.lower(), word.capitalize()):
            i = self.index.get(form)
            if i is not None:
                return self.matrix[i]
        return None

    def ablated(self, kind: str, rng: random.Random, dim: int, words: list[str]):
        """system copy with mechanism turned off: whole-map permutation, or principal subspace projected out"""
        if kind == "whole-map":
            perm = words[:]
            rng.shuffle(perm)
            mapping = dict(zip(words, perm))
            parent = self

            class Permuted(VectorSystem):
                def __init__(self):
                    self.matrix, self.words, self.index = parent.matrix, parent.words, parent.index

                def lookup(self, word):
                    return parent.lookup(mapping.get(word, word))

            return Permuted()
        if kind == "subspace":
            rows = [self.lookup(w) for w in words]
            stack = np.vstack([r for r in rows if r is not None])
            centered = stack - stack.mean(axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            basis = vt[:dim]
            projected = self.matrix - self.matrix @ basis.T @ basis
            norms = np.linalg.norm(projected, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return VectorSystem(projected / norms, self.words)
        raise ValueError(kind)


class Word2VecSystem(VectorSystem):
    """nltk pruned sample or word2vec text file; fragment from WordNet"""

    def __init__(self, path: Path | None):
        if path is None:
            from nltk.data import find
            path = Path(str(find("models/word2vec_sample/pruned.word2vec.txt")))
        words, vecs = [], []
        with open(path, encoding="utf-8", errors="replace") as handle:
            first = handle.readline().split()
            if len(first) != 2:
                handle.seek(0)
            for line in handle:
                parts = line.rstrip().split(" ")
                words.append(parts[0])
                vecs.append(np.asarray(parts[1:], dtype=np.float32))
        super().__init__(np.vstack(vecs), words)
        self.path = path

    @staticmethod
    def _lemmas(word: str, pos: str):
        from nltk.corpus import wordnet as wn
        syns, ants = set(), set()
        for ss in wn.synsets(word, pos=pos):
            for lm in ss.lemmas():
                name = lm.name().lower()
                if "_" not in name and name != word:
                    syns.add(name)
                for an in lm.antonyms():
                    aname = an.name().lower()
                    if "_" not in aname:
                        ants.add(aname)
        return syns, ants

    def fragment(self, rng: random.Random, nouns_wanted: list[str]):
        """adjectives with in-vocabulary synonym and antonym; nouns with WordNet synonyms"""
        from nltk.corpus import wordnet as wn
        adjs, seen = [], set()
        for w in tqdm(sorted(self.index), desc="fragment", unit="word", leave=False):
            lw = w.lower()
            if lw in seen or not lw.isalpha() or len(lw) < 3:
                continue
            seen.add(lw)
            if self.lookup(lw) is None or not wn.synsets(lw, pos="a"):
                continue
            syns, ants = self._lemmas(lw, "a")
            syns = sorted(s for s in syns if self.lookup(s) is not None)
            ants = sorted(a for a in ants if self.lookup(a) is not None)
            if syns and ants:
                adjs.append((lw, syns[0], ants[0]))
        nouns = [n for n in nouns_wanted if self.lookup(n) is not None]
        noun_syn = {}
        for n in nouns:
            syns, _ = self._lemmas(n, "n")
            syns = sorted(s for s in syns if self.lookup(s) is not None)
            if syns:
                noun_syn[n] = syns[0]
        rng.shuffle(adjs)
        return adjs, nouns, noun_syn


class SyntheticSystem(VectorSystem):
    """random unit vectors over pseudo-words with synthetic gold"""

    def __init__(self, rng: np.random.Generator, n_words: int = 1500, dim: int = 50):
        self.rng = rng
        words = list({self._word() for _ in range(n_words * 2)})[:n_words]
        matrix = rng.standard_normal((len(words), dim))
        super().__init__(matrix, words)

    def _word(self) -> str:
        return "".join(self.rng.choice(list(ALPHABET), size=int(self.rng.integers(4, 9))))

    def add_word(self, word: str, vector: np.ndarray) -> None:
        self.matrix = np.vstack([self.matrix, vector / np.linalg.norm(vector)])
        self.words.append(word)
        self.index[word] = len(self.words) - 1

    def pairs(self, count: int, noise: float):
        """synthetic rated pairs: gold is rescaled cosine plus noise"""
        out = []
        for _ in range(count):
            a, b = self.rng.choice(self.words, size=2, replace=False)
            c = float(self.lookup(a) @ self.lookup(b))
            out.append((a, b, 5.0 + 5.0 * c + noise * self.rng.standard_normal()))
        return out

    def fragment(self, rng: random.Random, nouns_wanted: list[str]):
        adjs = []
        for a in list(self.rng.choice(self.words, size=60, replace=False)):
            base = self.lookup(a)
            syn, ant = a + "syn", a + "ant"
            self.add_word(syn, base + 0.3 * self.rng.standard_normal(base.shape))
            self.add_word(ant, -base + 0.3 * self.rng.standard_normal(base.shape))
            adjs.append((a, syn, ant))
        nouns = list(self.rng.choice([w for w in self.words if not w.endswith(("syn", "ant"))], size=20, replace=False))
        noun_syn = {}
        for n in nouns:
            self.add_word(n + "syn", self.lookup(n) + 0.3 * self.rng.standard_normal(self.matrix.shape[1]))
            noun_syn[n] = n + "syn"
        rng.shuffle(adjs)
        return adjs, nouns, noun_syn


NOUNS = ["house", "road", "answer", "story", "voice", "car", "room", "coat", "winter", "job", "meal", "river",
         "child", "city", "song", "wall", "field", "market", "letter", "engine"]


def read_pairs(path: Path):
    """rated pairs from MEN natural form (word word score) or SimLex-999 (tab separated with header)"""
    out = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if not parts or parts[0].lower() == "word1":
                continue
            if len(parts) >= 4 and "\t" in line:
                a, b, s = parts[0], parts[1], parts[3]
            else:
                a, b, s = parts[0], parts[1], parts[2]
            out.append((a.lower(), b.lower(), float(s)))
    return out


def cos(u, v) -> float:
    return float(np.dot(u, v))


def ang(u, v) -> float:
    """angular distance normalized to [0, 1]; metric on sphere"""
    return float(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0)) / np.pi)


def quantile_fn(values):
    """empirical distribution function from calibration sample"""
    v = np.sort(np.asarray(values))

    def F(x):
        return float(np.searchsorted(v, x, side="right")) / len(v)
    return F


class Composer:
    """Gamma for composites: additive pooling, or weighted pooling with fitted adjective weight"""

    def __init__(self, system: VectorSystem, weight: float = 0.5):
        self.system = system
        self.weight = weight

    def __call__(self, tokens):
        vs = [(self.system.lookup(t), i) for i, t in enumerate(tokens)]
        vs = [(v, i) for v, i in vs if v is not None]
        if not vs:
            return None
        if len(vs) == 1:
            return vs[0][0]
        v = sum((self.weight if i == 0 else 1.0 - self.weight) * vec for vec, i in vs)
        n = np.linalg.norm(v)
        return v / n if n > 0 else None


def audit_atoms(system, pairs, label, rng, tol, log):
    """G1 on one gold standard: quantile metric on evaluation half, transforms from calibration half"""
    scored = []
    for a, b, s in pairs:
        va, vb = system.lookup(a), system.lookup(b)
        if va is not None and vb is not None:
            scored.append((a, b, cos(va, vb), s))
    rng.shuffle(scored)
    half = len(scored) // 2
    cal, ev = scored[:half], scored[half:]
    Fm = quantile_fn([c for _, _, c, _ in cal])
    Fg = quantile_fn([s for *_, s in cal])
    per_atom = defaultdict(list)
    for a, b, c, s in ev:
        e = abs(Fm(c) - Fg(s))
        per_atom[a].append(e)
        per_atom[b].append(e)
    atom_err = {w: float(np.mean(es)) for w, es in per_atom.items()}
    ae = np.array(list(atom_err.values()))
    rho = stats.spearmanr([c for _, _, c, _ in ev], [s for *_, s in ev]).statistic
    result = {"gold": label, "n_pairs_eval": len(ev), "n_atoms_eval": len(atom_err),
              "coverage": len(scored) / len(pairs), "median": float(np.median(ae)),
              "p90": float(np.percentile(ae, 90)), "quantile": quantile_bound(ae, tol.alpha),
              "pass_rate": float(np.mean(ae <= tol.eps_pres)) if tol.eps_pres is not None else None,
              "spearman": float(rho), "verdict": tolerance_verdict(quantile_bound(ae, tol.alpha), tol.eps_pres,
                                                                    "eps_pres (quantile over atoms)")}
    log.info("G1 [%s]: median %.3f, p90 %.3f, pass rate %.3f at %s, spearman %.3f over %d atoms",
             label, result["median"], result["p90"], result["pass_rate"] or 0.0, tol.eps_pres, rho, len(atom_err))
    return result, scored


def kendall_disagreement(compose, a, syn, ant, n, ra, rn):
    """fraction of three variant pairs ordered against syn > ant > unrelated; None on missing vectors"""
    p, ps, pa, pr = compose([a, n]), compose([syn, n]), compose([ant, n]), compose([ra, rn])
    if any(v is None for v in (p, ps, pa, pr)):
        return None
    sims = [cos(p, ps), cos(p, pa), cos(p, pr)]
    return sum(1 for i, j in combinations(range(3), 2) if sims[i] <= sims[j]) / 3.0


def audit_composites(compose, adjs, nouns, rng):
    """per-item Kendall disagreement for every adjective-noun base phrase"""
    vocab_adj = [a for a, _, _ in adjs]
    results = []
    for a, syn, ant in tqdm(adjs, desc="composites", unit="adj", leave=False):
        for n in nouns:
            ra = rng.choice([x for x in vocab_adj if x not in (a, syn, ant)])
            rn = rng.choice([x for x in nouns if x != n])
            d = kendall_disagreement(compose, a, syn, ant, n, ra, rn)
            if d is not None:
                results.append((a, n, d))
    return results


def summarize(results, adj_set, tol):
    sel = np.array([d for a, _, d in results if a in adj_set])
    return {"n_items": int(len(sel)), "kendall_mean": float(np.mean(sel)), "kendall_p90": float(np.percentile(sel, 90)),
            "pass_rate": float(np.mean(sel <= tol.tau))}


def fit_weight(system, adjs, nouns, rng, tol, log):
    """weighted pooling: adjective weight chosen on calibration adjectives by pass rate"""
    best = (None, -1.0)
    for weight in np.linspace(0.1, 0.9, 9):
        composer = Composer(system, float(weight))
        local = random.Random(rng.random())
        res = audit_composites(composer, adjs, nouns, local)
        rate = summarize(res, {a for a, _, _ in adjs}, tol)["pass_rate"]
        if rate > best[1]:
            best = (float(weight), rate)
    log.info("composer weight fitted on calibration adjectives: %.2f (pass rate %.3f)", *best)
    return best[0]


def typo(word: str, k: int, rng: random.Random) -> str:
    w = list(word)
    for _ in range(k):
        op = rng.choice("ids" + ("w" if len(w) > 1 else ""))
        i = rng.randrange(len(w))
        if op == "i":
            w.insert(i, rng.choice(ALPHABET))
        elif op == "d" and len(w) > 1:
            del w[i]
        elif op == "s":
            w[i] = rng.choice(ALPHABET)
        elif op == "w" and i < len(w) - 1:
            w[i], w[i + 1] = w[i + 1], w[i]
    return "".join(w)


def deviation(compose, tokens, tokens_pert, oov_policy: str = "maximal"):
    """angular deviation; phrase whose tokens all leave vocabulary scores 1.0 (maximal) or is excluded"""
    v0, v1 = compose(tokens), compose(tokens_pert)
    if v0 is None:
        return None
    if v1 is None:
        return 1.0 if oov_policy == "maximal" else None
    return ang(v0, v1)


def audit_robustness(system, compose, adjs, nouns, noun_syn, args, tol, rng, log):
    """empirical modulus under typo and synonym threats; median and 90th percentile per scale"""
    phrases = [[a, n] for a, _, _ in adjs for n in nouns]
    rng.shuffle(phrases)
    phrases = phrases[:args.n_g3_phrases]
    out = {"protocol": args.typo_protocol, "oov_policy": args.oov_policy, "typo": {}, "synonym": {}, "antonym_diagnostic": {}}
    for k in range(1, args.typo_max + 1):
        devs, oov_tokens, all_oov = [], 0, 0
        for toks in phrases:
            pert = list(toks)
            if args.typo_protocol == "per-token":
                i = rng.randrange(len(toks))
                pert[i] = typo(pert[i], k, rng)
            else:
                for _ in range(k):
                    i = rng.randrange(len(toks))
                    pert[i] = typo(pert[i], 1, rng)
            missing = sum(1 for t in pert if system.lookup(t) is None)
            oov_tokens += missing
            all_oov += int(missing == len(pert))
            d = deviation(compose, toks, pert, args.oov_policy)
            if d is not None:
                devs.append(d)
        out["typo"][k] = {"median": float(np.median(devs)), "p90": float(np.percentile(devs, 90)),
                          "quantile": quantile_bound(devs, tol.alpha), "oov_token_rate": oov_tokens / (2 * len(phrases)),
                          "all_oov_rate": all_oov / len(phrases), "n": len(devs)}
        log.info("G3 typo eps=%d: median %.3f, p90 %.3f, oov token rate %.3f, all-oov phrase rate %.3f (%s)",
                 k, out["typo"][k]["median"], out["typo"][k]["p90"], out["typo"][k]["oov_token_rate"],
                 out["typo"][k]["all_oov_rate"], "deviation 1.0 by policy" if args.oov_policy == "maximal" else "excluded by policy")
    for k in (1, 2):
        devs = []
        for a, syn, _ in adjs:
            for n in nouns[:args.nouns_for_synonyms]:
                pert = [syn, n]
                if k == 2:
                    if n not in noun_syn:
                        continue
                    pert = [syn, noun_syn[n]]
                d = deviation(compose, [a, n], pert)
                if d is not None:
                    devs.append(d)
        out["synonym"][k] = {"median": float(np.median(devs)), "p90": float(np.percentile(devs, 90)),
                             "quantile": quantile_bound(devs, tol.alpha), "n": len(devs)}
        log.info("G3 synonym eps=%d: median %.3f, p90 %.3f over %d phrases", k, out["synonym"][k]["median"],
                 out["synonym"][k]["p90"], len(devs))
    devs = []
    for a, _, ant in adjs:
        for n in nouns[:args.nouns_for_synonyms]:
            d = deviation(compose, [a, n], [ant, n])
            if d is not None:
                devs.append(d)
    out["antonym_diagnostic"] = {"median": float(np.median(devs)), "p90": float(np.percentile(devs, 90)), "n": len(devs)}
    log.info("G3 antonym diagnostic: median %.3f, p90 %.3f", out["antonym_diagnostic"]["median"],
             out["antonym_diagnostic"]["p90"])
    scales = list(range(1, args.typo_max + 1))
    out["verdict_typo"] = robustness_verdict(scales, [out["typo"][k]["quantile"] for k in scales], tol.lipschitz)
    out["verdict_synonym"] = robustness_verdict([1, 2], [out["synonym"][k]["quantile"] for k in (1, 2)], tol.lipschitz)
    return out

def audit_intervention(system, scored, args, tol, rng, nprng, log):
    """causal contribution to decodability, ACE_D, with a paired cluster-bootstrap interval"""
    words = sorted({w for a, b, *_ in scored for w in (a, b)})
    ratings = np.array([s for *_, s in scored], dtype=float)
    valid = (ratings[:, None] - ratings[None, :]) != 0  # ordered pair-pairs with distinct ratings
    sign = (ratings[:, None] - ratings[None, :]) > 0
    validf = valid.astype(np.float32)

    def hits(sys_):
        cosines = np.array([cos(sys_.lookup(a), sys_.lookup(b)) for a, b, _, _ in scored])
        return (((cosines[:, None] - cosines[None, :]) > 0) == sign).astype(np.float32)

    on = hits(system)
    n = len(scored)
    out = {"succ_on": float(on[valid].mean()), "population": "all ordered pair-pairs with distinct ratings",
           "n_pairs": n, "n_pairpairs": int(valid.sum()), "paired_arms": True, "mechanisms": {}}
    brng = np.random.default_rng(args.seed + 1)
    weights = [np.bincount(brng.integers(0, n, n), minlength=n).astype(np.float32) for _ in range(args.boot)]
    denominators = [float(w @ validf @ w) for w in weights]
    for kind in ("whole-map", "subspace"):
        off = hits(system.ablated(kind, rng, args.subspace_dim, words))
        ace = float(on[valid].mean() - off[valid].mean())
        delta = (on - off) * validf
        draws = np.array([float(w @ delta @ w) / d for w, d in zip(weights, denominators)])
        low, high = (float(x) for x in np.percentile(draws, [2.5, 97.5]))
        out["mechanisms"][kind] = {
            "succ_off": float(off[valid].mean()), "ACE_D": ace, "ci95_paired_cluster": [low, high],
            "point_at_eta": None if tol.eta is None else bool(ace >= tol.eta),
            "ci_low_at_eta": None if tol.eta is None else bool(low >= tol.eta),
            "note": ("total ablation: effect equals concordance less chance" if kind == "whole-map"
                     else f"top {args.subspace_dim} principal components of the MEN vocabulary projected out"),
        }
        log.info("ACE_D [%s]: success %.3f -> %.3f, ACE_D %.3f, paired cluster ci95 [%.3f, %.3f]",
                 kind, out["succ_on"], off[valid].mean(), ace, low, high)
    return out


def cos(u, v) -> float:
    """cosine on unit vectors"""
    return float(np.dot(u, v))

def plot_modulus(ctx, g3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    typo_k = sorted(g3["typo"])
    syn_k = sorted(g3["synonym"])
    series = [("typo, median", [g3["typo"][k]["median"] for k in typo_k], typo_k),
              ("typo, 90th percentile", [g3["typo"][k]["p90"] for k in typo_k], typo_k),
              ("synonym, median", [g3["synonym"][k]["median"] for k in syn_k], syn_k),
              ("synonym, 90th percentile", [g3["synonym"][k]["p90"] for k in syn_k], syn_k)]
    for i, (label, values, xs) in enumerate(series):
        ax.plot([0, *xs], [0.0, *values], label=label, **line_style(i))
    ax.set_xlabel(r"perturbation scale $\varepsilon$ (edits or substitutions)")
    ax.set_ylabel("angular deviation")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False)
    save_figure(ctx, fig, "modulus")
    plt.close(fig)


def main(ctx, args) -> None:
    rng = random.Random(args.seed)
    nprng = np.random.default_rng(args.seed)
    tol = Tolerances(eps_pres=args.eps_pres, eps_faith=None, delta_comp=0.0, eta=args.eta, tau=args.kendall_tol,
                     alpha=args.alpha, lipschitz=args.lipschitz)
    evaluation = EvaluationTuple(context="ling", meaning_type="inf",
                                 threats=[f"typo, {args.typo_protocol}, up to {args.typo_max} edits", "synonym substitution, 1 to 2"],
                                 reference="MEN and SimLex-999 pairs for atoms; adjective-noun fragment with WordNet tiers for composites",
                                 spurious_covariates=[])
    report = {"evaluation_tuple": evaluation.to_dict(), "tolerances": tol.to_dict(), "smoke": args.smoke}
    if args.smoke:
        system = SyntheticSystem(nprng)
        men = system.pairs(1200, noise=1.0)
        simlex = system.pairs(600, noise=2.5)
        report["system"] = "synthetic random vectors (smoke test)"
    else:
        system = Word2VecSystem(args.vectors)
        men = read_pairs(args.data_dir / args.men)
        simlex = read_pairs(args.data_dir / args.simlex)
        report["system"] = f"word2vec {system.path.name}, {len(system.words)} words, {system.matrix.shape[1]} dimensions"
    ctx.log.info("system: %s", report["system"])
    ctx.log.info("no spurious covariate declared: P^do = P")

    with logging_redirect_tqdm(loggers=[ctx.log]):
        report["G1"] = {}
        report["G1"]["men"], scored_men = audit_atoms(system, men, "MEN relatedness", rng, tol, ctx.log)
        report["G1"]["simlex"], _ = audit_atoms(system, simlex, "SimLex-999 similarity", rng, tol, ctx.log)

        adjs, nouns, noun_syn = system.fragment(rng, NOUNS)
        half = len(adjs) // 2
        cal_adjs, ev_adjs = adjs[:half], adjs[half:]
        report["fragment"] = {"adjectives": len(adjs), "nouns": len(nouns), "nouns_with_synonym": len(noun_syn)}
        weight = 0.5
        if args.composer == "weighted":
            weight = fit_weight(system, cal_adjs, nouns, rng, tol, ctx.log)
        compose = Composer(system, weight)
        report["composer"] = {"kind": args.composer, "adjective_weight": weight,
                              "G0": "instrument: composition declared by the auditor" + (" and fitted on the calibration adjectives" if args.composer == "weighted" else "")}
        results = audit_composites(compose, adjs, nouns, rng)
        report["G2a"] = {"all": summarize(results, {a for a, _, _ in adjs}, tol),
                         "calibration_half": summarize(results, {a for a, _, _ in cal_adjs}, tol)}
        report["G4"] = {"held_out_half": summarize(results, {a for a, _, _ in ev_adjs}, tol),
                        "delta_comp": 0.0 if args.composer == "additive" else None,
                        "note": ("delta_comp is zero by construction under the declared additive composition"
                                 if args.composer == "additive" else "delta_comp against additive algebra is the fitted weight's departure from 0.5")}
        report["G2a"]["note"] = ("the two halves are disjoint adjective sets under one stipulated rule; their agreement is a stability check"
                                 if args.composer == "additive" else "the weight was fitted on the calibration half; the held-out half tests its transfer")
        ctx.log.info("G2a composites: kendall mean %.3f, pass rate %.3f (all); calibration half %.3f, %.3f",
                     report["G2a"]["all"]["kendall_mean"], report["G2a"]["all"]["pass_rate"],
                     report["G2a"]["calibration_half"]["kendall_mean"], report["G2a"]["calibration_half"]["pass_rate"])
        ctx.log.info("G4 held-out half: kendall mean %.3f, beta %.3f at %.2f", report["G4"]["held_out_half"]["kendall_mean"],
                     report["G4"]["held_out_half"]["pass_rate"], tol.tau)

        report["G3"] = audit_robustness(system, compose, ev_adjs, nouns, noun_syn, args, tol, rng, ctx.log)
        report["G2b"] = audit_intervention(system, scored_men, args, tol, rng, nprng, ctx.log)
        report["G2b"]["historical"] = ("retention under skip-gram training declared, unmeasured; predicate bridge "
                                       "(co-occurrence prediction to rating concordance) a recorded assumption")
        report["G0"] = {"atoms": "strong: Phi acquired under skip-gram T", "composites": "instrument: composition declared by the auditor"}

    ctx.write_json("report", report, prefix="data")
    rows = [[k, v["median"], v["p90"], v["pass_rate"], v["spearman"]] for k, v in report["G1"].items()]
    write_table(ctx, "atoms", ["Gold", "Median", "P90", "Pass rate", "Spearman"], rows)
    rows = [["all", *[report["G2a"]["all"][k] for k in ("n_items", "kendall_mean", "kendall_p90", "pass_rate")]],
            ["calibration half", *[report["G2a"]["calibration_half"][k] for k in ("n_items", "kendall_mean", "kendall_p90", "pass_rate")]],
            ["held-out half", *[report["G4"]["held_out_half"][k] for k in ("n_items", "kendall_mean", "kendall_p90", "pass_rate")]]]
    write_table(ctx, "composites", ["Split", "Items", "Kendall mean", "Kendall p90", "Pass rate"], rows)
    g3 = report["G3"]
    rows = [[f"typo {k}", g3["typo"][k]["median"], g3["typo"][k]["p90"], g3["typo"][k]["oov_token_rate"],
             g3["typo"][k]["all_oov_rate"]] for k in sorted(g3["typo"])]
    rows += [[f"synonym {k}", g3["synonym"][k]["median"], g3["synonym"][k]["p90"], "", ""] for k in sorted(g3["synonym"])]
    rows += [["antonym diagnostic", g3["antonym_diagnostic"]["median"], g3["antonym_diagnostic"]["p90"], "", ""]]
    write_table(ctx, "modulus", ["Threat", "Median", "P90", "OOV token rate", "All-oov phrase rate"], rows)
    rows = [[k, report["G2b"]["succ_on"], v["succ_off"], v["ACE"], f"[{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"]
            for k, v in report["G2b"]["mechanisms"].items()]
    write_table(ctx, "mechanisms", ["Mechanism", "Success on", "Success off", "ACE", "CI95"], rows)
    profile = [["G0", "atoms strong; composites instrument"],
               ["G1 MEN", f"median {report['G1']['men']['median']:.3f}; pass {report['G1']['men']['pass_rate']:.3f}"],
               ["G1 SimLex", f"median {report['G1']['simlex']['median']:.3f}; pass {report['G1']['simlex']['pass_rate']:.3f}"],
               ["G2a composites", f"kendall {report['G2a']['all']['kendall_mean']:.3f}; pass {report['G2a']['all']['pass_rate']:.3f}"],
               ["G2b whole-map", f"ACE {report['G2b']['mechanisms']['whole-map']['ACE']:.3f}"],
               ["G2b subspace", f"ACE {report['G2b']['mechanisms']['subspace']['ACE']:.3f}"],
               ["G3 typo", "; ".join(f"{g3['typo'][k]['median']:.3f}" for k in sorted(g3["typo"])) + " (median)"],
               ["G3 synonym", "; ".join(f"{g3['synonym'][k]['median']:.3f}" for k in sorted(g3["synonym"])) + " (median)"],
               ["G4 beta", f"{report['G4']['held_out_half']['pass_rate']:.3f}"]]
    write_table(ctx, "profile", ["Coordinate", "Value"], profile)
    plot_modulus(ctx, g3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="pilot grounding audit of a distributional model")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--men", type=str, default="men.txt")
    parser.add_argument("--simlex", type=str, default="simlex.txt")
    parser.add_argument("--vectors", type=Path, default=None, help="word2vec text file; default nltk pruned sample")
    parser.add_argument("--smoke", action="store_true", help="synthetic system, no data or nltk needed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps-pres", type=float, default=0.20)
    parser.add_argument("--kendall-tol", type=float, default=0.25)
    parser.add_argument("--eta", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--lipschitz", type=float, default=None)
    parser.add_argument("--typo-protocol", choices=["per-phrase", "per-token"], default="per-phrase")
    parser.add_argument("--typo-max", type=int, default=3)
    parser.add_argument("--oov-policy", choices=["maximal", "exclude"], default="maximal",
                        help="phrase with every token out of vocabulary: deviation 1.0, or excluded with its rate reported")
    parser.add_argument("--n-g3-phrases", type=int, default=400)
    parser.add_argument("--nouns-for-synonyms", type=int, default=5)
    parser.add_argument("--composer", choices=["additive", "weighted"], default="additive")
    parser.add_argument("--pairpairs", type=int, default=20000)
    parser.add_argument("--boot", type=int, default=1000)
    parser.add_argument("--subspace-dim", type=int, default=10)
    add_run_options(parser)
    return parser


if __name__ == "__main__":
    run_script(Path(__file__), build_parser(), main, path_options=("--data-dir", "--vectors"))
