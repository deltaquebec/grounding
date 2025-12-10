# On measuring grounding and generalizing grounding problems

This repository accompanies the paper:

> **On measuring grounding and generalizing grounding problems**
> Daniel Quigley and Eric Maynard
> [arXiv link](https://www.arxiv.org/abs/2512.06205)

## Overview

The symbol grounding problem asks how tokens like *cat* can be *about* cats, as opposed to mere shapes manipulated in a calculus. We recast grounding from a binary judgment into an audit across measurable desiderata:

- **G0 (Authenticity):** Mechanisms reside inside the agent and were acquired through learning or evolution
- **G1 (Preservation):** Atomic meanings remain intact through processing
- **G2a (Correlational Faithfulness):** Realized meanings match intended ones
- **G2b (Etiological Faithfulness):** Internal mechanisms causally contribute to success
- **G3 (Robustness):** Graceful degradation under declared perturbations
- **G4 (Compositionality):** The whole is built systematically from the parts

The framework applies to symbolic, referential, vectorial, and relational grounding modes, and yields grounding *profiles* rather than binary verdicts.

## Repository contents
```
.
├── paper/
│   └── grounding.pdf          # main paper
├── code/
│   └── gridworld.py           # toy example implementation
└── README.md
```

## Toy example: gridworld

We provide a minimal implementation of the agent described in the paper. The agent learns to navigate a 2D grid world based on color-direction commands (e.g., `RED NORTH`), trained via the classic REINFORCE.

### Requirements
```bash
pip install torch numpy matplotlib
```

### Running the Audit
```bash
python code/gridworld.py
```

This will do three things:
1. train the agent (~3000 episodes);
2. run the full grounding audit (G0–G4);
3. report results to .txt file as well as to screen

### Sample Output
```
GROUNDING PROFILE SUMMARY
============================================================
  G0  (authenticity):     Strong (learned via REINFORCE)
  G1  (preservation):     ε_pres  = 0.0300
  G2a (faithfulness):     ε_faith = 0.0409
  G2b (etiological):      ACE(M)  = 1.0
  G3  (robustness):       ω_U(0.5) = 0.0881
  G4  (compositionality): δ_comp  = 0.0309
============================================================
```

## Citation
```bibtex
@misc{quigley2025measuringgroundinggeneralizinggrounding,
      title={On measuring grounding and generalizing grounding problems}, 
      author={Daniel Quigley and Eric Maynard},
      year={2025},
      eprint={2512.06205},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.06205}, 
}
```
