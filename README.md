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
│   ├── audit_common.py           # collection of modules
│   ├── gridworld_audit.py        # toy example implementation
│   ├── pilot_audit.py            # toy example implementation
│   └── test_gridworld_stub.py    # implementation artifact
├── data/
│   ├── men.txt            # MEN natural form full file
│   └── simlex.txt         # SimLex-999.txt
└── README.md
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
