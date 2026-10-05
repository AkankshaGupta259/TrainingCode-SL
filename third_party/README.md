# third_party — official model repositories

Native in this repo: **ST-GCN** (`src/slr/models/stgcn.py`) and the **video models**
(I3D, SlowFast via PyTorchVideo). The other skeleton models are loaded from their
**official** implementations through a uniform adapter (`src/slr/models/external_gcn.py`,
`motionbert_adapter.py`) so the benchmark uses the authors' exact architectures.

## Clone the repos

```bash
cd third_party
git clone https://github.com/Uason-Chen/CTR-GCN.git      CTR-GCN
git clone https://github.com/Jho-Yonsei/HD-GCN.git       HD-GCN
git clone https://github.com/stnoah1/infogcn.git         infogcn
git clone https://github.com/kenziyuliu/MS-G3D.git       MS-G3D
git clone https://github.com/Walter0807/MotionBERT.git   MotionBERT
# HA-GCN: place its implementation at third_party/HA-GCN
```

Each model's config (`configs/models/<name>.yaml`) sets `third_party`, the dotted
`import` path to the model class, and its constructor `args`. The adapter adds the repo
to `sys.path`, imports the class, injects `num_class`, and builds it. Most repos already
take `(N, C, T, V, M)` input and return logits — matching our ST-GCN — so no glue is needed.

## The shared 27-node graph

Our skeleton has **27 nodes** (see `src/slr/data/graph.py`). The GCN repos ship graphs for
NTU (25 body joints), so add a 27-node graph class to each repo that these configs can import,
e.g. `third_party/CTR-GCN/graph/sign27.py`:

```python
import numpy as np
from slr.data.graph import Graph as _SignGraph   # or copy INWARD/NUM_NODES in

class Graph:
    def __init__(self, labeling_mode='spatial'):
        g = _SignGraph(strategy=labeling_mode)
        self.A = g.A                 # (K, 27, 27)
        self.num_node = 27
```

MS-G3D expects an `AdjMatrixGraph` exposing a binary adjacency `A_binary`; adapt accordingly.
Keep the node ordering identical to `graph.NODE_SPEC` so the cached poses line up.

## Fairness

Use the **same cached poses** (`data/wlasl/poses/`), the **same 27-node graph**, and
**joint-stream only** for the headline table across every skeleton model. Report any
multi-stream (bone/motion) fusion separately.
