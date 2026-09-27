# Fusion configuration

`nuscenes_fuser.json` contains the manuscript's **fusion-module** settings,
in the standalone core's `[B, C, Y, X]` convention. It is not a full detector
training configuration. Load it as follows:

```python
import json
from bpdaf import BPDAFFuser

with open("configs/nuscenes_fuser.json", encoding="utf-8") as handle:
    fuser = BPDAFFuser(**json.load(handle))
```

The physical extent is `x,y in [-54,54]` meters. The paper's 180 by 180 BEV
grid therefore has 0.6 m cell spacing; it is not the 0.075 m input voxel grid.
The module obtains H and W from the actual feature tensors, so the integration
must ensure their physical extent and axis order are correct.

Use the native configs under `integrations/mmdetection3d/configs/` for the
optional detector integration. Its adapter handles that host's `[B,C,X,Y]`
axis convention. The inherited host recipe is an integration example, not a
reconstruction of the paper's complete training run.

No guessed KITTI extent is supplied. The paper does not specify its exact
feature-grid geometry; a KITTI integration must take bounds and axis order
from the chosen feature producer. See [reproduction status](../docs/reproduction.md).
