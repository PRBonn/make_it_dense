import random
from typing import Tuple

import torch


def collate_models(batch: Tuple[torch.Tensor], batch_size: int = -1) -> Tuple[torch.Tensor]:
    """Each batch contains multiple (8, 8, 8) patches of sdf values, torch.stack doesn't work due
    the fact that each sample in the batch will have different number of leaf nodes.

    Since each leaf node is not related to any other, we simply
    concatenate all the leaf nodes in the batch into a big (N, 8, 8, 8)
    tensor, where N = sum(leaf_nodes).

    Example:
    =======

    Let's say we have a batch_size of 2. The first item in the dataset will
    have 200 leaf_nodes, thus batch[0].shape == (200, 8, 8, 8). The second
    sample will have 150 leaf_nodes and thus batch[1].shape == (150, 8, 8, 8).
    The output of this collate_fn will be a tensor of size == (350, 8, 8, 8).


    TODO:
    ====

    Update this explanation
    """
    try:
        elem = batch[0]
    except IndexError:
        print("[WARNING] Discarding empty batch")
        return None
    if elem is None:
        return None
    # For simplicity we recive both inputs and targets on the same dictionary, but later on we split
    # it to make it easier to handle.
    try:
        targets = {
            key: {key2: torch.cat([d[key][key2] for d in batch]) for key2 in elem[key].keys()}
            for key in elem.keys()
        }
    except:
        return None

    # Hack: select random indices, please fix this SHIT!
    n_volumes = targets["gt_tsdf_10"]["nodes"].shape[0]
    random_indices = random.sample(torch.arange(n_volumes).tolist(), batch_size)
    for key in targets.keys():
        targets[key]["nodes"] = targets[key]["nodes"][random_indices]
        targets[key]["origin"] = targets[key]["origin"][random_indices]
    # First element must be the inputs, if not, I'm so sorry :)
    inputs = targets.pop([*targets][0])
    return inputs, targets
