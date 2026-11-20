# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import Tensor

def compute_cfg_routing_masks(
    advantage: torch.Tensor,
    *,
    positive_only_conditional: bool,
    unconditional_prob: float,
    random_values: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute sample routing masks for CFG training.

    Args:
        advantage: Boolean tensor where True marks positive samples.
        positive_only_conditional: Route only positive samples to the
            conditional branch when True.
        unconditional_prob: Dropout probability for unconditional routing.
            When ``positive_only_conditional`` is True, applies only to
            positive samples; otherwise applies to all samples.
        random_values: Optional pre-sampled uniform noise in ``[0, 1)`` used to
            make routing deterministic in tests.

    Returns:
        Dictionary of boolean masks describing how the batch is routed.
    """
    advantage = advantage.to(dtype=torch.bool)
    batch_size = advantage.shape[0]
    device = advantage.device

    if random_values is None:
        random_values = torch.rand(batch_size, device=device)
    else:
        random_values = random_values.to(device=device)

    positive_mask = advantage
    negative_mask = ~positive_mask

    if positive_only_conditional:
        positive_conditional_mask = positive_mask & (random_values > unconditional_prob)
        negative_conditional_mask = torch.zeros_like(positive_mask)
    else:
        guidance_mask = random_values > unconditional_prob
        positive_conditional_mask = positive_mask & guidance_mask
        negative_conditional_mask = negative_mask & guidance_mask

    conditional_mask = positive_conditional_mask | negative_conditional_mask
    positive_unconditional_mask = positive_mask & ~positive_conditional_mask
    negative_unconditional_mask = negative_mask & ~negative_conditional_mask

    return {
        "positive_mask": positive_mask,
        "negative_mask": negative_mask,
        "conditional_mask": conditional_mask,
        "positive_conditional_mask": positive_conditional_mask,
        "positive_unconditional_mask": positive_unconditional_mask,
        "negative_conditional_mask": negative_conditional_mask,
        "negative_unconditional_mask": negative_unconditional_mask,
    }
