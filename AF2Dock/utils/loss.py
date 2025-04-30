# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

import logging
from openfold.utils import loss as of_loss

logger = logging.getLogger(__name__)

class AF2DockLoss(nn.Module):
    """Aggregation of the various losses described in the supplement"""

    def __init__(self, config):
        super(AF2DockLoss, self).__init__()
        self.config = config

    def loss(self, out, batch, _return_breakdown=False):
        """
        Rename previous forward() as loss()
        so that can be reused in the subclass 
        """
        if "violation" not in out.keys():
            out["violation"] = of_loss.find_structural_violations(
                batch,
                out["sm"]["positions"][-1],
                **self.config.violation,
            )

        if "renamed_atom14_gt_positions" not in out.keys():
            batch.update(
                of_loss.compute_renamed_ground_truth(
                    batch,
                    out["sm"]["positions"][-1],
                )
            )

        loss_fns = {
            "distogram": lambda: of_loss.distogram_loss(
                logits=out["distogram_logits"],
                **{**batch, **self.config.distogram},
            ),
            "experimentally_resolved": lambda: of_loss.experimentally_resolved_loss(
                logits=out["experimentally_resolved_logits"],
                **{**batch, **self.config.experimentally_resolved},
            ),
            "fape": lambda: of_loss.fape_loss(
                out,
                batch,
                self.config.fape,
            ),
            "plddt_loss": lambda: of_loss.lddt_loss(
                logits=out["lddt_logits"],
                all_atom_pred_pos=out["final_atom_positions"],
                **{**batch, **self.config.plddt_loss},
            ),
            "supervised_chi": lambda: of_loss.supervised_chi_loss(
                out["sm"]["angles"],
                out["sm"]["unnormalized_angles"],
                **{**batch, **self.config.supervised_chi},
            ),
            "violation": lambda: of_loss.violation_loss(
                out["violation"],
                **{**batch, **self.config.violation},
            ),
        }

        if self.config.tm.enabled:
            loss_fns["tm"] = lambda: of_loss.tm_loss(
                logits=out["tm_logits"],
                **{**batch, **out, **self.config.tm},
            )

        if self.config.chain_center_of_mass.enabled:
            loss_fns["chain_center_of_mass"] = lambda: of_loss.chain_center_of_mass_loss(
                all_atom_pred_pos=out["final_atom_positions"],
                **{**batch, **self.config.chain_center_of_mass},
            )

        cum_loss = 0.
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.config[loss_name].weight
            loss = loss_fn()
            if torch.isnan(loss) or torch.isinf(loss):
                # for k,v in batch.items():
                #    if torch.any(torch.isnan(v)) or torch.any(torch.isinf(v)):
                #        logging.warning(f"{k}: is nan")
                # logging.warning(f"{loss_name}: {loss}")
                logging.warning(f"{loss_name} loss is NaN. Skipping...")
                loss = loss.new_tensor(0., requires_grad=True)
            cum_loss = cum_loss + weight * loss
            losses[loss_name] = loss.detach().clone()
        losses["unscaled_loss"] = cum_loss.detach().clone()

        # Scale the loss by the square root of the minimum of the crop size and
        # the (average) sequence length. See subsection 1.9.
        seq_len = torch.mean(batch["seq_length"].float())
        crop_len = batch["aatype"].shape[-1]
        cum_loss = cum_loss * torch.sqrt(min(seq_len, crop_len))

        losses["loss"] = cum_loss.detach().clone()

        if not _return_breakdown:
            return cum_loss

        return cum_loss, losses

    def forward(self, out, batch, _return_breakdown=False):
        if not _return_breakdown:
            cum_loss = self.loss(out, batch, _return_breakdown)
            return cum_loss
        else:
            cum_loss, losses = self.loss(out, batch, _return_breakdown)
            return cum_loss, losses
