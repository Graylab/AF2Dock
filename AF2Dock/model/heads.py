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

import torch.nn as nn

from openfold.utils.loss import (
    compute_plddt,
    compute_tm,
    compute_predicted_aligned_error,
)
from openfold.model.heads import (
    PerResidueLDDTCaPredictor,
    DistogramHead,
    ExperimentallyResolvedHead,
    TMScoreHead,
)

class AuxiliaryHeads(nn.Module):
    def __init__(self, config):
        super(AuxiliaryHeads, self).__init__()

        self.plddt = PerResidueLDDTCaPredictor(
            **config["lddt"],
        )

        self.distogram = DistogramHead(
            **config["distogram"],
        )

        self.experimentally_resolved = ExperimentallyResolvedHead(
            **config["experimentally_resolved"],
        )

        if config.tm.enabled:
            self.tm = TMScoreHead(
                **config.tm,
            )

        self.config = config

    def forward(self, outputs):
        aux_out = {}
        lddt_logits = self.plddt(outputs["sm"]["single"])
        aux_out["lddt_logits"] = lddt_logits

        # Required for relaxation later on
        aux_out["plddt"] = compute_plddt(lddt_logits)

        distogram_logits = self.distogram(outputs["pair"])
        aux_out["distogram_logits"] = distogram_logits

        experimentally_resolved_logits = self.experimentally_resolved(
            outputs["single"]
        )
        aux_out[
            "experimentally_resolved_logits"
        ] = experimentally_resolved_logits

        if self.config.tm.enabled:
            tm_logits = self.tm(outputs["pair"])
            aux_out["tm_logits"] = tm_logits
            aux_out["ptm_score"] = compute_tm(
                tm_logits, **self.config.tm
            )
            asym_id = outputs.get("asym_id")
            if asym_id is not None:
                aux_out["iptm_score"] = compute_tm(
                    tm_logits, asym_id=asym_id, interface=True, **self.config.tm
                )
                aux_out["weighted_ptm_score"] = (self.config.tm["iptm_weight"] * aux_out["iptm_score"]
                                                 + self.config.tm["ptm_weight"] * aux_out["ptm_score"])

            aux_out.update(
                compute_predicted_aligned_error(
                    tm_logits,
                    **self.config.tm,
                )
            )

        return aux_out
