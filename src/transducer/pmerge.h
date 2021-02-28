#ifndef TRANSDUCER_PMERGE_H
#define TRANSDUCER_PMERGE_H

#include <torch/torch.h>

namespace transducer{

float LogSumExp(float a, float b);

bool IsDuplicateHyp(
    at::Tensor &paths,
    at::Tensor &length,
    int64_t hyp1_id,
    int64_t hyp2_id
);

std::tuple<at::Tensor, at::Tensor>
MergeDulpicatedHyp(
    at::Tensor &paths,
    at::Tensor &paths_length,
    at::Tensor &scores
);
    
}

#endif