#include "pmerge.h"
#include <iostream>
#include <vector>

namespace transducer{

float LogSumExp(float a, float b) {
  const float m = std::max(a, b);
  return m + std::log(std::exp(a - m) + std::exp(b - m));
}

bool IsDuplicateHyp(
    at::Tensor &paths,
    at::Tensor &length,
    int64_t hyp1_id,
    int64_t hyp2_id
){
    /*
    This Function is utilized to determine whether two paths are the same.
    paths: tensor with shape [num_hpys, length]
    length: tensor with shape [num_hpys]
    */

    AT_ASSERT(paths.dim() == 2);
    AT_ASSERT(length.dim() == 1);

    auto paths_v = paths.accessor<int64_t, 2>();
    auto length_v = length.accessor<int64_t, 1>();

    int64_t path1_len = length_v[hyp1_id];
    int64_t path2_len = length_v[hyp2_id];

    // if two paths have different lengths, they must be different.
    if (path1_len != path2_len){
        return false;
    }

    for (int64_t i=0; i < path1_len; i++){
        auto token1 = paths_v[hyp1_id][i];
        auto token2 = paths_v[hyp2_id][i];
        if (token1 != token2){
            return false;
        }
    }
    return true;
}

std::tuple<at::Tensor, at::Tensor>
MergeDulpicatedHyp(
    at::Tensor &paths,
    at::Tensor &paths_length,
    at::Tensor &scores
){
    /*
    This function is utilzed to merge the path with same tokens.
    Args:
        paths: LongTensor [in_num_hyps, lens]
        paths_length: LongTensor with shape [in_num_hyps]
        scores: FloatTensor with shape [in_um_hyps]
    Returns:
        merged_scores: FloatTensor with shape [in_num_hyps]
        indices: LongTensor with shape [in_num_hyps]. If the status is 1, it means this hyps is kept. Otherwise, the hyps will be removed. 
    */

    auto LongOption = torch::TensorOptions().dtype(torch::kLong);
    auto FloatOption = torch::TensorOptions().dtype(torch::kFloat32);

    AT_ASSERT(paths.dtype() == at::kLong);
    AT_ASSERT(paths_length.dtype() == at::kLong);
    AT_ASSERT(scores.dtype() == at::kFloat);

    AT_ASSERT(paths.dim() == 2);
    AT_ASSERT(paths_length.dim() == 1);
    AT_ASSERT(scores.dim() == 1);

    auto in_num_hyps = paths.size(0);

    auto scores_v = scores.accessor<float_t, 1>();
    at::Tensor merged_scores = torch::zeros({in_num_hyps}, FloatOption);
    at::Tensor status = torch::ones({in_num_hyps}, LongOption);

    auto merged_scores_v = merged_scores.accessor<float_t, 1>();
    auto status_v = status.accessor<int64_t, 1>();

    for (int64_t j=0; j < in_num_hyps; j++){

        if (status_v[j] == 0){
            // std::cout << "beam: " << j << " has been removed!" << std::endl;
            continue;
        }
        else{
            merged_scores_v[j] = scores_v[j];
        }
    
        int64_t k = j+1;
        while(k < in_num_hyps){
            // std::cout << "compare BEAM: " << j << " and BEAM: " << k << std::endl;
            bool is_same = IsDuplicateHyp(paths, paths_length, j, k);
            if (is_same){
                // std::cout << "BEAM: " << j << " and BEAM: " << k << " have the same prefix" << std::endl;
                merged_scores_v[j] = LogSumExp(scores_v[j], scores_v[k]);
                merged_scores_v[k] = -1e+10;
                status_v[k] = 0;
            }
            k++;
        }
    }

    // std::cout << "status: " <<status << std::endl;
    // std::cout << "merged_scores: " << merged_scores << std::endl;

    return std::make_tuple(merged_scores, status);

}
}