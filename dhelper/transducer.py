import torch
from ._ext import dhelper


def MergeDuplicatedPaths(paths, paths_length, scores, topk=5, path_include_bos=True):
    """Merge duplicated paths during the inference of RNN-Transducer model.

    Args:
        paths ([LongTensor or IntTensor]): the padded path tensor with shape [in_num_hyps, max_lenth].
        paths_length ([LongTensor of IntTensor]): the length tensor with shape [in_num_hyps]
        scores: ([FloatTensor]): The score tensor with shape [in_num_hyps]
        topk: Keep TopK hypys, Default: 5, if topk=-1, it means keep all hyps.
    Return:
        topk_paths: ([LongTensor or IntTensor]): the merged path tensor with shape [out_num_hyps, max_lenth].
        topk_paths_length: ([IntTensor]) the length of merged paths with shape [out_num_hyps]
        topk_scores: ([FloatTensor]) the merged scores with shape [out_num_hyps]
        indices: ([LongTensor]): the index of merged_path, [out_num_hyps]
    """

    assert paths.dim() == 2
    assert paths_length.dim() == 1
    assert scores.dim() == 1
    device = paths.device
    out_scores, status = \
        dhelper.MergeDulpicatedHyp(
            paths[:, 1:].cpu().contiguous() if path_include_bos else paths.cpu().contiguous(),
            paths_length.cpu().contiguous(),
            scores.cpu().contiguous())

    keep_num_hyps = torch.sum(status).item()
    out_num_hyps = keep_num_hyps if topk == -1 else min(keep_num_hyps, topk)

    topk_scores, indices = torch.topk(out_scores, out_num_hyps)
    topk_paths = paths[indices]
    topk_paths_length = paths_length[indices]

    return topk_paths, topk_paths_length, topk_scores.to(device), indices.to(device)

