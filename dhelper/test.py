import torch
import dhelper


paths = torch.LongTensor(
       [[   1,  760,    1],
        [   1,  760,    4],
        [   1,  760,  255],
        [   1,  760,   14],
        [   1,  760,    3],
        [   1,    1,    1],
        [   1,  760,    1],
        [   1,   14,    1],
        [   1,    4,    1],
        [   1,  154,    1],
        [   1, 1878,    1],
        [   1, 1878,  760],
        [   1, 1878,    3],
        [   1, 1878,   14],
        [   1, 1878,   13],
        [   1,  437,    1],
        [   1,  437,  760],
        [   1,  437,  273],
        [   1,  437,   14],
        [   1,  437,    3],
        [   1,  195,    1],
        [   1,  195,  760],
        [   1,  195,    4],
        [   1,  195,   14],
        [   1,  195,  111]])
paths_length = torch.LongTensor(
    [1, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2])
scores = torch.FloatTensor(
       [-1.6785e-04, -3.1105e+01, -3.1396e+01, -3.1461e+01, -3.1622e+01,
        -8.7159e+00, -2.4534e+01, -3.2179e+01, -3.2259e+01, -3.2372e+01,
        -1.3672e+01, -4.2483e+01, -4.2548e+01, -4.3313e+01, -4.3875e+01,
        -1.4625e+01, -3.9936e+01, -4.0471e+01, -4.0828e+01, -4.1123e+01,
        -1.5679e+01, -4.0469e+01, -4.2335e+01, -4.3674e+01, -4.4092e+01])


topk_paths, topk_paths_length, topk_scores, indices = dhelper.MergeDuplicatedPaths(paths, paths_length, scores, 5)
print('topk_paths', topk_paths)
print('topk_paths_length', topk_paths_length)
print('topk_scores', topk_scores)
print('indices', indices)