#include "transducer/pmerge.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("MergeDulpicatedHyp", &transducer::MergeDulpicatedHyp, "merge two paths with same prefix");
}