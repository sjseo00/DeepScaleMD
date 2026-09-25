
#pragma once
#ifndef MLIP_MD_DATA_SOURCES_H
#define MLIP_MD_DATA_SOURCES_H

#include "force_libtorch/operations/operation_registry.h"
#include <functional>

namespace mlip {

struct MdDevicePointers {
    std::function<void*()> get_x;
    std::function<void*()> get_y;
    std::function<void*()> get_z;
    std::function<void*()> get_fx;
    std::function<void*()> get_fy;
    std::function<void*()> get_fz;
    std::function<void*()> get_type;
    std::function<void*()> get_global_id;

    std::function<void*()>    get_edge_src;  // int32 [num_edges] full list src
    std::function<void*()>    get_edge_dst;  // int32 [num_edges] full list dst
    std::function<void*()>    get_edge_coo_packed;
    std::function<int64_t()>  get_num_edges;

    std::function<int()> get_num_local;
    std::function<int()> get_num_ghost;
    std::function<int()> get_num_total;

    std::function<void*()> get_pe;
    std::function<void*()> get_virial;

    std::function<void*()> get_pos_interleaved;
    std::function<void*()> get_mlip_forces;
    std::function<void*()> get_cell;  // may be nullptr if not yet allocated

    std::function<void*()> get_edge_vectors;  // may be nullptr if not built

    std::function<void*()> get_batch;      // may be nullptr if not allocated

    std::function<void*()> get_num_atoms;  // may be nullptr if not allocated
};

void register_md_sources(const MdDevicePointers& ptrs);

} // namespace mlip
#endif // MLIP_MD_DATA_SOURCES_H
