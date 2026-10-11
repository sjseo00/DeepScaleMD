#include "neighbor_list.h"

#include <stdexcept>

std::unique_ptr<NeighborList> create_neighbor_list(NeighborListBackend kind, Real cutoff, Real skin) {
    switch (kind) {
        case NeighborListBackend::CPU_SPATIAL_VERLET:
            return std::make_unique<SpatialVerletList>(cutoff, skin);
    }
    throw std::runtime_error("create_neighbor_list: unhandled NeighborListBackend");
}
