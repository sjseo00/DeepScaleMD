#ifndef TILE_GRID_INFO_H
#define TILE_GRID_INFO_H

// The cell grid and tile layout left behind by launch_cuda_tile_list_build when it is asked
// for the tiles only. The per-atom list builder searches the neighboring cells directly from
// this instead of going through a tile-pair list. The pointers refer to buffers owned by the
// builder and stay valid until its next call.
struct TileGridInfo {
    const int* tile_offset = nullptr;   // [cells] first tile of each cell
    const int* tile_count = nullptr;    // [cells] tiles in each cell
    const int* tile_cell_id = nullptr;  // [tiles] cell of each tile
    const double* tile_bbox_min = nullptr;  // [tiles * 3] bounding box relative to the cell origin
    const double* tile_bbox_max = nullptr;
    int nx = 0, ny = 0, nz = 0;
    double cell_size_x = 0.0, cell_size_y = 0.0, cell_size_z = 0.0;
    int stencil_radius = 0;             // neighbor cells searched in each direction
    // [cells] 1 when the cell holds a tile with a ghost atom; null without ghost atoms. The list
    // builder uses it to skip the boundary part of the rows of tiles that cannot have one.
    const unsigned char* cell_has_ghost = nullptr;
};

#endif
