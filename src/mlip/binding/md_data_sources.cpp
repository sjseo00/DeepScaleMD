/**
 * md_data_sources.cpp
 *
 * Registers MD device data sources and output targets.
 */

#include "mlip/binding/md_data_sources.h"
#include <iostream>

namespace mlip {

void register_md_sources(const MdDevicePointers& p)
{
    auto& sreg = SourceRegistry::instance();
    auto& treg = TargetRegistry::instance();

    // Helper to skip re-registration
    auto reg_src = [&](const std::string& name, DataSourceDescriptor d) {
        if (!sreg.has(name)) sreg.register_source(name, std::move(d));
    };
    auto reg_tgt = [&](const std::string& name, PointerResolver r) {
        if (!treg.has(name)) treg.register_target(name, std::move(r));
    };

    // ---- Sources ----

    // md.positions — AoS [N, 3] float64 from interleaved buffer
    reg_src("md.positions", {
        ScalarType::Float64, 2,
        [](const OpContext& c) -> ShapeSpec {
            return {{ShapeSymbol::Nall, 0}, {ShapeSymbol::Literal, 3}};
        },
        [pg = p.get_pos_interleaved](const OpContext&) -> void* { return pg ? pg() : nullptr; }
    });

    reg_src("md.pos_x", {
        ScalarType::Float64, 1,
        [](const OpContext& c) { return ShapeSpec{{ShapeSymbol::Nall, 0}}; },
        [pg = p.get_x](const OpContext&) -> void* { return pg(); }
    });
    reg_src("md.pos_y", {
        ScalarType::Float64, 1,
        [](const OpContext& c) { return ShapeSpec{{ShapeSymbol::Nall, 0}}; },
        [pg = p.get_y](const OpContext&) -> void* { return pg(); }
    });
    reg_src("md.pos_z", {
        ScalarType::Float64, 1,
        [](const OpContext& c) { return ShapeSpec{{ShapeSymbol::Nall, 0}}; },
        [pg = p.get_z](const OpContext&) -> void* { return pg(); }
    });

    reg_src("md.species", {
        ScalarType::Int32, 1,
        [](const OpContext& c) { return ShapeSpec{{ShapeSymbol::Nall, 0}}; },
        [pg = p.get_type](const OpContext&) -> void* { return pg(); }
    });

    reg_src("md.atomic_numbers", {  // alias; actual mapping done by species_map op
        ScalarType::Int32, 1,
        [](const OpContext& c) { return ShapeSpec{{ShapeSymbol::Nall, 0}}; },
        [pg = p.get_type](const OpContext&) -> void* { return pg(); }
    });

    // md.neighbor_list — packed half-list [src (E_half) | dst (E_half)] int32.
    // The full_directed_edge_list preprocess op reads this and expands to [2, E_full] int64.
    // Buffer built by build_coo_edge_list_on_device into d_edge_coo_packed_.
    // num_edges in context = E_full (2*E_half); the packed buffer has E_half pairs.
    reg_src("md.neighbor_list", {
        ScalarType::Int32, 2,
        [](const OpContext& c) { return ShapeSpec{{ShapeSymbol::Literal, 2},
                                                  {ShapeSymbol::Nedges, 0}}; },
        [pg = p.get_edge_coo_packed](const OpContext&) -> void* { return pg ? pg() : nullptr; }
    });

    reg_src("md.edge_index", {
        ScalarType::Int32, 2,
        [](const OpContext& c) { return ShapeSpec{{ShapeSymbol::Literal, 2},
                                                  {ShapeSymbol::Nedges, 0}}; },
        [pg = p.get_edge_src](const OpContext&) -> void* { return pg(); }
    });

    reg_src("md.global_id", {
        ScalarType::Int32, 1,
        [](const OpContext& c) { return ShapeSpec{{ShapeSymbol::Nall, 0}}; },
        [pg = p.get_global_id](const OpContext&) -> void* { return pg(); }
    });

    // md.cell — orthorhombic box [3,3] float64 flattened as [9] = [lx,0,0, 0,ly,0, 0,0,lz]
    // Updated each step by compute_mlip_runtime_forces before the MLIP call.
    reg_src("md.cell", {
        ScalarType::Float64, 2,
        [](const OpContext& c) { return ShapeSpec{{ShapeSymbol::Literal, 3},
                                                  {ShapeSymbol::Literal, 3}}; },
        [pg = p.get_cell](const OpContext&) -> void* { return pg ? pg() : nullptr; }
    });

    // md.edge_vectors — minimum-image edge vectors [E_full, 3] float32.
    // Computed each step in compute_mlip_runtime_forces after edge list build.
    reg_src("md.edge_vectors", {
        ScalarType::Float32, 2,
        [](const OpContext& c) { return ShapeSpec{{ShapeSymbol::Nedges, 0},
                                                  {ShapeSymbol::Literal, 3}}; },
        [pg = p.get_edge_vectors](const OpContext&) -> void* { return pg ? pg() : nullptr; }
    });

    // md.batch — batch indices [N] int64, all zeros (single-system models like NequIP).
    reg_src("md.batch", {
        ScalarType::Int64, 1,
        [](const OpContext& c) { return ShapeSpec{{ShapeSymbol::Nall, 0}}; },
        [pg = p.get_batch](const OpContext&) -> void* { return pg ? pg() : nullptr; }
    });

    // md.num_atoms — number of atoms [1] int64.
    reg_src("md.num_atoms", {
        ScalarType::Int64, 1,
        [](const OpContext& c) { return ShapeSpec{{ShapeSymbol::Literal, 1}}; },
        [pg = p.get_num_atoms](const OpContext&) -> void* { return pg ? pg() : nullptr; }
    });

    // ---- Targets ----

    reg_tgt("md.forces",      [pg = p.get_fx](const OpContext&) -> void* { return pg(); });
    reg_tgt("md.mlip_forces", [pg = p.get_mlip_forces](const OpContext&) -> void* { return pg ? pg() : nullptr; });
    reg_tgt("md.energy",  [pg = p.get_pe](const OpContext&) -> void* { return pg(); });
    reg_tgt("md.virial",  [pg = p.get_virial](const OpContext&) -> void* { return pg(); });

    std::cout << "[MLIP] MD data sources registered.\n";
}

} // namespace mlip
