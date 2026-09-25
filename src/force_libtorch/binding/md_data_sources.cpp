
#include "force_libtorch/binding/md_data_sources.h"
#include <iostream>

namespace mlip {

void register_md_sources(const MdDevicePointers& p)
{
    auto& sreg = SourceRegistry::instance();
    auto& treg = TargetRegistry::instance();

    auto reg_src = [&](const std::string& name, DataSourceDescriptor d) {
        if (!sreg.has(name)) sreg.register_source(name, std::move(d));
    };
    auto reg_tgt = [&](const std::string& name, PointerResolver r) {
        if (!treg.has(name)) treg.register_target(name, std::move(r));
    };


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

    reg_src("md.cell", {
        ScalarType::Float64, 2,
        [](const OpContext& c) { return ShapeSpec{{ShapeSymbol::Literal, 3},
                                                  {ShapeSymbol::Literal, 3}}; },
        [pg = p.get_cell](const OpContext&) -> void* { return pg ? pg() : nullptr; }
    });

    reg_src("md.edge_vectors", {
        ScalarType::Float32, 2,
        [](const OpContext& c) { return ShapeSpec{{ShapeSymbol::Nedges, 0},
                                                  {ShapeSymbol::Literal, 3}}; },
        [pg = p.get_edge_vectors](const OpContext&) -> void* { return pg ? pg() : nullptr; }
    });

    reg_src("md.batch", {
        ScalarType::Int64, 1,
        [](const OpContext& c) { return ShapeSpec{{ShapeSymbol::Nall, 0}}; },
        [pg = p.get_batch](const OpContext&) -> void* { return pg ? pg() : nullptr; }
    });

    reg_src("md.num_atoms", {
        ScalarType::Int64, 1,
        [](const OpContext& c) { return ShapeSpec{{ShapeSymbol::Literal, 1}}; },
        [pg = p.get_num_atoms](const OpContext&) -> void* { return pg ? pg() : nullptr; }
    });


    reg_tgt("md.forces",      [pg = p.get_fx](const OpContext&) -> void* { return pg(); });
    reg_tgt("md.mlip_forces", [pg = p.get_mlip_forces](const OpContext&) -> void* { return pg ? pg() : nullptr; });
    reg_tgt("md.energy",  [pg = p.get_pe](const OpContext&) -> void* { return pg(); });
    reg_tgt("md.virial",  [pg = p.get_virial](const OpContext&) -> void* { return pg(); });

    std::cout << "[MLIP] MD data sources registered.\n";
}

} // namespace mlip
