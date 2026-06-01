#include "boundary/BCFactory.h"
#include "boundary/PeriodicBC.h"
#include "boundary/NoFluxBC.h"
#include "boundary/FixedBC.h"

#include <stdexcept>
#include <string>

namespace PhiX {

// ---------------------------------------------------------------------------
// Helper: extract BC type string from a JSON value (string or object)
// ---------------------------------------------------------------------------
static std::string getBCType(const nlohmann::json& v)
{
    if (v.is_string()) return v.get<std::string>();
    if (v.is_object()) return v.at("type").get<std::string>();
    throw std::runtime_error("buildBCs: BC entry must be a string or an object");
}

// ---------------------------------------------------------------------------
// Helper: process one axis pair (lo/hi) from the JSON config
// ---------------------------------------------------------------------------
static void addAxisBCs(const Mesh& mesh, BCSet& set, Axis axis,
                       const nlohmann::json& lo, const nlohmann::json& hi)
{
    std::string lo_type = getBCType(lo);
    std::string hi_type = getBCType(hi);

    bool lo_periodic = (lo_type == "Periodic");
    bool hi_periodic = (hi_type == "Periodic");

    if (lo_periodic != hi_periodic) {
        const char* name = (axis == Axis::X) ? "X" : (axis == Axis::Y) ? "Y" : "Z";
        throw std::runtime_error(
            std::string("buildBCs: Periodic BC must be set on both sides of axis ")
            + name + ", got \"" + lo_type + "\" / \"" + hi_type + "\"");
    }

    if (lo_periodic) {
        set.storage.push_back(std::make_unique<PeriodicBC>(mesh.facePatch(axis, Side::LOW)));
        set.ptrs.push_back(set.storage.back().get());
        return;
    }

    // Non-periodic: handle each side independently
    auto addSide = [&](const nlohmann::json& cfg, Side side) {
        std::string type = getBCType(cfg);
        const Patch& patch = mesh.facePatch(axis, side);
        if (type == "NoFlux") {
            set.storage.push_back(std::make_unique<NoFluxBC>(patch));
            set.ptrs.push_back(set.storage.back().get());
        } else if (type == "Fixed") {
            double val = (cfg.is_object() && cfg.contains("value"))
                         ? cfg.at("value").get<double>() : 0.0;
            set.storage.push_back(std::make_unique<FixedBC>(patch, val));
            set.ptrs.push_back(set.storage.back().get());
        } else {
            throw std::runtime_error(
                "buildBCs: unsupported BC type \"" + type + "\"");
        }
    };

    addSide(lo, Side::LOW);
    addSide(hi, Side::HIGH);
}

// ---------------------------------------------------------------------------
// buildBCs
// ---------------------------------------------------------------------------
BCSet buildBCs(const Mesh& mesh, const nlohmann::json& bc_config)
{
    BCSet set;

    // X axis (required)
    addAxisBCs(mesh, set, Axis::X,
               bc_config.at("x_min"),
               bc_config.at("x_max"));

    // Y axis (required for 2D/3D)
    if (bc_config.contains("y_min") && bc_config.contains("y_max")) {
        addAxisBCs(mesh, set, Axis::Y,
                   bc_config.at("y_min"),
                   bc_config.at("y_max"));
    }

    // Z axis (optional, for 3D)
    if (bc_config.contains("z_min") && bc_config.contains("z_max")) {
        addAxisBCs(mesh, set, Axis::Z,
                   bc_config.at("z_min"),
                   bc_config.at("z_max"));
    }

    return set;
}

} // namespace PhiX
