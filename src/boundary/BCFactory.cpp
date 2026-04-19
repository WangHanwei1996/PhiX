#include "boundary/BCFactory.h"
#include "boundary/PeriodicBC.h"
#include "boundary/NoFluxBC.h"

#include <stdexcept>
#include <string>

namespace PhiX {

// ---------------------------------------------------------------------------
// Helper: process one axis pair (lo/hi) from the JSON config
// ---------------------------------------------------------------------------
static void addAxisBCs(BCSet& set, Axis axis,
                       const std::string& lo, const std::string& hi)
{
    bool lo_periodic = (lo == "Periodic");
    bool hi_periodic = (hi == "Periodic");

    if (lo_periodic != hi_periodic) {
        const char* name = (axis == Axis::X) ? "X" : (axis == Axis::Y) ? "Y" : "Z";
        throw std::runtime_error(
            std::string("buildBCs: Periodic BC must be set on both sides of axis ")
            + name + ", got \"" + lo + "\" / \"" + hi + "\"");
    }

    if (lo_periodic) {
        set.storage.push_back(std::make_unique<PeriodicBC>(axis));
        set.ptrs.push_back(set.storage.back().get());
        return;
    }

    // Non-periodic: handle each side independently
    auto addSide = [&](const std::string& type, Side side) {
        if (type == "NoFlux") {
            set.storage.push_back(std::make_unique<NoFluxBC>(axis, side));
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
BCSet buildBCs(const nlohmann::json& bc_config)
{
    BCSet set;

    // X axis (required)
    addAxisBCs(set, Axis::X,
               bc_config.at("x_min").get<std::string>(),
               bc_config.at("x_max").get<std::string>());

    // Y axis (required for 2D/3D)
    if (bc_config.contains("y_min") && bc_config.contains("y_max")) {
        addAxisBCs(set, Axis::Y,
                   bc_config.at("y_min").get<std::string>(),
                   bc_config.at("y_max").get<std::string>());
    }

    // Z axis (optional, for 3D)
    if (bc_config.contains("z_min") && bc_config.contains("z_max")) {
        addAxisBCs(set, Axis::Z,
                   bc_config.at("z_min").get<std::string>(),
                   bc_config.at("z_max").get<std::string>());
    }

    return set;
}

} // namespace PhiX
