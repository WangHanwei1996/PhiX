#pragma once

#include "boundary/BoundaryCondition.h"

#include <nlohmann/json.hpp>
#include <memory>
#include <vector>

namespace PhiX {

// ---------------------------------------------------------------------------
// BCSet — owns a collection of BoundaryCondition objects and exposes raw
// pointers for use with Solver / SolverStep.
// ---------------------------------------------------------------------------
struct BCSet {
    std::vector<std::unique_ptr<BoundaryCondition>> storage;  ///< owns lifetime
    std::vector<BoundaryCondition*> ptrs;                     ///< non-owning view
};

// ---------------------------------------------------------------------------
// buildBCs — construct boundary conditions from a JSON config block.
//
// Expected JSON layout (2D example):
//   {
//       "x_min": "Periodic",   // "Periodic" | "NoFlux"
//       "x_max": "Periodic",
//       "y_min": "NoFlux",
//       "y_max": "NoFlux"
//   }
//
// Rules:
//   - Periodic must be set on both sides of the same axis; mismatched
//     Periodic/non-Periodic on the same axis throws std::runtime_error.
//   - z_min / z_max are processed when present (3D problems).
//
// Supported BC types: "Periodic", "NoFlux".
// ---------------------------------------------------------------------------
BCSet buildBCs(const nlohmann::json& bc_config);

} // namespace PhiX
