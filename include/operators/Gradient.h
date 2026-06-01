#pragma once

#include "field/ScalarField.h"
#include "equation/Term.h"
#include "scheme/CentralDifference.h"
#include "scheme/Isotropic.h"

namespace PhiX {

template<typename Scheme>
Term grad(const ScalarField& f, int axis, double coeff = 1.0);

// Default (CD2) overload
Term grad(const ScalarField& f, int axis, double coeff);

// Runtime-string scheme dispatch
// Supported: "CD2" (default), "Iso9"
Term grad(const ScalarField& f, int axis, const std::string& scheme, double coeff = 1.0);

} // namespace PhiX
