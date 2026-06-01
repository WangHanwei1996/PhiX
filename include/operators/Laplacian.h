#pragma once

#include "field/ScalarField.h"
#include "equation/Term.h"
#include "scheme/CentralDifference.h"
#include "scheme/Isotropic.h"

namespace PhiX {

template<typename Scheme>
Term lap(const ScalarField& f, double coeff = 1.0);

// Default (CD2) overload — no second default arg to avoid redefinition
Term lap(const ScalarField& f, double coeff);

// Convenience: runtime-string scheme dispatch
// Supported: "CD2" (default), "Iso9"
Term lap(const ScalarField& f, const std::string& scheme, double coeff = 1.0);

} // namespace PhiX
