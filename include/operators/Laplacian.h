#pragma once

#include "field/ScalarField.h"
#include "equation/Term.h"
#include "scheme/CentralDifference.h"

namespace PhiX {

template<typename Scheme>
Term lap(const ScalarField& f, double coeff = 1.0);

Term lap(const ScalarField& f, double coeff);

} // namespace PhiX
