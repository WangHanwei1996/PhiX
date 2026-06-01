#pragma once

#include "field/ScalarField.h"
#include "equation/Term.h"
#include "scheme/CentralDifference.h"

namespace PhiX {

template<typename Scheme>
Term grad(const ScalarField& f, int axis, double coeff = 1.0);

Term grad(const ScalarField& f, int axis, double coeff);

} // namespace PhiX
