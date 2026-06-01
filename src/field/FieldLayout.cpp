#include "field/FieldLayout.h"

namespace PhiX {

FieldLayout::FieldLayout(const Mesh& mesh_, int ghost_, Centering centering_)
    : mesh(&mesh_)
    , ghost(ghost_)
    , centering(centering_)
{
    if (ghost_ < 0)
        throw std::invalid_argument("FieldLayout: ghost must be >= 0");
    if (centering_ != Centering::CELL)
        throw std::invalid_argument("FieldLayout: only CELL centering is implemented");

    for (int ax = 0; ax < 3; ++ax)
        storedDims[ax] = mesh_.n[ax] + 2 * ghost;

    storedSize = static_cast<std::size_t>(storedDims[0])
               * storedDims[1]
               * storedDims[2];
}

} // namespace PhiX
