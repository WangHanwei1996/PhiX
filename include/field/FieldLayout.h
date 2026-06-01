#pragma once

#include "mesh/Mesh.h"

#include <cstddef>
#include <stdexcept>

namespace PhiX {

enum class Centering {
    CELL
};

// ---------------------------------------------------------------------------
// FieldLayout
//
// Describes the padded storage geometry of a field on a Mesh.
// The current implementation supports only cell-centred storage, but the
// object exists so later versions can extend it to face/node layouts without
// re-embedding storage rules into ScalarField / VectorField.
// ---------------------------------------------------------------------------
class FieldLayout {
public:
    const Mesh*  mesh = nullptr;
    int          ghost = 0;
    Centering    centering = Centering::CELL;
    int          storedDims[3] = {0, 0, 0};
    std::size_t  storedSize = 0;

    FieldLayout() = default;
    explicit FieldLayout(const Mesh& mesh,
                         int ghost = 1,
                         Centering centering = Centering::CELL);

    const Mesh& meshRef() const {
        if (!mesh) throw std::runtime_error("FieldLayout: mesh is null");
        return *mesh;
    }

    int index(int i, int j, int k) const {
        return (i + ghost)
             + storedDims[0] * ((j + ghost)
             + storedDims[1] *  (k + ghost));
    }
    int index(int i, int j) const { return index(i, j, 0); }
    int index(int i) const { return index(i, 0, 0); }
};

} // namespace PhiX
