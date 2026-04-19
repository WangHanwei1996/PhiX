#pragma once

namespace PhiX {

// ---------------------------------------------------------------------------
// Output format selector for field IO
// ---------------------------------------------------------------------------
enum class FieldFormat {
    BINARY,   ///< Custom binary format (.field / .vfield)  — default; smallest files
    DAT,      ///< ASCII text with x y z value columns — easy for gnuplot/matplotlib
    VTS       ///< VTK XML StructuredGrid (.vts) — opens directly in ParaView/VisIt
};

} // namespace PhiX
