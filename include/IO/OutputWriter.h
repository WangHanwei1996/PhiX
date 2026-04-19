#pragma once

#include "IO/FieldFormat.h"
#include "field/ScalarField.h"

#include <nlohmann/json.hpp>
#include <chrono>
#include <string>

namespace PhiX {
namespace IO {

// ---------------------------------------------------------------------------
// OutputWriter
//
// Manages field output and progress logging based on a JSON config block.
//
// Expected JSON layout:
//   {
//       "print_interval": 10000,
//       "write_interval": 100000,
//       "format"        : "ALL"      // "BINARY", "DAT", "VTK" or "ALL"
//   }
//
// Usage:
//   IO::OutputWriter writer(cfg["output"]);
//   writer.writeFields(c, step, simTime);       // write field files
//   writer.printProgress(step, simTime);         // print progress + elapsed
//   if (writer.shouldPrint(step)) { ... }
//   if (writer.shouldWrite(step)) { ... }
// ---------------------------------------------------------------------------
class OutputWriter {
public:
    explicit OutputWriter(const nlohmann::json& output_config);

    int printInterval;
    int writeInterval;

    /// Check whether this step should trigger a progress print.
    bool shouldPrint(int step) const { return step % printInterval == 0; }

    /// Check whether this step should trigger a file write.
    bool shouldWrite(int step) const { return step % writeInterval == 0; }

    /// Write field to output/ in all configured formats.
    /// Downloads from device, writes files, and prints a status line.
    void writeFields(ScalarField& f, int step, double simTime);

    /// Print a progress line with step, simulation time, and wall-clock elapsed.
    void printProgress(int step, double simTime);

    /// Reset the internal wall-clock timer (called automatically on construction).
    void resetTimer();

private:
    bool writeBinary_;
    bool writeDat_;
    bool writeVtk_;

    using Clock = std::chrono::steady_clock;
    Clock::time_point t_start_;
};

} // namespace IO
} // namespace PhiX
