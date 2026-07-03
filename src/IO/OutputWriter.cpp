#include "IO/OutputWriter.h"
#include "IO/FieldIO.h"

#include <iomanip>
#include <iostream>
#include <filesystem>
#include <sstream>

namespace {
// Format simulation time in scientific notation via a local stream, so the
// global std::cout format state is never touched (sticky flags like
// std::fixed would otherwise leak into every later double print).
std::string fmtTime(double t) {
    std::ostringstream ss;
    ss << std::scientific << std::setprecision(3) << t;
    return ss.str();
}
} // anonymous namespace

namespace PhiX {
namespace IO {

OutputWriter::OutputWriter(const nlohmann::json& output_config)
    : printInterval(output_config.at("print_interval").get<int>())
    , writeInterval(output_config.at("write_interval").get<int>())
{
    const std::string fmt = output_config.at("format").get<std::string>();
    writeBinary_ = (fmt == "BINARY" || fmt == "ALL");
    writeDat_    = (fmt == "DAT"    || fmt == "ALL");
    writeVtk_    = (fmt == "VTK"    || fmt == "ALL");

    // Optional VTS coordinate scale (e.g. 1e9 → nm); default 1.0 = physical metres.
    coordScale_ = output_config.value("coord_scale", 1.0);

    std::filesystem::create_directories("output");
    resetTimer();
}

void OutputWriter::writeFields(ScalarField& f, int step, double simTime) {
    f.downloadCurrFromDevice();
    std::string base = "output/" + f.name + "_" + std::to_string(step);
    if (writeBinary_) writeField(f, base + ".field", FieldFormat::BINARY);
    if (writeDat_)    writeField(f, base + ".dat",   FieldFormat::DAT);
    if (writeVtk_)    writeField(f, base + ".vts",   FieldFormat::VTS, coordScale_);
    std::cout << "  step " << step
              << "  t=" << fmtTime(simTime)
              << "  written: " << base << "\n" << std::flush;
}

void OutputWriter::printProgress(int step, double simTime) {
    double elapsed = std::chrono::duration<double>(Clock::now() - t_start_).count();
    std::ostringstream el;
    el << std::fixed << std::setprecision(1) << elapsed;
    std::cout << "  [progress] step=" << step
              << "  t=" << fmtTime(simTime)
              << "  elapsed=" << el.str() << "s\n" << std::flush;
}

void OutputWriter::resetTimer() {
    t_start_ = Clock::now();
}

} // namespace IO
} // namespace PhiX
