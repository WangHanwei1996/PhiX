#include "IO/OutputWriter.h"
#include "IO/FieldIO.h"

#include <iomanip>
#include <iostream>
#include <filesystem>

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

    std::filesystem::create_directories("output");
    resetTimer();
}

void OutputWriter::writeFields(ScalarField& f, int step, double simTime) {
    f.downloadCurrFromDevice();
    std::string base = "output/" + f.name + "_" + std::to_string(step);
    if (writeBinary_) writeField(f, base + ".field", FieldFormat::BINARY);
    if (writeDat_)    writeField(f, base + ".dat",   FieldFormat::DAT);
    if (writeVtk_)    writeField(f, base + ".vts",   FieldFormat::VTS);
    std::cout << "  step " << step
              << "  t=" << simTime
              << "  written: " << base << "\n" << std::flush;
}

void OutputWriter::printProgress(int step, double simTime) {
    double elapsed = std::chrono::duration<double>(Clock::now() - t_start_).count();
    std::cout << "  [progress] step=" << step
              << "  t=" << simTime
              << "  elapsed=" << std::fixed << std::setprecision(1)
              << elapsed << "s\n" << std::flush;
}

void OutputWriter::resetTimer() {
    t_start_ = Clock::now();
}

} // namespace IO
} // namespace PhiX
