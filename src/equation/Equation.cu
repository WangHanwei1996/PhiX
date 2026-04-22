#include "equation/Equation.h"
#include "field/ScalarField.h"
#include "field/VectorField.h"
#include "equation/Term.h"
#include "boundary/BoundaryCondition.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>

namespace PhiX {

// ---------------------------------------------------------------------------
// CUDA error-checking macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess)                                                 \
            throw std::runtime_error(                                          \
                std::string("CUDA error in " __FILE__ " line ")               \
                + std::to_string(__LINE__) + ": "                             \
                + cudaGetErrorString(_e));                                     \
    } while (0)

// ===========================================================================
// GPU kernels for built-in differential operators
// ===========================================================================

// ---------------------------------------------------------------------------
// Laplacian accumulate:
//   rhs[i,j,k] += coeff * (d²f/dx² [+ d²f/dy²] [+ d²f/dz²])
//
// 2nd-order central FD:
//   d²f/dx² ≈ (f[i+1,j,k] - 2f[i,j,k] + f[i-1,j,k]) / dx²
//
// Ghost cells of `src` must be valid before calling (BCs applied by Solver).
// Only physical cells are written in `rhs`.
// ---------------------------------------------------------------------------
__global__ void kernel_lap_accumulate(
        double*       rhs,
        const double* src,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,          // storedDims[0], storedDims[1]
        int ghost, int dim,
        double inv_dx2, double inv_dy2, double inv_dz2)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int is = i + ghost;
    int js = j + ghost;
    int ks = k + ghost;

    // Flat stored indices for stencil neighbours
    int c  = is       + sx * (js       + sy * ks);
    int xm = (is - 1) + sx * (js       + sy * ks);
    int xp = (is + 1) + sx * (js       + sy * ks);

    double val = (src[xp] - 2.0 * src[c] + src[xm]) * inv_dx2;

    if (dim >= 2) {
        int ym = is + sx * ((js - 1) + sy * ks);
        int yp = is + sx * ((js + 1) + sy * ks);
        val += (src[yp] - 2.0 * src[c] + src[ym]) * inv_dy2;
    }

    if (dim >= 3) {
        int zm = is + sx * (js + sy * (ks - 1));
        int zp = is + sx * (js + sy * (ks + 1));
        val += (src[zp] - 2.0 * src[c] + src[zm]) * inv_dz2;
    }

    rhs[c] += coeff * val;
}

// ---------------------------------------------------------------------------
// Gradient accumulate (one component):
//   rhs[i,j,k] += coeff * df/dx_axis
//
// 2nd-order central FD:
//   df/dx ≈ (f[i+1] - f[i-1]) / (2*dx)
// ---------------------------------------------------------------------------
__global__ void kernel_grad_accumulate(
        double*       rhs,
        const double* src,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,
        int ghost, int axis,
        double inv_2d)   // 1 / (2 * d[axis])
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int is = i + ghost;
    int js = j + ghost;
    int ks = k + ghost;

    int c   = is + sx * (js + sy * ks);
    int fwd, bwd;

    if (axis == 0) {
        fwd = (is + 1) + sx * (js + sy * ks);
        bwd = (is - 1) + sx * (js + sy * ks);
    } else if (axis == 1) {
        fwd = is + sx * ((js + 1) + sy * ks);
        bwd = is + sx * ((js - 1) + sy * ks);
    } else {
        fwd = is + sx * (js + sy * (ks + 1));
        bwd = is + sx * (js + sy * (ks - 1));
    }

    rhs[c] += coeff * (src[fwd] - src[bwd]) * inv_2d;
}

// ===========================================================================
// Built-in operator factories (return Term with launchers set up)
// ===========================================================================

Term lap(const ScalarField& f, double coeff) {
    Term t;
    t.type  = TermType::LAPLACIAN;
    t.field = &f;
    t.coeff = coeff;

    // Capture layout and mesh spacing at construction time
    int    nx = f.mesh.n[0], ny = f.mesh.n[1], nz = f.mesh.n[2];
    int    sx = f.storedDims[0], sy = f.storedDims[1];
    int    g  = f.ghost;
    int    dim = f.mesh.dim;
    double inv_dx2 = 1.0 / (f.mesh.d[0] * f.mesh.d[0]);
    double inv_dy2 = (dim >= 2) ? 1.0 / (f.mesh.d[1] * f.mesh.d[1]) : 0.0;
    double inv_dz2 = (dim >= 3) ? 1.0 / (f.mesh.d[2] * f.mesh.d[2]) : 0.0;

    // Capture pointer-to-field so RK4's d_curr swap trick keeps working.
    const ScalarField* pf = &f;

    t.gpu_launcher = [pf, nx, ny, nz, sx, sy, g, dim, inv_dx2, inv_dy2, inv_dz2]
                     (double* d_rhs, double c, ScratchPool&) {
        const double* d_src = pf->d_curr;
        if (!d_src)
            throw std::runtime_error("lap GPU: source field not on device");
        int total = nx * ny * nz;
        kernel_lap_accumulate<<<(total + 255) / 256, 256>>>(
            d_rhs, d_src, c,
            nx, ny, nz, sx, sy,
            g, dim, inv_dx2, inv_dy2, inv_dz2);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("lap GPU kernel error: ") + cudaGetErrorString(err));
    };

    t.cpu_kernel = [pf, nx, ny, nz, sx, sy, g, dim, inv_dx2, inv_dy2, inv_dz2]
                   (double* rhs, double c, ScratchPool&) {
        const double* src = pf->curr.data();
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int is = i+g, js = j+g, ks = k+g;
            int ctr = is + sx*(js + sy*ks);
            double val =
                (src[(is+1)+sx*(js+sy*ks)] - 2.0*src[ctr] + src[(is-1)+sx*(js+sy*ks)]) * inv_dx2;
            if (dim >= 2)
                val += (src[is+sx*((js+1)+sy*ks)] - 2.0*src[ctr] + src[is+sx*((js-1)+sy*ks)]) * inv_dy2;
            if (dim >= 3)
                val += (src[is+sx*(js+sy*(ks+1))] - 2.0*src[ctr] + src[is+sx*(js+sy*(ks-1))]) * inv_dz2;
            rhs[ctr] += c * val;
        }
    };

    return t;
}

Term grad(const ScalarField& f, int axis, double coeff) {
    if (axis < 0 || axis >= f.mesh.dim)
        throw std::invalid_argument("grad: axis out of range for this mesh dimension");

    Term t;
    t.type  = TermType::GRADIENT;
    t.field = &f;
    t.coeff = coeff;
    t.axis  = axis;

    int    nx = f.mesh.n[0], ny = f.mesh.n[1], nz = f.mesh.n[2];
    int    sx = f.storedDims[0], sy = f.storedDims[1];
    int    g  = f.ghost;
    double inv_2d = 0.5 / f.mesh.d[axis];

    const ScalarField* pf = &f;

    t.gpu_launcher = [pf, nx, ny, nz, sx, sy, g, axis, inv_2d]
                     (double* d_rhs, double c, ScratchPool&) {
        const double* d_src = pf->d_curr;
        if (!d_src)
            throw std::runtime_error("grad GPU: source field not on device");
        int total = nx * ny * nz;
        kernel_grad_accumulate<<<(total + 255) / 256, 256>>>(
            d_rhs, d_src, c, nx, ny, nz, sx, sy, g, axis, inv_2d);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("grad GPU kernel error: ") + cudaGetErrorString(err));
    };

    t.cpu_kernel = [pf, nx, ny, nz, sx, sy, g, axis, inv_2d]
                   (double* rhs, double c, ScratchPool&) {
        const double* src = pf->curr.data();
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int is = i+g, js = j+g, ks = k+g;
            int ctr = is + sx*(js + sy*ks);
            int fwd, bwd;
            if (axis == 0) {
                fwd = (is+1) + sx*(js + sy*ks);
                bwd = (is-1) + sx*(js + sy*ks);
            } else if (axis == 1) {
                fwd = is + sx*((js+1) + sy*ks);
                bwd = is + sx*((js-1) + sy*ks);
            } else {
                fwd = is + sx*(js + sy*(ks+1));
                bwd = is + sx*(js + sy*(ks-1));
            }
            rhs[ctr] += c * (src[fwd] - src[bwd]) * inv_2d;
        }
    };

    return t;
}

// ===========================================================================
// Equation
// ===========================================================================

Equation::Equation(ScalarField& unknown_, const std::string& name_)
    : name(name_), unknown(unknown_) {}

void Equation::setRHS(const RHSExpr& expr) {
    rhs_expr_ = expr;
}

void Equation::setRHS(const Term& t) {
    rhs_expr_ = RHSExpr(t);
}

void Equation::computeRHS(ScalarField& rhs) const {
    if (!rhs.deviceAllocated())
        throw std::runtime_error("Equation::computeRHS: rhs device memory not allocated");
    if (rhs_expr_.terms.empty())
        throw std::runtime_error("Equation::computeRHS: RHS not set (call setRHS first)");

    // Zero physical cells of rhs (not whole stored array — ghost stays 0)
    // For simplicity zero the whole stored array; cost is negligible.
    CUDA_CHECK(cudaMemset(rhs.d_curr, 0, rhs.storedSize * sizeof(double)));

    scratch_pool_.reset();
    for (const auto& term : rhs_expr_.terms) {
        if (!term.gpu_launcher)
            throw std::runtime_error(
                "Equation::computeRHS: a Term has no GPU launcher. "
                "Did you build it with a non-CUDA path?");
        term.gpu_launcher(rhs.d_curr, term.coeff, scratch_pool_);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Equation::computeRHSCPU(ScalarField& rhs) const {
    if (rhs_expr_.terms.empty())
        throw std::runtime_error("Equation::computeRHSCPU: RHS not set");

    std::fill(rhs.curr.begin(), rhs.curr.end(), 0.0);

    scratch_pool_.reset();
    for (const auto& term : rhs_expr_.terms) {
        if (!term.cpu_kernel)
            throw std::runtime_error(
                "Equation::computeRHSCPU: a Term has no CPU kernel.");
        term.cpu_kernel(rhs.curr.data(), term.coeff, scratch_pool_);
    }
}

// ===========================================================================
// Vector operator factories
// ===========================================================================

// lap(VectorField) — component-wise Laplacian
VectorRHSExpr lap(const VectorField& vf, double coeff) {
    VectorRHSExpr expr(vf.nComponents());
    for (int c = 0; c < vf.nComponents(); ++c)
        expr[c] = RHSExpr(lap(vf[c], coeff));
    return expr;
}

// grad(ScalarField) — returns mesh.dim-component gradient
VectorRHSExpr grad(const ScalarField& f, double coeff) {
    const int dim = f.mesh.dim;
    VectorRHSExpr expr(dim);
    for (int ax = 0; ax < dim; ++ax)
        expr[ax] = RHSExpr(grad(f, ax, coeff));
    return expr;
}

// div(VectorField) — scalar divergence
RHSExpr div(const VectorField& vf, double coeff) {
    if (vf.nComponents() < vf.mesh.dim)
        throw std::invalid_argument(
            "div: VectorField must have at least mesh.dim components");
    RHSExpr expr;
    for (int ax = 0; ax < vf.mesh.dim; ++ax)
        expr += grad(vf[ax], ax, coeff);
    return expr;
}

// curl(VectorField) — 3D only
VectorRHSExpr curl(const VectorField& vf, double coeff) {
    if (vf.mesh.dim != 3 || vf.nComponents() != 3)
        throw std::invalid_argument(
            "curl: VectorField must be 3-component on a 3D mesh");
    VectorRHSExpr expr(3);
    // curl[0] =  dv2/dy - dv1/dz
    expr[0] += grad(vf[2], 1,  coeff);
    expr[0] += grad(vf[1], 2, -coeff);
    // curl[1] =  dv0/dz - dv2/dx
    expr[1] += grad(vf[0], 2,  coeff);
    expr[1] += grad(vf[2], 0, -coeff);
    // curl[2] =  dv1/dx - dv0/dy
    expr[2] += grad(vf[1], 0,  coeff);
    expr[2] += grad(vf[0], 1, -coeff);
    return expr;
}

// ===========================================================================
// ScratchPool
// ===========================================================================

ScratchPool::~ScratchPool() {
    for (double* p : dev_bufs_) {
        if (p) cudaFree(p);
    }
}

double* ScratchPool::acquireDevice(std::size_t size) {
    if (next_dev_ < dev_bufs_.size()) {
        if (dev_sizes_[next_dev_] < size) {
            cudaFree(dev_bufs_[next_dev_]);
            CUDA_CHECK(cudaMalloc(&dev_bufs_[next_dev_], size * sizeof(double)));
            dev_sizes_[next_dev_] = size;
        }
        return dev_bufs_[next_dev_++];
    }
    double* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, size * sizeof(double)));
    dev_bufs_.push_back(p);
    dev_sizes_.push_back(size);
    ++next_dev_;
    return p;
}

double* ScratchPool::acquireHost(std::size_t size) {
    if (next_host_ < host_bufs_.size()) {
        if (host_bufs_[next_host_].size() < size)
            host_bufs_[next_host_].assign(size, 0.0);
        return host_bufs_[next_host_++].data();
    }
    host_bufs_.emplace_back(size, 0.0);
    ++next_host_;
    return host_bufs_.back().data();
}

// ===========================================================================
// kernel_mul_accumulate  --  rhs[idx] += coeff * s1[idx] * s2[idx]
// (physical cells only)
// ===========================================================================

__global__ void kernel_mul_accumulate(
        double*       rhs,
        const double* s1,
        const double* s2,
        double        coeff,
        int nx, int ny, int nz,
        int sx, int sy,
        int ghost)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny * nz) return;

    int i = tid % nx;
    int j = (tid / nx) % ny;
    int k = tid / (nx * ny);

    int idx = (i + ghost) + sx * ((j + ghost) + sy * (k + ghost));
    rhs[idx] += coeff * s1[idx] * s2[idx];
}

// ===========================================================================
// Composite expressions on the right-hand side
// ===========================================================================
//
// The two helpers below materialise a Term / RHSExpr into a scratch buffer
// (device or host).  They are used by:
//   * Term * Term / Term * RHSExpr / RHSExpr * RHSExpr  (Phase 2)
//   * lap(expr, bcs) / grad(expr, ax, bcs)              (Phase 3)
//
// For GPU evaluation the launchers accept a ScratchPool& and obtain device
// buffers via pool.acquireDevice(storedSize).  Buffers are zeroed before
// each child launcher is invoked so accumulation semantics match a fresh
// rhs.

namespace detail {

// Pick a representative source field from a Term / RHSExpr (for layout info
// and for rhs nullptr checks).  Throws if the expression has no field.
const ScalarField* repField(const Term& t) {
    if (!t.field)
        throw std::runtime_error(
            "Composite Term: no representative source field captured");
    return t.field;
}
const ScalarField* repField(const RHSExpr& e) {
    for (const auto& t : e.terms)
        if (t.field) return t.field;
    throw std::runtime_error(
        "Composite RHSExpr: no representative source field captured");
}

// Materialise an RHSExpr into d_buf on GPU.  d_buf must have been allocated
// to at least storedSize doubles; it is zeroed first.
void materialiseGPU(const RHSExpr& expr,
                    double* d_buf, std::size_t storedSize,
                    ScratchPool& pool)
{
    CUDA_CHECK(cudaMemset(d_buf, 0, storedSize * sizeof(double)));
    for (const auto& t : expr.terms) {
        if (!t.gpu_launcher)
            throw std::runtime_error(
                "Composite GPU: a Term has no GPU launcher");
        t.gpu_launcher(d_buf, t.coeff, pool);
    }
}
void materialiseGPU(const Term& t,
                    double* d_buf, std::size_t storedSize,
                    ScratchPool& pool)
{
    CUDA_CHECK(cudaMemset(d_buf, 0, storedSize * sizeof(double)));
    if (!t.gpu_launcher)
        throw std::runtime_error("Composite GPU: Term has no GPU launcher");
    t.gpu_launcher(d_buf, t.coeff, pool);
}

void materialiseCPU(const RHSExpr& expr,
                    double* h_buf, std::size_t storedSize,
                    ScratchPool& pool)
{
    std::fill(h_buf, h_buf + storedSize, 0.0);
    for (const auto& t : expr.terms) {
        if (!t.cpu_kernel)
            throw std::runtime_error(
                "Composite CPU: a Term has no CPU kernel");
        t.cpu_kernel(h_buf, t.coeff, pool);
    }
}
void materialiseCPU(const Term& t,
                    double* h_buf, std::size_t storedSize,
                    ScratchPool& pool)
{
    std::fill(h_buf, h_buf + storedSize, 0.0);
    if (!t.cpu_kernel)
        throw std::runtime_error("Composite CPU: Term has no CPU kernel");
    t.cpu_kernel(h_buf, t.coeff, pool);
}

} // namespace detail

// ===========================================================================
// Term * Term / Term * ScalarField / RHSExpr * RHSExpr ...  -- Phase 2
// ===========================================================================
//
// Implemented in include/equation/FieldOps.inl as inline functions, but the
// underlying mul_accumulate launcher helper lives here so that the kernel
// symbol is emitted in this translation unit.

namespace detail {

// Launch mul_accumulate kernel.  Public to FieldOps.inl via detail::.
void mulAccumulateGPU(double* d_rhs,
                      const double* d_s1, const double* d_s2,
                      double coeff,
                      int nx, int ny, int nz,
                      int sx, int sy, int g)
{
    int total = nx * ny * nz;
    int threads = 256;
    int blocks  = (total + threads - 1) / threads;
    kernel_mul_accumulate<<<blocks, threads>>>(
        d_rhs, d_s1, d_s2, coeff, nx, ny, nz, sx, sy, g);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("mul_accumulate GPU error: ") + cudaGetErrorString(err));
}

void mulAccumulateCPU(double* rhs,
                      const double* s1, const double* s2,
                      double coeff,
                      int nx, int ny, int nz,
                      int sx, int sy, int g)
{
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        int idx = (i+g) + sx*((j+g) + sy*(k+g));
        rhs[idx] += coeff * s1[idx] * s2[idx];
    }
}

} // namespace detail

// ===========================================================================
// lap(Term/RHSExpr, bcs) and grad(...) on composite expressions  -- Phase 3
// ===========================================================================

// Helper: build a Term that materialises `src_expr` into a scratch buffer,
// applies BCs, then runs `op` (lap or grad) on the scratch buffer.  Generic
// over the source-expression type (Term / RHSExpr) and the FD operator.

template<typename SrcExpr>
static Term makeStencilOnExprTerm(
        SrcExpr             src_expr,
        const ScalarField&  layout,                                // for mesh/ghost/storedSize
        std::vector<BoundaryCondition*> bcs,
        std::function<void(double* /*rhs*/, const double* /*src*/,
                           double /*coeff*/)> gpu_op,
        std::function<void(double* /*rhs*/, const double* /*src*/,
                           double /*coeff*/)> cpu_op,
        TermType  out_type,
        int       out_axis,
        double    coeff)
{
    Term out;
    out.type  = out_type;
    out.axis  = out_axis;
    out.coeff = coeff;
    out.field = &layout;     // representative; ensures rhs sanity in nesting

    // Capture by value for thread-safety of the std::function.
    const Mesh*  pmesh      = &layout.mesh;
    int          ghost      = layout.ghost;
    std::size_t  storedSize = layout.storedSize;

    out.gpu_launcher = [src_expr, pmesh, ghost, storedSize, bcs,
                        gpu_op]
                       (double* d_rhs, double c, ScratchPool& pool) {
        // 1. Allocate / reuse scratch
        double* d_scratch = pool.acquireDevice(storedSize);

        // 2. Evaluate src_expr into d_scratch (zeros first)
        detail::materialiseGPU(src_expr, d_scratch, storedSize, pool);

        // 3. Apply BCs on the shell view of d_scratch
        ScalarField shell = ScalarField::makeShell(*pmesh, ghost, d_scratch);
        for (auto* bc : bcs) bc->applyOnGPU(shell);

        // 4. Run the FD operator: rhs += c * op(scratch)
        gpu_op(d_rhs, d_scratch, c);
    };

    out.cpu_kernel = [src_expr, pmesh, ghost, storedSize, bcs,
                      cpu_op]
                     (double* rhs, double c, ScratchPool& pool) {
        double* h_scratch = pool.acquireHost(storedSize);
        detail::materialiseCPU(src_expr, h_scratch, storedSize, pool);

        // CPU BCs require a CPU-resident ScalarField; build one and copy in.
        ScalarField tmp(*pmesh, "shell_cpu", ghost);
        std::copy(h_scratch, h_scratch + storedSize, tmp.curr.begin());
        for (auto* bc : bcs) bc->applyOnCPU(tmp);
        std::copy(tmp.curr.begin(), tmp.curr.end(), h_scratch);

        cpu_op(rhs, h_scratch, c);
    };

    return out;
}

// --- lap(Term, bcs) / lap(RHSExpr, bcs) -------------------------------------

template<typename SrcExpr>
static Term lapOnExpr(SrcExpr src_expr, const ScalarField& layout,
                      const std::vector<BoundaryCondition*>& bcs,
                      double coeff)
{
    int    nx = layout.mesh.n[0], ny = layout.mesh.n[1], nz = layout.mesh.n[2];
    int    sx = layout.storedDims[0], sy = layout.storedDims[1];
    int    g  = layout.ghost;
    int    dim = layout.mesh.dim;
    double inv_dx2 = 1.0 / (layout.mesh.d[0] * layout.mesh.d[0]);
    double inv_dy2 = (dim >= 2) ? 1.0 / (layout.mesh.d[1] * layout.mesh.d[1]) : 0.0;
    double inv_dz2 = (dim >= 3) ? 1.0 / (layout.mesh.d[2] * layout.mesh.d[2]) : 0.0;

    auto gpu_op = [nx, ny, nz, sx, sy, g, dim, inv_dx2, inv_dy2, inv_dz2]
                  (double* d_rhs, const double* d_src, double c) {
        int total = nx * ny * nz;
        kernel_lap_accumulate<<<(total + 255) / 256, 256>>>(
            d_rhs, d_src, c, nx, ny, nz, sx, sy, g, dim,
            inv_dx2, inv_dy2, inv_dz2);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("lap(expr) GPU error: ") + cudaGetErrorString(err));
    };

    auto cpu_op = [nx, ny, nz, sx, sy, g, dim, inv_dx2, inv_dy2, inv_dz2]
                  (double* rhs, const double* src, double c) {
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int is = i+g, js = j+g, ks = k+g;
            int ctr = is + sx*(js + sy*ks);
            double val =
                (src[(is+1)+sx*(js+sy*ks)] - 2.0*src[ctr] + src[(is-1)+sx*(js+sy*ks)]) * inv_dx2;
            if (dim >= 2)
                val += (src[is+sx*((js+1)+sy*ks)] - 2.0*src[ctr] + src[is+sx*((js-1)+sy*ks)]) * inv_dy2;
            if (dim >= 3)
                val += (src[is+sx*(js+sy*(ks+1))] - 2.0*src[ctr] + src[is+sx*(js+sy*(ks-1))]) * inv_dz2;
            rhs[ctr] += c * val;
        }
    };

    return makeStencilOnExprTerm(std::move(src_expr), layout, bcs,
                                 gpu_op, cpu_op,
                                 TermType::COMPOSITE, 0, coeff);
}

// TODO(vector): add VectorRHSExpr lap(VectorRHSExpr, bcs) overload.
Term lap(const Term& t, const std::vector<BoundaryCondition*>& bcs, double coeff) {
    return lapOnExpr<Term>(t, *detail::repField(t), bcs, coeff);
}
Term lap(const RHSExpr& e, const std::vector<BoundaryCondition*>& bcs, double coeff) {
    return lapOnExpr<RHSExpr>(e, *detail::repField(e), bcs, coeff);
}

// --- grad(Term, axis, bcs) / grad(RHSExpr, axis, bcs) -----------------------

template<typename SrcExpr>
static Term gradOnExpr(SrcExpr src_expr, const ScalarField& layout, int axis,
                       const std::vector<BoundaryCondition*>& bcs,
                       double coeff)
{
    if (axis < 0 || axis >= layout.mesh.dim)
        throw std::invalid_argument("grad(expr): axis out of range");

    int    nx = layout.mesh.n[0], ny = layout.mesh.n[1], nz = layout.mesh.n[2];
    int    sx = layout.storedDims[0], sy = layout.storedDims[1];
    int    g  = layout.ghost;
    double inv_2d = 0.5 / layout.mesh.d[axis];

    auto gpu_op = [nx, ny, nz, sx, sy, g, axis, inv_2d]
                  (double* d_rhs, const double* d_src, double c) {
        int total = nx * ny * nz;
        kernel_grad_accumulate<<<(total + 255) / 256, 256>>>(
            d_rhs, d_src, c, nx, ny, nz, sx, sy, g, axis, inv_2d);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("grad(expr) GPU error: ") + cudaGetErrorString(err));
    };

    auto cpu_op = [nx, ny, nz, sx, sy, g, axis, inv_2d]
                  (double* rhs, const double* src, double c) {
        for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            int is = i+g, js = j+g, ks = k+g;
            int ctr = is + sx*(js + sy*ks);
            int fwd, bwd;
            if (axis == 0) {
                fwd = (is+1) + sx*(js + sy*ks);
                bwd = (is-1) + sx*(js + sy*ks);
            } else if (axis == 1) {
                fwd = is + sx*((js+1) + sy*ks);
                bwd = is + sx*((js-1) + sy*ks);
            } else {
                fwd = is + sx*(js + sy*(ks+1));
                bwd = is + sx*(js + sy*(ks-1));
            }
            rhs[ctr] += c * (src[fwd] - src[bwd]) * inv_2d;
        }
    };

    return makeStencilOnExprTerm(std::move(src_expr), layout, bcs,
                                 gpu_op, cpu_op,
                                 TermType::COMPOSITE, axis, coeff);
}

Term grad(const Term& t, int axis,
          const std::vector<BoundaryCondition*>& bcs, double coeff) {
    return gradOnExpr<Term>(t, *detail::repField(t), axis, bcs, coeff);
}
Term grad(const RHSExpr& e, int axis,
          const std::vector<BoundaryCondition*>& bcs, double coeff) {
    return gradOnExpr<RHSExpr>(e, *detail::repField(e), axis, bcs, coeff);
}

// grad(expr, bcs)  ->  VectorRHSExpr (one component per mesh axis)
// TODO(vector): mirror this for grad(VectorRHSExpr, bcs) returning a
// rank-2 tensor expression once vector solvers need it.
VectorRHSExpr grad(const Term& t,
                   const std::vector<BoundaryCondition*>& bcs, double coeff) {
    const int dim = detail::repField(t)->mesh.dim;
    VectorRHSExpr expr(dim);
    for (int ax = 0; ax < dim; ++ax)
        expr[ax] = RHSExpr(grad(t, ax, bcs, coeff));
    return expr;
}
VectorRHSExpr grad(const RHSExpr& e,
                   const std::vector<BoundaryCondition*>& bcs, double coeff) {
    const int dim = detail::repField(e)->mesh.dim;
    VectorRHSExpr expr(dim);
    for (int ax = 0; ax < dim; ++ax)
        expr[ax] = RHSExpr(grad(e, ax, bcs, coeff));
    return expr;
}

// div(VectorRHSExpr, bcs) — divergence of expression-valued flux.
RHSExpr div(const VectorRHSExpr& v,
            const std::vector<BoundaryCondition*>& bcs, double coeff) {
    RHSExpr expr;
    const int n = v.nComponents();
    for (int ax = 0; ax < n; ++ax)
        expr += grad(v[ax], ax, bcs, coeff);
    return expr;
}

} // namespace PhiX
