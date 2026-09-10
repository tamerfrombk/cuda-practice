#include "cuda-utilities.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

// An inclusive scan (prefix sum) turns X[0..N) into
// Y[i] = X[0] + X[1] + ... + X[i]
// (an *exclusive* scan would instead give Y[i] = X[0] + ... + X[i-1], with
// Y[0] = 0). Every kernel in this file computes the inclusive version.
//
// Kogge-Stone is a parallel scan pattern originally designed as a hardware
// carry-lookahead adder circuit, adapted here to software. With N threads,
// one per element, it runs log2(N) rounds. In round `stride` (1, 2, 4, 8,
// ...), every thread whose index i >= stride adds the element `stride`
// positions behind it into its own slot. After log2(N) rounds, slot i holds
// the sum of the first i+1 elements. Unlike reduction's tree, every thread
// stays active for its own output slot the whole time -- there's no thread
// count halving each round, just a growing "reach" per thread.
//
// This file builds the technique up in stages, exactly like reduction.cu
// does for sum-reduction:
//   1. simple_kogge_stone_scan_kernel: correct, single-block, single shared
//      buffer -- needs two __syncthreads() per round to avoid a
//      read-after-write hazard.
//   2. double_buffered_kogge_stone_scan_kernel: same algorithm, two shared
//      buffers ping-ponged, so the hazard can't happen -- one
//      __syncthreads() per round.
//   3. kogge_stone_scan_warp_primitives_kernel: replace the shared-memory
//      exchange with __shfl_up_sync inside each warp (no shared memory, no
//      barriers, for 5 of the rounds), then combine the (few) warp results
//      with a small shared-memory step -- two barriers total, regardless of
//      block size.
//   4. general_scan_local_kernel / add_block_offsets_kernel: generalize
//      beyond one block with a local-scan + block-sums + add-offsets
//      pipeline, since __syncthreads() can't coordinate across blocks.
//   5. coarsened_scan_local_kernel: thread coarsening + register tiling to
//      fix Kogge-Stone's fundamental work-inefficiency (see the comment
//      above that kernel).

static constexpr int BLOCK_DIM = 128;

// Compares a kernel's output against the closed-form answer for input
// X[i] = i + 1 (i.e. Y[i] should be (i+1)(i+2)/2) and prints a PASS/FAIL
// summary line.
inline bool verify_inclusive_scan(const char *label, const float *c, int n) {
  bool ok = true;
  float max_rel_err = 0.0f;
  for (int i = 0; i < n; i++) {
    double expected = (double)(i + 1) * (i + 2) / 2.0;
    float rel_err = std::fabs((double)c[i] - expected) / expected;
    max_rel_err = std::max(max_rel_err, rel_err);
    // Floats accumulate rounding error over thousands of additions, so allow
    // a little slack rather than requiring an exact match.
    if (rel_err > 1e-3f) {
      ok = false;
    }
  }
  double want_last = (double)n * (n + 1) / 2.0;
  printf("%-40s N=%-6d c[0]=%.0f c[N/2]=%.0f c[N-1]=%.0f (want %.0f) "
         "max_rel_err=%.2e -> %s\n",
         label, n, c[0], c[n / 2], c[n - 1], want_last, max_rel_err,
         ok ? "PASS" : "FAIL");
  return ok;
}

// Precondition: single block, N == blockDim.x (one thread per element).
//
// This kernel works out of ONE shared buffer, which creates a hazard: on
// round `stride`, thread i reads XY[i - stride] while thread (i - stride) is
// *simultaneously* trying to overwrite XY[i - stride] with its own update for
// the very same round. If thread (i - stride) wins that race, thread i reads
// the *new* value instead of the value that existed at the start of the
// round, which corrupts the scan (effectively double-adding).
//
// We avoid that with two barriers per round: the first barrier ensures every
// thread has finished computing this round's value into a private register
// `temp` before the second barrier lets anyone write it back to shared
// memory. That turns "everyone reads, then everyone writes" into two
// block-wide phases that can't interleave, at the cost of two
// __syncthreads() per round. The next kernel shows how to get down to one.
__global__ void simple_kogge_stone_scan_kernel(float *input, float *,
                                               float *output, int N) {
  assert(BLOCK_DIM == blockDim.x);
  __shared__ float XY[BLOCK_DIM];

  int i = threadIdx.x;
  XY[i] = input[i];

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads();
    float temp = XY[i];
    if (i >= stride) {
      temp += XY[i - stride];
    }
    __syncthreads();
    XY[i] = temp;
  }

  output[i] = XY[i];
}

inline void run_simple_kogge_stone_scan_kernel(int n, dim3 threadsPerBlock,
                                               dim3 blocksPerGrid) {
  cuda_context ctx;
  auto hm = ctx.allocate_host_memory(n);
  for (int i = 0; i < n; i++) {
    hm.a[i] = i + 1;
    hm.c[i] = 0;
  }

  auto dm = ctx.upload_inputs_to_device(hm);
  RUN_KERNEL(simple_kogge_stone_scan_kernel, dm.a, dm.b, dm.c, n,
             blocksPerGrid, threadsPerBlock);
  ctx.download_result_to_host(dm, hm);

  verify_inclusive_scan("simple_kogge_stone_scan_kernel", hm.c, n);
}

// Precondition: single block, N == blockDim.x.
//
// Same recurrence as simple_kogge_stone_scan_kernel, but split the shared
// buffer into two: buf[0] and buf[1]. Each round reads exclusively from one
// buffer and writes exclusively to the other, then the two swap roles for
// the next round. Because the buffer being written is never the buffer being
// read *within the same round*, the read-after-write hazard above simply
// cannot occur, no matter what order threads run in. That means only ONE
// __syncthreads() per round is needed (to make sure the buffer just written
// is fully visible before it becomes next round's read buffer) -- half of
// what the single-buffer version needs. The cost is double the shared memory
// footprint.
__global__ void double_buffered_kogge_stone_scan_kernel(float *input, float *,
                                                         float *output,
                                                         int N) {
  assert(BLOCK_DIM == blockDim.x);
  __shared__ float buf[2][BLOCK_DIM];

  int i = threadIdx.x;
  int in_buf = 0, out_buf = 1;
  buf[in_buf][i] = input[i];
  __syncthreads();

  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    float val = buf[in_buf][i];
    if (i >= stride) {
      val += buf[in_buf][i - stride];
    }
    buf[out_buf][i] = val;
    __syncthreads();

    int swap = in_buf;
    in_buf = out_buf;
    out_buf = swap;
  }

  output[i] = buf[in_buf][i];
}

inline void run_double_buffered_kogge_stone_scan_kernel(int n,
                                                         dim3 threadsPerBlock,
                                                         dim3 blocksPerGrid) {
  cuda_context ctx;
  auto hm = ctx.allocate_host_memory(n);
  for (int i = 0; i < n; i++) {
    hm.a[i] = i + 1;
    hm.c[i] = 0;
  }

  auto dm = ctx.upload_inputs_to_device(hm);
  RUN_KERNEL(double_buffered_kogge_stone_scan_kernel, dm.a, dm.b, dm.c, n,
             blocksPerGrid, threadsPerBlock);
  ctx.download_result_to_host(dm, hm);

  verify_inclusive_scan("double_buffered_kogge_stone_scan_kernel", hm.c, n);
}

// Return the index of the warp given the thread id
__device__ int warpIdx() { return threadIdx.x / warpSize; }

// Return the position of the thread within a warp
__device__ int laneIdx() { return threadIdx.x % warpSize; }

// Inclusive scan of `val` across the 32 lanes of one warp using
// __shfl_up_sync. __shfl_up_sync(mask, var, delta) lets a lane read `var`
// directly out of the register of the lane `delta` positions below it -- no
// shared memory, no explicit __syncthreads(), because all lanes of a warp
// execute the shuffle together as a single instruction. A lane with
// lane < delta has no such neighbor; the hardware just hands it back its own
// value, which is why the `lane >= stride` guard exists (otherwise we'd add
// our own value to itself). This is the exact same doubling-stride
// recurrence as the shared-memory kernels above, just running register-to-
// register instead of through a shared array.
__device__ float warp_inclusive_scan(float val) {
  int lane = laneIdx();
  for (int stride = 1; stride < warpSize; stride *= 2) {
    float n = __shfl_up_sync(0xFFFFFFFF, val, stride);
    if (lane >= stride) {
      val += n;
    }
  }
  return val;
}

// Inclusive scan of `val` across an entire BLOCK_DIM-thread block
// (BLOCK_DIM must be a multiple of warpSize). Three phases:
//  1. Every warp independently scans its own 32 values with
//     warp_inclusive_scan -- no shared memory, no barriers, and every warp in
//     the block runs this concurrently.
//  2. Each warp's *last* lane now holds that warp's total. Stash the
//     BLOCK_DIM/32 warp totals into shared memory and have warp 0 scan that
//     tiny array with the same warp_inclusive_scan primitive, so
//     warp_totals[w] ends up holding the sum of every warp <= w.
//  3. Every thread adds the exclusive prefix of its own warp (the inclusive
//     total of the *previous* warp) onto its phase-1 result.
// Total synchronization cost: two __syncthreads(), regardless of BLOCK_DIM --
// compare to O(log2(BLOCK_DIM)) barriers for the shared-memory kernels above.
__device__ float block_inclusive_scan(float val) {
  int lane = laneIdx();
  int warp = warpIdx();

  float warp_scanned = warp_inclusive_scan(val);

  __shared__ float warp_totals[BLOCK_DIM / 32];
  if (lane == warpSize - 1) {
    warp_totals[warp] = warp_scanned;
  }
  __syncthreads();

  if (warp == 0) {
    float t = lane < BLOCK_DIM / warpSize ? warp_totals[lane] : 0.0f;
    float scanned = warp_inclusive_scan(t);
    if (lane < BLOCK_DIM / warpSize) {
      warp_totals[lane] = scanned;
    }
  }
  __syncthreads();

  float exclusive_prefix = warp == 0 ? 0.0f : warp_totals[warp - 1];
  return warp_scanned + exclusive_prefix;
}

// Precondition: single block, N <= blockDim.x == BLOCK_DIM.
//
// The whole single-block scan is now just "load, block_inclusive_scan,
// store" -- all the round-by-round buffering above collapses into the warp
// shuffles and the two barriers inside block_inclusive_scan. N is allowed to
// be less than BLOCK_DIM (with out-of-range lanes contributing 0) because
// this kernel is reused below to scan the small `block_sums` array in the
// general multi-block pipeline.
__global__ void kogge_stone_scan_warp_primitives_kernel(float *input, float *,
                                                         float *output,
                                                         int N) {
  assert(BLOCK_DIM == blockDim.x);
  assert(warpSize == 32);

  int i = threadIdx.x;
  float val = i < N ? input[i] : 0.0f;
  float scanned = block_inclusive_scan(val);
  if (i < N) {
    output[i] = scanned;
  }
}

inline void run_kogge_stone_scan_warp_primitives_kernel(int n,
                                                         dim3 threadsPerBlock,
                                                         dim3 blocksPerGrid) {
  cuda_context ctx;
  auto hm = ctx.allocate_host_memory(n);
  for (int i = 0; i < n; i++) {
    hm.a[i] = i + 1;
    hm.c[i] = 0;
  }

  auto dm = ctx.upload_inputs_to_device(hm);
  RUN_KERNEL(kogge_stone_scan_warp_primitives_kernel, dm.a, dm.b, dm.c, n,
             blocksPerGrid, threadsPerBlock);
  ctx.download_result_to_host(dm, hm);

  verify_inclusive_scan("kogge_stone_scan_warp_primitives_kernel", hm.c, n);
}

// __syncthreads() only coordinates threads within one block, so none of the
// kernels above can scan an input that spans more than one block. The
// standard fix is a three-phase pipeline:
//   Phase A (general_scan_local_kernel): each block independently scans its
//     own BLOCK_DIM-sized segment with block_inclusive_scan and records its
//     segment's total into block_sums[blockIdx.x].
//   Phase B: scan block_sums itself. It only has `numBlocks` entries, so as
//     long as numBlocks <= BLOCK_DIM this is just one more launch of
//     kogge_stone_scan_warp_primitives_kernel<<<1, BLOCK_DIM>>>, in place.
//     (For inputs so large that numBlocks > BLOCK_DIM, this phase would
//     itself need to be split across blocks, recursively -- not needed at
//     the sizes exercised here.)
//   Phase C (add_block_offsets_kernel): add each block's *exclusive* prefix
//     -- the scanned total of every block before it, i.e.
//     block_sums[blockIdx.x - 1] -- onto every element that block produced
//     in phase A.
__global__ void general_scan_local_kernel(float *input, float *block_sums,
                                          float *output, int N) {
  assert(BLOCK_DIM == blockDim.x);
  assert(warpSize == 32);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float val = i < N ? input[i] : 0.0f;
  float scanned = block_inclusive_scan(val);
  if (i < N) {
    output[i] = scanned;
  }

  // block_inclusive_scan has already folded every element of the block in by
  // the time the last thread's result is ready, so the last thread's value
  // IS this block's total -- correct even for a block that only partially
  // overlaps N, since the out-of-range lanes contributed 0.
  if (threadIdx.x == blockDim.x - 1) {
    block_sums[blockIdx.x] = scanned;
  }
}

// Phase C: block 0 has no predecessor, so it's already correct and is left
// untouched.
__global__ void add_block_offsets_kernel(float *output, float *block_sums,
                                         float *, int N) {
  if (blockIdx.x == 0) {
    return;
  }
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    output[i] += block_sums[blockIdx.x - 1];
  }
}

inline void run_general_scan(int n, dim3 threadsPerBlock,
                             dim3 blocksPerGrid) {
  cuda_context ctx;
  auto hm = ctx.allocate_host_memory(n);
  for (int i = 0; i < n; i++) {
    hm.a[i] = i + 1;
    hm.c[i] = 0;
  }

  auto dm = ctx.upload_inputs_to_device(hm);

  // Phase A: local per-block scans + block sums (dm.b doubles as scratch).
  RUN_KERNEL(general_scan_local_kernel, dm.a, dm.b, dm.c, n, blocksPerGrid,
             threadsPerBlock);

  // Phase B: scan block_sums itself, in place.
  assert((int)blocksPerGrid.x <= BLOCK_DIM);
  RUN_KERNEL(kogge_stone_scan_warp_primitives_kernel, dm.b, dm.b, dm.b,
             (int)blocksPerGrid.x, dim3(1), dim3(BLOCK_DIM));

  // Phase C: fold each block's exclusive prefix into its output elements.
  RUN_KERNEL(add_block_offsets_kernel, dm.c, dm.b, dm.a, n, blocksPerGrid,
             threadsPerBlock);

  ctx.download_result_to_host(dm, hm);

  verify_inclusive_scan("run_general_scan", hm.c, n);
}

// Every Kogge-Stone kernel above does, per block, sum_{stride=1,2,4,...}(N -
// stride) additions -- O(N log2 N) total. A plain sequential scan does
// exactly N - 1 additions -- O(N). So Kogge-Stone trades *more total work*
// for a *much shorter critical path* (O(log N) dependent steps instead of
// O(N)): it is step-efficient but NOT work-efficient. That's a good trade
// when there are far more ALUs than N, but a GPU only has so many; once N is
// large enough to keep every ALU busy, the extra O(N log N - N) additions are
// pure waste -- wasted throughput, wasted energy, for no shorter wall-clock
// time.
//
// Thread coarsening fixes this by shrinking the problem Kogge-Stone actually
// has to solve. Instead of 1 element per thread, each thread owns
// COARSE_FACTOR contiguous elements:
//   1. Sequentially scan those COARSE_FACTOR elements in a private register
//      array (register tiling -- see below). This is O(N) work in total
//      across all threads (COARSE_FACTOR - 1 adds each, N/COARSE_FACTOR
//      threads) -- exactly as work-efficient as the sequential algorithm --
//      and touches no shared memory at all.
//   2. Only the N/COARSE_FACTOR per-thread totals (the last register of each
//      thread's local scan) go through block_inclusive_scan. The expensive
//      O((N/C) log(N/C)) part is now working on a C-times-smaller problem.
//   3. Add the exclusive prefix that step 2 hands back onto every one of a
//      thread's COARSE_FACTOR register values before writing them out.
// Total work becomes O(N) + O((N/C) log(N/C)), which converges toward the
// sequential algorithm's O(N) as C grows, while the critical path stays a
// short O(C + log(N/C)) -- still far shorter than a fully sequential scan.
//
// Register tiling is the "keep it in registers" half of that: each thread's
// COARSE_FACTOR-sized chunk is private -- no other thread ever needs to read
// it -- so there is no reason to stage it through shared memory. A small
// fixed-size local array (float reg[COARSE_FACTOR]) is compiled straight
// into registers, so the sequential scan in step 1 never pays shared
// memory's extra address-generation/bank-conflict latency; only the single
// per-thread total in step 2 ever needs to leave the thread's own registers.
static constexpr int COARSE_FACTOR = 4;

__global__ void coarsened_scan_local_kernel(float *input, float *block_sums,
                                            float *output, int N) {
  assert(BLOCK_DIM == blockDim.x);
  assert(warpSize == 32);

  int base = (blockIdx.x * blockDim.x + threadIdx.x) * COARSE_FACTOR;

  // Step 1: sequential inclusive scan of this thread's chunk, entirely in
  // registers.
  float reg[COARSE_FACTOR];
  for (int c = 0; c < COARSE_FACTOR; c++) {
    int idx = base + c;
    float v = idx < N ? input[idx] : 0.0f;
    reg[c] = (c == 0 ? 0.0f : reg[c - 1]) + v;
  }

  // Step 2: scan just the per-thread totals across the block, then convert
  // the inclusive result back to an exclusive prefix by subtracting our own
  // contribution.
  float thread_total = reg[COARSE_FACTOR - 1];
  float exclusive_prefix = block_inclusive_scan(thread_total) - thread_total;

  // Step 3: fold that prefix into every register value and write it out.
  for (int c = 0; c < COARSE_FACTOR; c++) {
    int idx = base + c;
    if (idx < N) {
      output[idx] = reg[c] + exclusive_prefix;
    }
  }

  if (threadIdx.x == blockDim.x - 1) {
    block_sums[blockIdx.x] = thread_total + exclusive_prefix;
  }
}

// Same role as add_block_offsets_kernel, but walks each thread's
// COARSE_FACTOR-element chunk instead of a single element.
__global__ void add_block_offsets_coarsened_kernel(float *output,
                                                    float *block_sums, float *,
                                                    int N) {
  if (blockIdx.x == 0) {
    return;
  }
  float offset = block_sums[blockIdx.x - 1];
  int base = (blockIdx.x * blockDim.x + threadIdx.x) * COARSE_FACTOR;
  for (int c = 0; c < COARSE_FACTOR; c++) {
    int idx = base + c;
    if (idx < N) {
      output[idx] += offset;
    }
  }
}

inline void run_general_coarsened_scan(int n, dim3 threadsPerBlock,
                                       dim3 blocksPerGrid) {
  cuda_context ctx;
  auto hm = ctx.allocate_host_memory(n);
  for (int i = 0; i < n; i++) {
    hm.a[i] = i + 1;
    hm.c[i] = 0;
  }

  auto dm = ctx.upload_inputs_to_device(hm);

  RUN_KERNEL(coarsened_scan_local_kernel, dm.a, dm.b, dm.c, n, blocksPerGrid,
             threadsPerBlock);

  assert((int)blocksPerGrid.x <= BLOCK_DIM);
  RUN_KERNEL(kogge_stone_scan_warp_primitives_kernel, dm.b, dm.b, dm.b,
             (int)blocksPerGrid.x, dim3(1), dim3(BLOCK_DIM));

  RUN_KERNEL(add_block_offsets_coarsened_kernel, dm.c, dm.b, dm.a, n,
             blocksPerGrid, threadsPerBlock);

  ctx.download_result_to_host(dm, hm);

  verify_inclusive_scan("run_general_coarsened_scan", hm.c, n);
}

int main() {
  print_cuda_properties();
  {
    int n = BLOCK_DIM; // fits in exactly one block
    dim3 threadsPerBlock(n);
    dim3 blocksPerGrid(1);
    run_simple_kogge_stone_scan_kernel(n, threadsPerBlock, blocksPerGrid);
  }
  {
    int n = BLOCK_DIM;
    dim3 threadsPerBlock(n);
    dim3 blocksPerGrid(1);
    run_double_buffered_kogge_stone_scan_kernel(n, threadsPerBlock,
                                                blocksPerGrid);
  }
  {
    int n = BLOCK_DIM;
    dim3 threadsPerBlock(n);
    dim3 blocksPerGrid(1);
    run_kogge_stone_scan_warp_primitives_kernel(n, threadsPerBlock,
                                                blocksPerGrid);
  }
  {
    int n = 2048; // purposely more data than fits in one block
    dim3 threadsPerBlock(BLOCK_DIM);
    dim3 blocksPerGrid(ceildiv(n, BLOCK_DIM));
    run_general_scan(n, threadsPerBlock, blocksPerGrid);
  }
  {
    int n = 8192; // large enough that work-efficiency actually matters
    dim3 threadsPerBlock(BLOCK_DIM);
    dim3 blocksPerGrid(ceildiv(n, BLOCK_DIM * COARSE_FACTOR));
    run_general_coarsened_scan(n, threadsPerBlock, blocksPerGrid);
  }
}
