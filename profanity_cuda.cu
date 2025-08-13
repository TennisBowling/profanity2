// CUDA host runtime for profanity2 (vanity generator) – multi-GPU, secure seeding

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <iomanip>

#include "ArgParser.hpp"
#include "Mode.hpp"
#include "help.hpp"
#include "precomp.hpp"
#include "types.hpp"
#include "SpeedSample.hpp"

// Kernel prototypes (defined in cuda_kernels.cu)
extern "C" {
__global__ void profanity_init(const point *precomp, mp_number *pDeltaX, mp_number *pPrevLambda, result *pResult, ulonglong4 seed, ulonglong4 seedX, ulonglong4 seedY, uint32_t sizeTotal, uint32_t inverseSize);
__global__ void profanity_inverse(const mp_number *pDeltaX, mp_number *pInverse, uint32_t sizeTotal, uint32_t inverseSize);
__global__ void profanity_iterate(mp_number *pDeltaX, mp_number *pInverse, mp_number *pPrevLambda, uint32_t sizeTotal);
__global__ void profanity_transform_contract(mp_number *pInverse, uint32_t sizeTotal);
__global__ void profanity_score_benchmark(mp_number *pInverse, result *pResult, const uint8_t *data1, const uint8_t *data2, uint8_t scoreMax, uint32_t sizeTotal);
__global__ void profanity_score_matching(mp_number *pInverse, result *pResult, const uint8_t *data1, const uint8_t *data2, uint8_t scoreMax, uint32_t sizeTotal);
__global__ void profanity_score_leading(mp_number *pInverse, result *pResult, const uint8_t *data1, const uint8_t *data2, uint8_t scoreMax, uint32_t sizeTotal);
__global__ void profanity_score_range(mp_number *pInverse, result *pResult, const uint8_t *data1, const uint8_t *data2, uint8_t scoreMax, uint32_t sizeTotal);
__global__ void profanity_score_zerobytes(mp_number *pInverse, result *pResult, const uint8_t *data1, const uint8_t *data2, uint8_t scoreMax, uint32_t sizeTotal);
__global__ void profanity_score_leadingrange(mp_number *pInverse, result *pResult, const uint8_t *data1, const uint8_t *data2, uint8_t scoreMax, uint32_t sizeTotal);
__global__ void profanity_score_mirror(mp_number *pInverse, result *pResult, const uint8_t *data1, const uint8_t *data2, uint8_t scoreMax, uint32_t sizeTotal);
__global__ void profanity_score_doubles(mp_number *pInverse, result *pResult, const uint8_t *data1, const uint8_t *data2, uint8_t scoreMax, uint32_t sizeTotal);
}

static constexpr int PROFANITY_MAX_SCORE = 40;

static bool get_secure_random_bytes(void *dst, size_t n) {
    std::ifstream ur("/dev/urandom", std::ios::binary);
    if (!ur.good()) return false;
    ur.read(reinterpret_cast<char*>(dst), n);
    return ur.good();
}

static inline uint64_t bswap64(uint64_t v) {
    return ((v & 0x00000000000000FFull) << 56) |
           ((v & 0x000000000000FF00ull) << 40) |
           ((v & 0x0000000000FF0000ull) << 24) |
           ((v & 0x00000000FF000000ull) << 8)  |
           ((v & 0x000000FF00000000ull) >> 8)  |
           ((v & 0x0000FF0000000000ull) >> 24) |
           ((v & 0x00FF000000000000ull) >> 40) |
           ((v & 0xFF00000000000000ull) >> 56);
}

static std::string::size_type fromHex(char c) {
    if (c >= 'A' && c <= 'F') c = c - 'A' + 'a';
    const std::string hex = "0123456789abcdef";
    return hex.find(c);
}

static ulonglong4 fromHexPK(const std::string &strHex) {
    uint8_t data[32] = {0};
    size_t index = 0;
    for (size_t i = 0; i < strHex.size(); i += 2) {
        auto hi = fromHex(strHex[i]);
        auto lo = (i + 1 < strHex.size()) ? fromHex(strHex[i + 1]) : std::string::npos;
        uint8_t val = ((hi == std::string::npos) ? 0 : (uint8_t)(hi << 4)) | ((lo == std::string::npos) ? 0 : (uint8_t)lo);
        if (index < 32) data[index++] = val;
    }
    ulonglong4 out;
    out.x = bswap64(*(uint64_t*)(data + 24));
    out.y = bswap64(*(uint64_t*)(data + 16));
    out.z = bswap64(*(uint64_t*)(data + 8));
    out.w = bswap64(*(uint64_t*)(data + 0));
    return out;
}

static std::string toHex(const uint8_t *s, size_t len) {
    static const char *b = "0123456789abcdef";
    std::string r; r.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) { r.push_back(b[s[i] >> 4]); r.push_back(b[s[i] & 0xF]); }
    return r;
}

struct DeviceCtx {
    explicit DeviceCtx(const Mode &m) : mode(m), speed(20) {}
    int devId = 0;
    uint32_t index = 0;
    uint32_t worksizeLocal = 256; // unused in CUDA path
    uint32_t inverseSize = 255;
    uint32_t sizeTotal = 0;
    Mode mode;
    ulonglong4 seed{};
    ulonglong4 seedX{};
    ulonglong4 seedY{};
    cudaStream_t stream = nullptr;
    point *d_precomp = nullptr;
    mp_number *d_deltaX = nullptr;
    mp_number *d_inverse = nullptr;
    mp_number *d_prevLambda = nullptr;
    result *d_result = nullptr;
    uint8_t *d_data1 = nullptr;
    uint8_t *d_data2 = nullptr;
    std::vector<result> h_result;
    uint8_t localScoreMax = 0;
    std::chrono::time_point<std::chrono::steady_clock> start;
    uint64_t rounds = 0;
    SpeedSample speed;
};

static std::mutex g_printMutex;
static std::atomic<uint8_t> g_globalScoreMax{0};
static std::atomic<bool> g_quit{false};
static std::mutex g_speedMutex;
static std::vector<double> g_gpuSpeeds;
static std::vector<double> g_gpuLastNonZero;
static std::vector<uint64_t> g_gpuLastUpdateMs;
static unsigned g_speedTick = 0;
static unsigned g_deviceCount = 0;
static std::atomic<bool> g_printRun{false};

static std::string formatRate(double hps) {
    std::ostringstream os; os.setf(std::ios::fixed);
    if (hps >= 1e9) { os << std::setprecision(2) << (hps / 1e9) << " GH/s"; }
    else if (hps >= 1e6) { os << std::setprecision(1) << (hps / 1e6) << " MH/s"; }
    else if (hps >= 1e3) { os << std::setprecision(1) << (hps / 1e3) << " kH/s"; }
    else { os << std::setprecision(0) << hps << " H/s"; }
    return os.str();
}

static void printLoop() {
    const uint64_t watchdogMs = 10000; // 10s without updates => mark as stalled
    while (g_printRun.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        double total = 0.0;
        std::ostringstream ss;
        ss.setf(std::ios::fixed);
        uint64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        {
            std::lock_guard<std::mutex> lk(g_speedMutex);
            for (unsigned i = 0; i < g_deviceCount; ++i) {
                double s = g_gpuSpeeds[i];
                double shown = (s > 0.0 ? s : g_gpuLastNonZero[i]);
                total += shown;
                bool stalled = (now > g_gpuLastUpdateMs[i]) && (now - g_gpuLastUpdateMs[i] > watchdogMs);
                ss << " GPU" << i << ": " << formatRate(shown);
                if (stalled) ss << " (stalled)";
            }
        }
        std::lock_guard<std::mutex> pk(g_printMutex);
        std::cout << "\33[2K\r  Total: " << formatRate(total) << ss.str() << std::flush;
    }
}

static void printFinding(int gpuIndex, const ulonglong4 &seed, uint64_t round, const result &r, uint8_t score, const std::chrono::time_point<std::chrono::steady_clock> &timeStart, const Mode &mode) {
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - timeStart).count();
    uint64_t s0 = seed.x + round; uint64_t carry = (s0 < seed.x) ? 1 : 0;
    uint64_t s1 = seed.y + carry; carry = (s1 < carry) ? 1 : 0;
    uint64_t s2 = seed.z + carry; carry = (s2 < carry) ? 1 : 0;
    uint64_t s3 = seed.w + carry + r.foundId;
    std::ostringstream ss; ss << std::hex << std::setfill('0');
    ss << std::setw(16) << s3 << std::setw(16) << s2 << std::setw(16) << s1 << std::setw(16) << s0;
    std::string priv = ss.str();
    std::string pub = toHex(r.foundHash, 20);
    std::lock_guard<std::mutex> lk(g_printMutex);
    std::cout << "\33[2K\r  Time: " << std::setw(5) << seconds << "s Score: " << std::setw(2) << (int)score
              << " Private: 0x" << priv << ' ' << mode.transformName() << ": 0x" << pub
              << " (GPU" << gpuIndex << ")" << std::endl;
}

static void deviceThread(DeviceCtx ctx) {
    cudaSetDevice(ctx.devId);
    cudaStreamCreate(&ctx.stream);
    size_t precompBytes = sizeof(g_precomp);
    cudaMalloc(&ctx.d_precomp, precompBytes);
    cudaMemcpyAsync(ctx.d_precomp, g_precomp, precompBytes, cudaMemcpyHostToDevice, ctx.stream);
    cudaMalloc(&ctx.d_deltaX, ctx.sizeTotal * sizeof(mp_number));
    cudaMalloc(&ctx.d_inverse, ctx.sizeTotal * sizeof(mp_number));
    cudaMalloc(&ctx.d_prevLambda, ctx.sizeTotal * sizeof(mp_number));
    cudaMalloc(&ctx.d_result, (PROFANITY_MAX_SCORE + 1) * sizeof(result));
    cudaMalloc(&ctx.d_data1, 20);
    cudaMalloc(&ctx.d_data2, 20);
    cudaMemcpyAsync(ctx.d_data1, ctx.mode.data1, 20, cudaMemcpyHostToDevice, ctx.stream);
    cudaMemcpyAsync(ctx.d_data2, ctx.mode.data2, 20, cudaMemcpyHostToDevice, ctx.stream);

    // Initialize
    dim3 block(256);
    dim3 grid((ctx.sizeTotal + block.x - 1) / block.x);
    profanity_init<<<grid, block, 0, ctx.stream>>>(ctx.d_precomp, ctx.d_deltaX, ctx.d_prevLambda, ctx.d_result, ctx.seed, ctx.seedX, ctx.seedY, ctx.sizeTotal, ctx.inverseSize);
    cudaStreamSynchronize(ctx.stream);

    ctx.h_result.resize(PROFANITY_MAX_SCORE + 1);
    ctx.start = std::chrono::steady_clock::now();

    // Main loop
    while (!g_quit.load(std::memory_order_relaxed)) {
        // inverse on groups
        uint32_t groups = (ctx.sizeTotal + ctx.inverseSize - 1) / ctx.inverseSize;
        dim3 gridInv((groups + block.x - 1) / block.x);
        profanity_inverse<<<gridInv, block, 0, ctx.stream>>>(ctx.d_deltaX, ctx.d_inverse, ctx.sizeTotal, ctx.inverseSize);
        profanity_iterate<<<grid, block, 0, ctx.stream>>>(ctx.d_deltaX, ctx.d_inverse, ctx.d_prevLambda, ctx.sizeTotal);
        if (ctx.mode.target == CONTRACT) {
            profanity_transform_contract<<<grid, block, 0, ctx.stream>>>(ctx.d_inverse, ctx.sizeTotal);
        }

        // scoring
        if (ctx.mode.kernel == "profanity_score_benchmark")
            profanity_score_benchmark<<<grid, block, 0, ctx.stream>>>(ctx.d_inverse, ctx.d_result, ctx.d_data1, ctx.d_data2, ctx.localScoreMax, ctx.sizeTotal);
        else if (ctx.mode.kernel == "profanity_score_matching")
            profanity_score_matching<<<grid, block, 0, ctx.stream>>>(ctx.d_inverse, ctx.d_result, ctx.d_data1, ctx.d_data2, ctx.localScoreMax, ctx.sizeTotal);
        else if (ctx.mode.kernel == "profanity_score_leading")
            profanity_score_leading<<<grid, block, 0, ctx.stream>>>(ctx.d_inverse, ctx.d_result, ctx.d_data1, ctx.d_data2, ctx.localScoreMax, ctx.sizeTotal);
        else if (ctx.mode.kernel == "profanity_score_range")
            profanity_score_range<<<grid, block, 0, ctx.stream>>>(ctx.d_inverse, ctx.d_result, ctx.d_data1, ctx.d_data2, ctx.localScoreMax, ctx.sizeTotal);
        else if (ctx.mode.kernel == "profanity_score_zerobytes")
            profanity_score_zerobytes<<<grid, block, 0, ctx.stream>>>(ctx.d_inverse, ctx.d_result, ctx.d_data1, ctx.d_data2, ctx.localScoreMax, ctx.sizeTotal);
        else if (ctx.mode.kernel == "profanity_score_leadingrange")
            profanity_score_leadingrange<<<grid, block, 0, ctx.stream>>>(ctx.d_inverse, ctx.d_result, ctx.d_data1, ctx.d_data2, ctx.localScoreMax, ctx.sizeTotal);
        else if (ctx.mode.kernel == "profanity_score_mirror")
            profanity_score_mirror<<<grid, block, 0, ctx.stream>>>(ctx.d_inverse, ctx.d_result, ctx.d_data1, ctx.d_data2, ctx.localScoreMax, ctx.sizeTotal);
        else if (ctx.mode.kernel == "profanity_score_doubles")
            profanity_score_doubles<<<grid, block, 0, ctx.stream>>>(ctx.d_inverse, ctx.d_result, ctx.d_data1, ctx.d_data2, ctx.localScoreMax, ctx.sizeTotal);

        // Fetch results
        cudaMemcpyAsync(ctx.h_result.data(), ctx.d_result, ctx.h_result.size() * sizeof(result), cudaMemcpyDeviceToHost, ctx.stream);
        cudaStreamSynchronize(ctx.stream);
        ++ctx.rounds;

        // Update per-GPU speed (H/s)
        ctx.speed.sample(static_cast<double>(ctx.sizeTotal));
        {
            std::lock_guard<std::mutex> lk(g_speedMutex);
            const double v = ctx.speed.getSpeed();
            g_gpuSpeeds[ctx.index] = v;
            if (v > 0.0) g_gpuLastNonZero[ctx.index] = v;
            g_gpuLastUpdateMs[ctx.index] = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        }

        for (int i = PROFANITY_MAX_SCORE; i > ctx.localScoreMax; --i) {
            result &r = ctx.h_result[i];
            if (r.found > 0 && i >= ctx.localScoreMax) {
                ctx.localScoreMax = i;
                uint8_t globalPrev = g_globalScoreMax.load();
                if (i > globalPrev) {
                    g_globalScoreMax.store(i);
                    printFinding(ctx.index, ctx.seed, ctx.rounds, r, (uint8_t)i, ctx.start, ctx.mode);
                }
                break;
            }
        }
    }

    cudaStreamDestroy(ctx.stream);
    cudaFree(ctx.d_precomp); cudaFree(ctx.d_deltaX); cudaFree(ctx.d_inverse); cudaFree(ctx.d_prevLambda);
    cudaFree(ctx.d_result); cudaFree(ctx.d_data1); cudaFree(ctx.d_data2);
}

int main(int argc, char **argv) {
    try {
        ArgParser argp(argc, argv);
        bool bHelp = false, bModeBenchmark = false, bModeZeros = false, bModeZeroBytes = false, bModeLetters = false, bModeNumbers = false;
        std::string strModeLeading, strModeMatching, strPublicKey; bool bModeLeadingRange = false, bModeRange = false, bModeMirror = false, bModeDoubles = false;
        int rangeMin = 0, rangeMax = 0; std::vector<size_t> vDeviceSkipIndex; uint32_t worksizeLocal = 256; size_t worksizeMax = 0; size_t inverseSize = 255; size_t inverseMultiple = 16384; bool bMineContract = false;
        bool bNoCache = false; // unused, kept for CLI parity
        argp.addSwitch('h', "help", bHelp);
        argp.addSwitch('0', "benchmark", bModeBenchmark);
        argp.addSwitch('1', "zeros", bModeZeros);
        argp.addSwitch('2', "letters", bModeLetters);
        argp.addSwitch('3', "numbers", bModeNumbers);
        argp.addSwitch('4', "leading", strModeLeading);
        argp.addSwitch('5', "matching", strModeMatching);
        argp.addSwitch('6', "leading-range", bModeLeadingRange);
        argp.addSwitch('7', "range", bModeRange);
        argp.addSwitch('8', "mirror", bModeMirror);
        argp.addSwitch('9', "leading-doubles", bModeDoubles);
        argp.addSwitch('m', "min", rangeMin);
        argp.addSwitch('M', "max", rangeMax);
        argp.addMultiSwitch('s', "skip", vDeviceSkipIndex);
        argp.addSwitch('w', "work", worksizeLocal);
        argp.addSwitch('W', "work-max", worksizeMax);
        argp.addSwitch('n', "no-cache", bNoCache);
        argp.addSwitch('i', "inverse-size", inverseSize);
        argp.addSwitch('I', "inverse-multiple", inverseMultiple);
        argp.addSwitch('c', "contract", bMineContract);
        argp.addSwitch('z', "publicKey", strPublicKey);
        argp.addSwitch('b', "zero-bytes", bModeZeroBytes);
        if (!argp.parse()) { std::cout << "error: bad arguments, try again :<" << std::endl; return 1; }
        if (bHelp) { std::cout << g_strHelp << std::endl; return 0; }

        Mode mode = Mode::benchmark();
        if (bModeBenchmark) mode = Mode::benchmark();
        else if (bModeZeros) mode = Mode::zeros();
        else if (bModeLetters) mode = Mode::letters();
        else if (bModeNumbers) mode = Mode::numbers();
        else if (!strModeLeading.empty()) mode = Mode::leading(strModeLeading.front());
        else if (!strModeMatching.empty()) mode = Mode::matching(strModeMatching);
        else if (bModeLeadingRange) mode = Mode::leadingRange(rangeMin, rangeMax);
        else if (bModeRange) mode = Mode::range(rangeMin, rangeMax);
        else if (bModeMirror) mode = Mode::mirror();
        else if (bModeDoubles) mode = Mode::doubles();
        else if (bModeZeroBytes) mode = Mode::zeroBytes();
        else { std::cout << g_strHelp << std::endl; return 0; }

        if (strPublicKey.empty()) { std::cout << "error: this tool requires your public key (-z)" << std::endl; return 1; }
        if (strPublicKey.length() != 128) { std::cout << "error: public key must be 128 hex characters" << std::endl; return 1; }

        mode.target = bMineContract ? CONTRACT : ADDRESS;
        std::cout << "Mode: " << mode.name << std::endl;
        std::cout << "Target: " << mode.transformName() << std::endl;

        int deviceCount = 0; cudaGetDeviceCount(&deviceCount);
        if (deviceCount <= 0) { std::cout << "No CUDA devices found" << std::endl; return 1; }

        std::vector<int> devices;
        std::cout << "Devices:" << std::endl;
        for (int i = 0; i < deviceCount; ++i) {
            if (std::find(vDeviceSkipIndex.begin(), vDeviceSkipIndex.end(), (size_t)i) != vDeviceSkipIndex.end()) continue;
            cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, i);
            std::cout << "  GPU" << i << ": " << prop.name << ", " << prop.totalGlobalMem << " bytes, " << prop.multiProcessorCount << " SMs" << std::endl;
            devices.push_back(i);
        }
        if (devices.empty()) return 1;

        g_deviceCount = static_cast<unsigned>(devices.size());
        g_gpuSpeeds.assign(g_deviceCount, 0.0);
        g_gpuLastNonZero.assign(g_deviceCount, 0.0);
        g_gpuLastUpdateMs.assign(g_deviceCount, 0);

        const uint32_t invSize = (uint32_t)std::min<size_t>(inverseSize, 255);
        const uint32_t sizeTotal = (uint32_t)std::min<size_t>(worksizeMax == 0 ? inverseSize * inverseMultiple : worksizeMax, 0xFFFFFFFFu);

        // Seeds derived from provided public key (X/Y) + secure per-device base seed
        ulonglong4 seedX = fromHexPK(strPublicKey.substr(0, 64));
        ulonglong4 seedY = fromHexPK(strPublicKey.substr(64, 64));

        std::vector<std::thread> threads;
        std::thread printer;
        uint32_t index = 0;
        for (int dev : devices) {
            DeviceCtx ctx(mode);
            ctx.devId = dev; ctx.index = index++; ctx.worksizeLocal = worksizeLocal; ctx.inverseSize = invSize; ctx.sizeTotal = sizeTotal; ctx.seedX = seedX; ctx.seedY = seedY;
            if (!get_secure_random_bytes(&ctx.seed, sizeof(ctx.seed))) { std::cerr << "error: failed to get secure random seed" << std::endl; return 1; }
            threads.emplace_back(deviceThread, ctx);
        }

        // Start periodic printer
        g_printRun.store(true);
        printer = std::thread(printLoop);

        std::cout << std::endl;
        std::cout << "Running..." << std::endl;
        std::cout << "  Always verify that a private key generated by this program corresponds to the" << std::endl;
        std::cout << "  public key printed by importing it to a wallet of your choice. This program" << std::endl;
        std::cout << "  like any software might contain bugs and it does by design cut corners to" << std::endl;
        std::cout << "  improve overall performance." << std::endl;
        std::cout << std::endl;

        for (auto &t : threads) t.join();
        g_printRun.store(false);
        if (printer.joinable()) printer.join();
        return 0;
    } catch (std::runtime_error &e) {
        std::cout << "std::runtime_error - " << e.what() << std::endl; return 1;
    } catch (...) {
        std::cout << "unknown exception occurred" << std::endl; return 1;
    }
}
