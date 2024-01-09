#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <sycl/sycl.hpp>
#include "Gate.h"

#include <unordered_map>
using std::make_pair;
using std::unordered_map;

#define TRUTH_SIZE 1025
#define C_THREAD_LIMIT 32

const size_t C_PARALLEL_LIMIT = 10000;

typedef struct {
  size_t funcSer;
  Event *iPort[MAX_INPUT_PORT];
  dUnit dTable[MAX_DTABLE];
} gpu_GATE;

class Simulator {
private:
  sycl::queue q;

  tUnit dumpOff;

  size_t parallelSize;

  char *d_eTableMgr;
  gpu_GATE *d_gateMgr;

  bool h_overflow[C_PARALLEL_LIMIT];
  bool *d_overflow;

  size_t h_oHisSize[C_PARALLEL_LIMIT];
  size_t *d_oHisSize;

  size_t h_oHisMax[C_PARALLEL_LIMIT];
  size_t *d_oHisMax;

  Event *h_oHisArr[C_PARALLEL_LIMIT];
  Event **d_oHisArr;

  unordered_map<size_t, Event *> d_wireMapper;
  // Temp cache
  Event *d_HisTmp;

public:
  Simulator(sycl::queue, tUnit);
  ~Simulator() {}

  void addEvalMgr(char *, size_t);
  void addTrans(tHistory *, size_t &);
  Event *getTrans(size_t &);
  void popTrans(size_t &);
  void addGateMgr(gpu_GATE *, size_t &);
  void simulateBlock(tHistory **, size_t *, size_t[][3]);
  void cleanGPU() {
    sycl::free(d_overflow, q);
    sycl::free(d_eTableMgr, q);
    sycl::free(d_oHisSize, q);
    sycl::free(d_oHisArr, q);
    sycl::free(d_oHisMax, q);
    sycl::free(d_gateMgr, q);
    for (auto &pair : d_wireMapper) {
      sycl::free(pair.second, q);
    }
  }

  void simulate(Gate *);
  void simulateBlock(vector<Gate *> &);

#ifdef GPU_DEBUG
  void checkEvalMgr(char *, size_t);
  void checkTrans(size_t, size_t);
  void checkGate(gpu_GATE *);
#endif
};

#endif