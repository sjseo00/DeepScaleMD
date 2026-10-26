#ifndef MLIP_RANK_COMM_H
#define MLIP_RANK_COMM_H

// How an MLIP backend reaches the other MPI ranks of the run. No MPI types here: PythonBackend (which unit tests build
// without MPI) and the replicated libtorch step use only this interface; the MPI version is make_mlip_rank_comm()
// (src/MLIPRankComm.cpp, in the MD target). A backend without a comm (nullptr) behaves as a single process.
//
// Used where an MLIP cannot work on owned + ghost atoms of one subdomain: every rank collects the atoms of all ranks,
// computes the whole system (periodic images as usual) and keeps the forces of its own atoms.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

class MLIPRankComm {
public:
    virtual ~MLIPRankComm() = default;
    virtual int size() const = 0;   // ranks that run the MD (PP ranks)
    virtual int rank() const = 0;   // this rank among them
    // Every rank passes its `count` values and receives all of them, the values of rank 0 first. counts[r] = values of rank r.
    virtual void allgatherv(const double* mine, int64_t count, std::vector<double>& all, std::vector<int64_t>& counts) = 0;
    virtual void allgatherv(const int* mine, int64_t count, std::vector<int>& all, std::vector<int64_t>& counts) = 0;
};

// The MPI version over the ranks that run the MD (the PP communicator when PME has its own ranks).
std::unique_ptr<MLIPRankComm> make_mlip_rank_comm();

// The `n` values of this rank and then those of all the other ranks (in rank order) in `ordered`: this rank's atoms stay
// at the front, so the first n entries of any per-atom result line up with this rank's owned atoms. `scratch` is work space.
template <class T>
void gather_mine_first(MLIPRankComm& comm, const T* mine, int64_t n, std::vector<T>& ordered, std::vector<T>& scratch) {
    std::vector<int64_t> counts;
    comm.allgatherv(mine, n, scratch, counts);
    int64_t offset = 0;
    for (int r = 0; r < comm.rank(); ++r) offset += counts[static_cast<size_t>(r)];
    ordered.resize(scratch.size());
    std::copy(scratch.begin() + offset, scratch.begin() + offset + n, ordered.begin());
    std::copy(scratch.begin(), scratch.begin() + offset, ordered.begin() + n);
    std::copy(scratch.begin() + offset + n, scratch.end(), ordered.begin() + n + offset);
}

#endif // MLIP_RANK_COMM_H
