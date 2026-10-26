#include "MLIPRankComm.h"

#include "pme_rank_split.h"

#include <mpi.h>

#include <climits>
#include <stdexcept>

namespace {

class MpiRankComm final : public MLIPRankComm {
public:
    MpiRankComm() {
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized) {
            comm_ = PmePpSplit::instance().pp_comm();
            MPI_Comm_size(comm_, &size_);
            MPI_Comm_rank(comm_, &rank_);
        }
    }

    int size() const override { return size_; }
    int rank() const override { return rank_; }

    void allgatherv(const double* mine, int64_t count, std::vector<double>& all, std::vector<int64_t>& counts) override {
        gather(mine, count, MPI_DOUBLE, all, counts);
    }

    void allgatherv(const int* mine, int64_t count, std::vector<int>& all, std::vector<int64_t>& counts) override {
        gather(mine, count, MPI_INT, all, counts);
    }

private:
    template <class T>
    void gather(const T* mine, int64_t count, MPI_Datatype type, std::vector<T>& all, std::vector<int64_t>& counts) {
        counts.assign(static_cast<size_t>(size_), count);
        if (size_ <= 1) {
            all.assign(mine, mine + count);
            return;
        }
        MPI_Allgather(&count, 1, MPI_INT64_T, counts.data(), 1, MPI_INT64_T, comm_);
        std::vector<int> receive_counts(static_cast<size_t>(size_)), displacements(static_cast<size_t>(size_));
        int64_t total = 0;
        for (int r = 0; r < size_; ++r) {
            displacements[static_cast<size_t>(r)] = static_cast<int>(total);
            total += counts[static_cast<size_t>(r)];
            if (total > INT_MAX) throw std::runtime_error("MLIPRankComm: more atoms than one MPI message can count");
            receive_counts[static_cast<size_t>(r)] = static_cast<int>(counts[static_cast<size_t>(r)]);
        }
        all.resize(static_cast<size_t>(total));
        MPI_Allgatherv(mine, static_cast<int>(count), type, all.data(), receive_counts.data(), displacements.data(), type,
                       comm_);
    }

    MPI_Comm comm_ = MPI_COMM_NULL;
    int size_ = 1;
    int rank_ = 0;
};

}  // namespace

std::unique_ptr<MLIPRankComm> make_mlip_rank_comm() { return std::make_unique<MpiRankComm>(); }
