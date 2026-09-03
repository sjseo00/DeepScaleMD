#include "mpi_optimizer.h"
#include <iostream>
#include <chrono>
#include <algorithm>

#include <mpi.h>

MPIOptimizer::MPIOptimizer() 
    : batching_enabled_(true),
      async_enabled_(true),
      total_reductions_(0),
      batched_reductions_(0),
      async_reductions_(0),
      total_comm_time_(0.0) {
}

MPIOptimizer::~MPIOptimizer() {
    wait_all();
}

void MPIOptimizer::allreduce_async_double(const double* sendbuf, double* recvbuf, int count, MPI_Op op) {
    if (!async_enabled_) {
        // Fallback to blocking
        MPI_Allreduce(sendbuf, recvbuf, count, MPI_DOUBLE, op, MPI_COMM_WORLD);
        return;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    ReductionRequest req;
    req.type = ReductionRequest::DOUBLE;
    req.sendbuf = sendbuf;
    req.recvbuf = recvbuf;
    req.count = count;
    req.op = op;
    req.completed = false;
    
    MPI_Iallreduce(sendbuf, recvbuf, count, MPI_DOUBLE, op, MPI_COMM_WORLD, &req.request);
    
    requests_.push_back(req);
    active_requests_.push_back(req.request);
    
    total_reductions_++;
    async_reductions_++;
    
    auto end = std::chrono::high_resolution_clock::now();
    total_comm_time_ += std::chrono::duration<double>(end - start).count();
}

void MPIOptimizer::allreduce_async_int(const int* sendbuf, int* recvbuf, int count, MPI_Op op) {
    if (!async_enabled_) {
        MPI_Allreduce(sendbuf, recvbuf, count, MPI_INT, op, MPI_COMM_WORLD);
        return;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    ReductionRequest req;
    req.type = ReductionRequest::INT;
    req.sendbuf = sendbuf;
    req.recvbuf = recvbuf;
    req.count = count;
    req.op = op;
    req.completed = false;
    
    MPI_Iallreduce(sendbuf, recvbuf, count, MPI_INT, op, MPI_COMM_WORLD, &req.request);
    
    requests_.push_back(req);
    active_requests_.push_back(req.request);
    
    total_reductions_++;
    async_reductions_++;
    
    auto end = std::chrono::high_resolution_clock::now();
    total_comm_time_ += std::chrono::duration<double>(end - start).count();
}

void MPIOptimizer::queue_reduction_double(const double* sendbuf, double* recvbuf, int count, MPI_Op op) {
    if (!batching_enabled_) {
        allreduce_async_double(sendbuf, recvbuf, count, op);
        return;
    }
    
    ReductionRequest req;
    req.type = ReductionRequest::DOUBLE;
    req.sendbuf = sendbuf;
    req.recvbuf = recvbuf;
    req.count = count;
    req.op = op;
    req.completed = false;
    
    requests_.push_back(req);
    batched_reductions_++;
}

void MPIOptimizer::queue_reduction_int(const int* sendbuf, int* recvbuf, int count, MPI_Op op) {
    if (!batching_enabled_) {
        allreduce_async_int(sendbuf, recvbuf, count, op);
        return;
    }
    
    ReductionRequest req;
    req.type = ReductionRequest::INT;
    req.sendbuf = sendbuf;
    req.recvbuf = recvbuf;
    req.count = count;
    req.op = op;
    req.completed = false;
    
    requests_.push_back(req);
    batched_reductions_++;
}

void MPIOptimizer::execute_batched_reductions() {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch all queued reductions as non-blocking operations
    for (auto& req : requests_) {
        if (!req.completed) {
            if (req.type == ReductionRequest::DOUBLE) {
                MPI_Iallreduce(req.sendbuf, req.recvbuf, req.count, MPI_DOUBLE, 
                              req.op, MPI_COMM_WORLD, &req.request);
            } else {
                MPI_Iallreduce(req.sendbuf, req.recvbuf, req.count, MPI_INT, 
                              req.op, MPI_COMM_WORLD, &req.request);
            }
            active_requests_.push_back(req.request);
            total_reductions_++;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    total_comm_time_ += std::chrono::duration<double>(end - start).count();
}

void MPIOptimizer::wait_all() {
    if (active_requests_.empty()) {
        return;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    MPI_Waitall(active_requests_.size(), active_requests_.data(), MPI_STATUSES_IGNORE);
    
    auto end = std::chrono::high_resolution_clock::now();
    total_comm_time_ += std::chrono::duration<double>(end - start).count();
    
    // Mark all as completed
    for (auto& req : requests_) {
        req.completed = true;
    }
    
    active_requests_.clear();
}

void MPIOptimizer::wait_for_reduction(int index) {
    if (index >= 0) {
        const size_t idx = static_cast<size_t>(index);
        if (idx < requests_.size() && !requests_[idx].completed) {
        auto start = std::chrono::high_resolution_clock::now();
        
        MPI_Wait(&requests_[idx].request, MPI_STATUS_IGNORE);
        requests_[idx].completed = true;
        
        auto end = std::chrono::high_resolution_clock::now();
        total_comm_time_ += std::chrono::duration<double>(end - start).count();
        }
    }
}

bool MPIOptimizer::test_all() {
    if (active_requests_.empty()) {
        return true;
    }
    
    int flag = 0;
    MPI_Testall(active_requests_.size(), active_requests_.data(), &flag, MPI_STATUSES_IGNORE);
    
    if (flag) {
        for (auto& req : requests_) {
            req.completed = true;
        }
        active_requests_.clear();
        return true;
    }
    return false;
}

void MPIOptimizer::cleanup_completed() {
    requests_.erase(
        std::remove_if(requests_.begin(), requests_.end(),
                      [](const ReductionRequest& req) { return req.completed; }),
        requests_.end()
    );
}

void MPIOptimizer::print_statistics() const {
    std::cout << "\n=== MPI Optimizer Statistics ===\n";
    std::cout << "Total reductions: " << total_reductions_ << "\n";
    std::cout << "Batched reductions: " << batched_reductions_ << "\n";
    std::cout << "Async reductions: " << async_reductions_ << "\n";
    std::cout << "Total comm time: " << total_comm_time_ << " s\n";
    
    if (total_reductions_ > 0) {
        std::cout << "Avg time per reduction: " 
                  << (total_comm_time_ / total_reductions_ * 1000.0) << " ms\n";
    }
    std::cout << "==============================\n\n";
}

void MPIOptimizer::reset_statistics() {
    total_reductions_ = 0;
    batched_reductions_ = 0;
    async_reductions_ = 0;
    total_comm_time_ = 0.0;
}
