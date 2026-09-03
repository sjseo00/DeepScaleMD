# MD 코드의 다중 MLIP 지원 구조 설계 및 리팩터링 지시서

## 목표

현재 구현 중인 C++/CUDA 기반 MD 코드를 리팩터링하여 다음 조건을 만족하도록 수정해 주세요.

1. 여러 종류의 MLIP를 공통 인터페이스로 쉽게 추가할 수 있어야 합니다.
2. MPI 기반 domain decomposition과 멀티노드·멀티GPU 실행을 지원해야 합니다.
3. 각 MPI rank가 담당 GPU를 소유하고, 해당 GPU 내부에서 MD 계산과 MLIP 추론을 모두 수행해야 합니다.
4. 원자 좌표, 원자 종류, neighbor list, force 등의 핵심 데이터는 가능한 한 GPU 메모리에 상주시켜야 합니다.
5. 매 MD step마다 CPU↔GPU 데이터 복사가 발생하지 않도록 해야 합니다.
6. MD 코드와 MLIP 런타임이 동일한 GPU 메모리와 CUDA stream을 공유할 수 있어야 합니다.
7. PyTorch, TorchScript, LibTorch, ONNX Runtime, TensorRT 또는 MLIP별 native C++/CUDA backend를 adapter 방식으로 연결할 수 있어야 합니다.
8. 특정 MLIP 프레임워크의 tensor나 데이터 구조를 MD 코어 전체에 노출하지 않아야 합니다.

---

## 기본 실행 모델

전체 프로그램은 다음 실행 모델을 기준으로 설계해 주세요.

```text
Node
 ├─ MPI rank 0 → GPU 0
 ├─ MPI rank 1 → GPU 1
 ├─ MPI rank 2 → GPU 2
 └─ MPI rank 3 → GPU 3
```

각 MPI rank는 다음 작업을 담당합니다.

```text
MPI domain decomposition
    ↓
local atom 및 ghost atom 관리
    ↓
GPU 상에서 neighbor list 생성
    ↓
GPU 상의 atom 및 neighbor 데이터를 MLIP에 전달
    ↓
동일 GPU에서 MLIP inference
    ↓
GPU force buffer에 force 누적
    ↓
ghost force reverse communication
    ↓
GPU 상에서 integration 및 position update
    ↓
필요한 경우 atom migration 및 neighbor rebuild
```

한 MPI rank가 여러 GPU에 데이터를 다시 분할하는 구조보다는 우선적으로 **1 MPI rank당 1 GPU** 구조를 기본 구현으로 사용해 주세요.

향후 rank당 여러 GPU가 필요한 경우 확장할 수 있도록 device ID와 execution context를 명시적으로 관리해 주세요.

---

# 1. 아키텍처 원칙

## 1.1 MD 코어와 MLIP backend 분리

MD 코어는 PyTorch, TensorFlow, JAX, ONNX 또는 특정 MLIP의 내부 구현을 알면 안 됩니다.

MD 코어가 알아야 하는 것은 다음뿐입니다.

- 입력 원자 수
- local atom 수
- ghost atom 수
- 원자 좌표
- 원자 종류
- simulation cell
- periodic boundary condition
- neighbor pair 또는 neighbor list
- cutoff
- 출력 energy
- 출력 force
- 출력 virial 또는 stress
- CUDA device
- CUDA stream

다음과 같은 계층으로 분리해 주세요.

```text
MD Core
   ↓
MLIP Common Interface
   ↓
Model Adapter
   ↓
Runtime Backend
   ↓
Actual MLIP Model
```

예시:

```text
MD Core
 └─ MlipPotential
     ├─ MaceAdapter
     │   └─ LibTorchBackend
     ├─ NequipAdapter
     │   └─ LibTorchBackend
     ├─ AllegroAdapter
     │   └─ LibTorchBackend
     ├─ DeePMDAdapter
     │   └─ DeePMD Native Backend
     ├─ OnnxMlipAdapter
     │   └─ ONNX Runtime CUDA
     └─ TensorRTMlipAdapter
         └─ TensorRT
```

---

## 1.2 MD 코어에서 PyTorch tensor를 직접 사용하지 말 것

MD 코드의 원자 데이터 구조 자체를 `torch::Tensor`로 구현하지 마세요.

MD 코어는 독립적인 C++/CUDA 데이터 구조를 사용해야 합니다.

실제 precision과 layout은 템플릿 또는 설정으로 선택 가능하게 설계해 주세요.

MLIP adapter 내부에서만 MD의 raw GPU pointer를 backend의 tensor/view로 감싸도록 해 주세요.

LibTorch의 경우 불필요한 복사 없이 다음 방식과 유사하게 처리해야 합니다.

```cpp
auto positions = torch::from_blob(
    atom_data.positions,
    {atom_data.num_total, 3},
    torch::TensorOptions()
        .dtype(torch::kFloat64)
        .device(torch::kCUDA, device_id)
);
```

다만 `from_blob`으로 생성한 tensor는 메모리를 소유하지 않으므로 lifetime을 명확히 관리해 주세요.

MD buffer가 model inference가 완료되기 전에 재할당되거나 해제되지 않도록 해야 합니다.

---

# 2. 공통 MLIP 인터페이스

다음과 유사한 추상 인터페이스를 구현해 주세요.

```cpp
class IMlipPotential {
public:
    virtual ~IMlipPotential() = default;

    virtual void initialize(
        const MlipModelConfig& model_config,
        const MlipExecutionContext& context
    ) = 0;

    virtual MlipCapabilities capabilities() const = 0;

    virtual double cutoff() const = 0;

    virtual void compute(
        const MlipInputView& input,
        MlipOutputView& output,
        const MlipComputeOptions& options
    ) = 0;

    virtual void synchronize() = 0;
};
```

초기에는 비동기 실행이 가능하도록 `compute()` 내부에서 불필요한 `cudaDeviceSynchronize()`를 호출하지 마세요.

가능하면 `compute()`는 전달된 CUDA stream에 작업을 enqueue하고 즉시 반환하도록 설계해 주세요.

---

## 2.1 실행 컨텍스트

```cpp
struct MlipExecutionContext {
    int mpi_rank;
    int mpi_size;

    int device_id;
    cudaStream_t stream;

    void* communicator;
    void* memory_resource;
};
```

`communicator`는 필요한 경우 MPI communicator를 추상화하기 위한 필드로 사용합니다.

MLIP backend가 MPI를 직접 호출하지 않는 것이 기본 원칙입니다. MPI domain decomposition, ghost exchange 및 force reverse communication은 MD 코어가 담당해야 합니다.

---

## 2.2 입력 데이터 구조

```cpp
struct MlipInputView {
    int num_local;
    int num_ghost;
    int num_total;

    DeviceArrayView positions;
    DeviceArrayView species;

    DeviceNeighborListView neighbors;

    DeviceArrayView cell;
    DeviceArrayView pbc;

    cudaStream_t stream;
};
```

`DeviceArrayView`는 메모리를 소유하지 않는 view여야 합니다.

```cpp
enum class ScalarType {
    Float32,
    Float64,
    Int32,
    Int64
};

struct DeviceArrayView {
    void* data;
    ScalarType dtype;

    int ndim;
    std::array<int64_t, 4> shape;
    std::array<int64_t, 4> stride;

    int device_id;
};
```

실제 코드에서는 고정 크기 배열 대신 더 적절한 경량 구조를 사용해도 됩니다.

핵심은 다음 정보를 명시적으로 전달하는 것입니다.

- device pointer
- dtype
- shape
- stride
- device ID
- 메모리 소유권 없음

---

## 2.3 Neighbor list 표현

MD 코드 내부의 canonical neighbor representation을 정의해 주세요.

권장 기본 표현은 edge list입니다.

```cpp
struct DeviceNeighborListView {
    int64_t num_edges;

    const int32_t* src;
    const int32_t* dst;

    const int32_t* cell_shift;  // [num_edges][3], optional
    const double* edge_vector;  // [num_edges][3], optional
    const double* distance;     // [num_edges], optional

    bool is_full_list;
    bool includes_ghost_atoms;
    bool directed;
};
```

또는 다음처럼 packed edge index를 사용할 수 있습니다.

```cpp
int32_t* edge_index;  // [2][num_edges]
```

MD 코어 내부에서는 하나의 canonical 형식을 유지하고, 각 MLIP adapter가 모델이 요구하는 형식으로 변환하도록 해 주세요.

예:

- MACE/NequIP: directed edge index
- 일부 classical-style 모델: half neighbor list
- DeePMD: 모델 고유 neighbor format
- 일부 모델: 상대 위치 벡터와 periodic shift 필요
- 일부 모델: 원자별 CSR neighbor list 필요

형식 변환은 adapter 내부에서 수행하되, 매 step마다 임시 메모리를 새로 할당하지 마세요.

---

# 3. Neighbor-list convention 처리

MLIP마다 neighbor list 정의가 다를 수 있으므로 다음 메타데이터를 모델 adapter가 명시하도록 해 주세요.

```cpp
struct NeighborRequirements {
    double cutoff;

    bool requires_full_list;
    bool requires_directed_edges;
    bool requires_self_edges;

    bool requires_edge_vectors;
    bool requires_distances;
    bool requires_cell_shifts;

    bool supports_ghost_atoms;
};
```

다음 차이를 반드시 처리해야 합니다.

- half list와 full list
- directed edge와 undirected pair
- `i → j`와 `j → i`를 둘 다 요구하는 모델
- self interaction 포함 여부
- local-local pair
- local-ghost pair
- ghost-ghost pair 제외 여부
- periodic image shift
- neighbor ordering
- cutoff 포함 조건
- skin distance
- 모델 cutoff와 MD neighbor cutoff 차이

MD neighbor cutoff는 최소한 다음 조건을 만족해야 합니다.

```text
MD neighbor cutoff >= MLIP model cutoff + neighbor skin
```

여러 potential을 동시에 사용하는 경우 필요한 cutoff의 최댓값을 사용해 주세요.

---

# 4. MPI domain decomposition과 ghost atom

MPI domain decomposition은 MD 코어의 책임으로 유지해 주세요.

각 rank의 MLIP 입력은 기본적으로 다음 데이터를 포함합니다.

```text
local atoms
+
해당 local atom의 MLIP cutoff 안에 필요한 ghost atoms
```

MLIP가 계산한 local atom force뿐 아니라 ghost atom에 누적된 force도 올바르게 처리해야 합니다.

실행 순서는 다음과 같습니다.

```text
1. forward communication
   - 필요한 ghost position과 species 갱신

2. neighbor list 구성

3. MLIP inference
   - local 및 ghost atom을 포함한 graph 입력
   - local atom의 energy/force 계산
   - ghost atom force가 생성될 수 있음

4. reverse communication
   - ghost atom에 누적된 force를 owner rank로 전달

5. owner rank의 local force에 합산
```

Newton's third law 적용 여부와 neighbor list 형식에 따라 force 중복 계산이 발생하지 않도록 명확히 설계해 주세요.

MLIP는 일반 pair potential과 달리 many-body graph model일 수 있으므로, 단순히 half neighbor list와 Newton pair 최적화를 그대로 적용할 수 있다고 가정하지 마세요.

---

# 5. GPU-resident 데이터 흐름

매 step의 기본 데이터 흐름은 다음과 같아야 합니다.

```text
GPU position
    ↓
GPU neighbor build
    ↓
GPU MLIP input view 생성
    ↓
GPU MLIP inference
    ↓
GPU force
    ↓
GPU integration
```

다음 동작을 금지하거나 최소화해 주세요.

- 매 step `cudaMemcpyDeviceToHost`
- 매 step `cudaMemcpyHostToDevice`
- 매 step tensor 재할당
- 매 step neighbor buffer 재할당
- 매 step CUDA stream 생성 및 제거
- adapter 내부의 암시적 CPU fallback
- adapter 내부의 무조건적 device synchronization
- GPU 결과를 CPU로 가져온 뒤 다시 GPU로 전달하는 경로

CPU로 가져와도 되는 데이터는 다음 정도로 제한합니다.

- thermodynamic logging
- checkpoint
- trajectory output
- 디버깅
- 낮은 주기의 global reduction
- atom migration에 필요한 최소 metadata

가능하면 MPI 통신에서도 CUDA-aware MPI를 사용하여 device pointer를 직접 전달할 수 있도록 설계해 주세요.

CUDA-aware MPI가 없는 환경을 위해 pinned host staging fallback을 별도로 구현할 수 있습니다.

```text
Preferred:
GPU → CUDA-aware MPI → GPU

Fallback:
GPU → pinned host → MPI → pinned host → GPU
```

---

# 6. CUDA stream 공유

MD와 MLIP는 동일 GPU에서 동일 stream 또는 명시적으로 동기화된 stream을 사용해야 합니다.

기본 정책은 다음과 같습니다.

```text
rank별 persistent CUDA stream 생성
    ↓
neighbor kernel enqueue
    ↓
MLIP inference enqueue
    ↓
force post-processing enqueue
    ↓
integration enqueue
```

LibTorch를 사용할 경우 현재 MD stream을 PyTorch가 사용하도록 stream guard를 적용해 주세요.

개념적 예:

```cpp
at::cuda::CUDAStream torch_stream =
    at::cuda::getStreamFromExternal(md_stream, device_id);

at::cuda::CUDAStreamGuard guard(torch_stream);

model.forward(inputs);
```

실제 사용 API는 설치된 LibTorch 버전에 맞게 확인해 주세요.

PyTorch default stream을 암묵적으로 사용할 경우 다음 문제가 발생할 수 있으므로 피해야 합니다.

- MD kernel과 model inference 사이의 잘못된 실행 순서
- 불필요한 synchronization
- race condition
- force가 완성되기 전에 integration 실행
- 예상치 못한 stream 간 dependency

서로 다른 stream을 사용할 경우 반드시 CUDA event로 dependency를 연결해 주세요.

```text
MD stream: neighbor 완료 event 기록
MLIP stream: event 대기 후 inference
MLIP stream: inference 완료 event 기록
MD stream: event 대기 후 integration
```

---

# 7. 메모리 관리

## 7.1 Persistent buffer 사용

다음 버퍼는 simulation 동안 재사용해 주세요.

- positions
- velocities
- forces
- species
- global IDs
- local-to-global mapping
- ghost mapping
- neighbor index
- neighbor offset
- edge vector
- periodic shift
- MLIP input staging
- MLIP output staging
- per-atom energy
- virial
- reduction workspace

capacity 기반으로 관리하고, atom 수나 edge 수가 capacity를 초과할 때만 재할당해 주세요.

```cpp
if (required_size > capacity) {
    reserve(grow_capacity(required_size));
}
```

growth factor를 적용하여 잦은 재할당을 피하세요.

---

## 7.2 메모리 소유권

다음 소유권 규칙을 명확히 지켜 주세요.

```text
MD Core:
    atom buffer와 canonical neighbor buffer 소유

MLIP Adapter:
    backend별 변환용 임시 buffer 소유

Runtime Backend:
    model weight와 runtime context 소유

Tensor/View:
    원칙적으로 MD buffer를 비소유 참조
```

adapter가 MD pointer를 해제해서는 안 됩니다.

---

## 7.3 메모리 allocator

가능하면 반복 할당 비용을 줄이기 위해 다음 중 하나를 사용해 주세요.

- `cudaMallocAsync`
- rank별 CUDA memory pool
- custom device allocator
- reusable scratch allocator

PyTorch CUDA caching allocator와 MD custom allocator가 함께 존재할 수 있으므로 GPU 메모리 사용량을 추적할 수 있게 해 주세요.

초기 구현에서는 MD 데이터는 MD allocator가, model weight 및 backend 내부 tensor는 backend allocator가 소유해도 됩니다.

단, 두 allocator 사이에서 데이터를 복사하지 않고 raw pointer view를 공유할 수 있어야 합니다.

---

# 8. 모델 capability와 validation

모델을 초기화할 때 다음 capability를 조회할 수 있도록 해 주세요.

```cpp
struct MlipCapabilities {
    bool computes_total_energy;
    bool computes_atomic_energy;
    bool computes_forces;
    bool computes_virial;
    bool computes_stress;

    bool force_from_autograd;
    bool supports_float32;
    bool supports_float64;

    bool supports_cuda;
    bool supports_cpu;

    bool supports_external_neighbor_list;
    bool supports_external_cuda_stream;

    bool requires_cell;
    bool requires_pbc;
};
```

시뮬레이션 시작 시 다음 항목을 검증하고, 조건을 만족하지 않으면 즉시 명확한 오류를 발생시켜 주세요.

- 모델이 CUDA를 지원하는지
- requested precision을 지원하는지
- force 출력을 지원하는지
- virial/stress가 필요한 ensemble에서 해당 출력을 제공하는지
- external neighbor list를 받을 수 있는지
- external CUDA stream을 지원하는지
- species mapping이 올바른지
- MD cutoff가 model cutoff 이상인지
- simulation cell 형식이 모델 요구사항과 맞는지
- periodic/non-periodic 조건을 모델이 지원하는지

성능상 CPU fallback이 발생하는 경우 조용히 진행하지 말고 warning 또는 error로 처리해 주세요.

---

# 9. Species mapping

MD 코드의 atom type과 MLIP의 chemical species 표현을 분리해 주세요.

예:

```cpp
struct SpeciesMapping {
    std::vector<int> md_type_to_atomic_number;
    std::unordered_map<int, int> atomic_number_to_model_index;
};
```

가능한 표현:

```text
MD type ID
→ atomic number
→ model species index
```

예:

```text
MD type 0 → H  → atomic number 1  → model index 0
MD type 1 → O  → atomic number 8  → model index 1
```

모델마다 species ordering이 다를 수 있으므로 config에서 명시적으로 설정할 수 있게 해 주세요.

잘못된 species나 모델이 지원하지 않는 원소가 입력되면 inference 전에 오류를 발생시켜 주세요.

---

# 10. Energy, force, virial 및 stress convention

모든 MLIP adapter가 MD 코어에 반환하는 값의 convention을 통일해 주세요.

권장 canonical convention:

```text
Energy:
    rank local contribution 또는 global total인지 명시

Force:
    F_i = -dE/dr_i

Virial:
    MD 코드 내부 정의로 통일

Stress:
    virial과 volume을 이용해 MD 코어에서 변환하는 것을 우선
```

다음 차이를 adapter에서 변환해 주세요.

- force 부호
- energy 단위
- force 단위
- length 단위
- stress 단위
- virial 부호
- stress tensor ordering
- Voigt notation ordering
- extensive virial과 intensive stress
- per-atom energy와 total energy
- cell derivative 기반 stress
- position derivative 기반 virial

단위 변환은 매 step CPU에서 하지 말고 GPU kernel 또는 모델 출력 후처리 kernel에서 수행해 주세요.

가능한 한 내부 단위계를 하나로 고정해 주세요.

---

# 11. Autograd 기반 force 처리

일부 PyTorch MLIP는 energy를 출력하고 force는 autograd를 통해 계산할 수 있습니다.

이 경우 adapter 내부에서 처리하되 다음을 주의해 주세요.

- positions tensor에 `requires_grad=true` 설정
- inference mode를 무조건 사용하면 안 됨
- gradient graph lifetime 관리
- 매 step graph가 누적되지 않도록 처리
- 이전 step의 gradient를 재사용하지 않도록 처리
- force 계산 후 불필요한 graph 해제
- model parameter gradient는 비활성화
- training mode가 아니라 evaluation mode 사용
- mixed precision 사용 여부를 명시

가능하면 production 경로에서는 force를 직접 반환하거나 scripted force computation을 포함한 모델을 우선 지원해 주세요.

Python callback이나 Python interpreter를 매 step 호출하는 구조는 피하세요.

---

# 12. Backend 구조

다음 역할을 분리해 주세요.

## 12.1 Model adapter

모델별 입력과 출력 의미를 처리합니다.

```text
MACE adapter
NequIP adapter
Allegro adapter
DeePMD adapter
custom model adapter
```

담당 업무:

- species mapping
- neighbor convention 변환
- input dictionary 구성
- output 이름 해석
- energy/force/virial 추출
- model-specific unit 변환
- model-specific validation

## 12.2 Runtime backend

실행 엔진과 tensor wrapping을 처리합니다.

```text
LibTorch backend
ONNX Runtime CUDA backend
TensorRT backend
Native CUDA backend
```

담당 업무:

- model load
- device 설정
- stream 연결
- raw pointer를 tensor/view로 wrapping
- inference 실행
- runtime-specific error handling
- runtime workspace 관리

초기 구현에서 adapter와 backend가 일부 결합되어도 되지만, 공통 인터페이스는 분리 가능한 형태로 작성해 주세요.

---

# 13. Factory 및 등록 방식

MLIP 구현을 쉽게 추가할 수 있도록 factory 또는 registry 패턴을 사용해 주세요.

예:

```cpp
class MlipRegistry {
public:
    using Creator =
        std::function<std::unique_ptr<IMlipPotential>()>;

    static MlipRegistry& instance();

    void register_model(
        const std::string& name,
        Creator creator
    );

    std::unique_ptr<IMlipPotential> create(
        const std::string& name
    ) const;
};
```

등록 예:

```cpp
REGISTER_MLIP("mace", MacePotential);
REGISTER_MLIP("nequip", NequipPotential);
REGISTER_MLIP("deepmd", DeepMdPotential);
REGISTER_MLIP("onnx", OnnxMlipPotential);
```

MD 코어에 다음과 같은 분기문이 증가하지 않도록 해 주세요.

```cpp
if (model == "mace") {
    ...
} else if (model == "nequip") {
    ...
} else if (...) {
    ...
}
```

---

# 14. 설정 파일

다음과 유사한 설정 구조를 지원해 주세요.

```yaml
potential:
  type: mlip

  model:
    adapter: mace
    backend: libtorch
    path: models/mace_model.pt

  execution:
    device: cuda
    precision: float32
    external_neighbor_list: true
    share_cuda_stream: true
    allow_cpu_fallback: false

  species:
    md_type_to_atomic_number:
      0: 1
      1: 8

  units:
    length: angstrom
    energy: eV

  neighbor:
    skin: 0.5
    rebuild_check_interval: 10
```

설정 파싱과 model runtime 구현을 분리해 주세요.

---

# 15. Force accumulation

MLIP force가 MD force buffer에 직접 기록되는 방식과 임시 출력 buffer를 사용하는 방식을 모두 고려해 주세요.

권장 초기 구현:

```text
MLIP output force buffer
    ↓
GPU accumulation kernel
    ↓
MD total force buffer
```

이렇게 하면 다음을 지원하기 쉽습니다.

- classical potential + MLIP 혼합
- 여러 potential force 합산
- bias force
- constraint force
- force scaling
- force validation
- NaN 검사

추후 backend가 안전하게 외부 output buffer를 직접 사용할 수 있다면 zero-copy direct output을 추가해 주세요.

---

# 16. 여러 potential 조합

향후 다음 조합을 지원할 수 있도록 potential interface를 일반화해 주세요.

```text
Total force
=
classical pair force
+
MLIP force
+
long-range force
+
constraint force
+
bias force
```

예:

```cpp
class IPotential {
public:
    virtual ~IPotential() = default;

    virtual void compute(
        const SimulationStateView& state,
        ForceAccumulator& accumulator,
        const ComputeContext& context
    ) = 0;
};
```

MLIP는 `IPotential`의 한 구현이 되도록 구성해도 됩니다.

다만 MLIP 전용 metadata와 capability는 별도 인터페이스 또는 확장 객체로 유지해 주세요.

---

# 17. 오류 처리

다음 상황에서 명시적 오류를 발생시켜 주세요.

- model 파일을 로드할 수 없음
- 모델이 요청 GPU에서 실행되지 않음
- CPU tensor가 출력됨
- 입력과 출력의 device가 다름
- dtype 불일치
- species mapping 누락
- cutoff 불일치
- neighbor index 범위 초과
- ghost atom 정보 부족
- force shape 불일치
- energy 또는 force에 NaN/Inf 발생
- stream/device context 불일치
- 모델 output key 누락
- virial이 필요한데 모델이 제공하지 않음

오류 메시지에는 최소한 다음 정보를 포함해 주세요.

- MPI rank
- GPU device ID
- model adapter
- backend
- atom 수
- edge 수
- dtype
- 오류가 발생한 단계

---

# 18. 로깅 및 프로파일링

다음 구간을 개별적으로 프로파일링할 수 있도록 NVTX range를 추가해 주세요.

```text
MD::Communication::Forward
MD::NeighborBuild
MLIP::InputPreparation
MLIP::Inference
MLIP::OutputProcessing
MD::Communication::ReverseForce
MD::Integration
```

step별로 다음 시간을 선택적으로 수집할 수 있게 해 주세요.

- forward communication
- neighbor build
- adapter conversion
- inference
- force accumulation
- reverse communication
- integration
- total step

또한 다음 메모리 사용량을 기록할 수 있게 해 주세요.

- MD persistent device memory
- neighbor-list capacity
- backend workspace
- model weight memory
- PyTorch caching allocator memory
- temporary high-water mark

프로파일링 기능이 비활성화된 경우 실행 성능에 미치는 영향을 최소화해 주세요.

---

# 19. 테스트 요구사항

## 19.1 단위 테스트

다음 테스트를 추가해 주세요.

- species mapping
- half list → full directed list 변환
- periodic cell shift 처리
- edge vector 계산
- force 부호 변환
- 단위 변환
- stress/virial 변환
- adapter factory 등록
- invalid species 처리
- cutoff validation
- device/dtype validation

## 19.2 단일 GPU 통합 테스트

작은 원자계에서 다음을 검증해 주세요.

- CPU reference와 GPU MLIP energy 비교
- CPU reference와 GPU MLIP force 비교
- finite-difference force 검증
- neighbor rebuild 전후 결과 일치
- stream 공유 경로 정상 동작
- 불필요한 D2H/H2D 복사 없음

## 19.3 MPI 멀티GPU 테스트

다음을 검증해 주세요.

- 1 rank 결과와 다중 rank 결과 일치
- domain boundary 원자의 force 정확성
- ghost force reverse communication 정확성
- atom migration 이후 결과 정확성
- rank별 GPU binding 정확성
- rank마다 동일 모델이 올바른 GPU에 로드되는지 확인

## 19.4 성능 테스트

다음 항목을 측정해 주세요.

- atom당 step time
- edge당 inference time
- neighbor build time
- MPI communication 비율
- model inference 비율
- GPU utilization
- CPU↔GPU transfer 횟수
- step별 allocation 횟수
- synchronization 횟수

production 경로에서 매 step CUDA allocation이 발생하지 않는 것을 검증해 주세요.

---

# 20. 구현 단계

한 번에 전체 코드를 바꾸지 말고 다음 순서로 작업해 주세요.

## Phase 1: 현재 코드 분석

먼저 현재 코드베이스를 분석하여 다음을 정리해 주세요.

- atom 데이터의 실제 소유 위치
- CPU와 GPU의 authoritative copy
- position, force, species의 데이터 layout
- neighbor list 구조
- MPI domain decomposition 구조
- ghost atom 생성 및 갱신 위치
- reverse force communication 위치
- integration kernel 위치
- 현재 potential interface
- GPU stream 관리 방식
- 매 step 발생하는 memory allocation
- 현재 CPU↔GPU 복사 위치
- build system과 optional dependency 구조

분석 결과를 바탕으로 수정 대상 파일 목록과 데이터 흐름을 먼저 제시해 주세요.

## Phase 2: 공통 interface 도입

다음을 추가해 주세요.

- `IMlipPotential`
- `MlipInputView`
- `MlipOutputView`
- `MlipExecutionContext`
- `MlipCapabilities`
- `NeighborRequirements`
- `SpeciesMapping`
- `MlipRegistry`

이 단계에서는 mock MLIP를 사용해도 됩니다.

## Phase 3: Mock GPU MLIP

간단한 CUDA kernel 기반 mock model을 구현해 interface와 데이터 흐름을 검증해 주세요.

예를 들어 neighbor distance 기반 harmonic energy와 force를 계산할 수 있습니다.

목표는 다음을 검증하는 것입니다.

- GPU pointer 전달
- neighbor list 전달
- output force accumulation
- CUDA stream 연결
- MPI ghost force 처리

## Phase 4: 첫 번째 실제 backend

우선순위는 다음과 같습니다.

```text
1. LibTorch/TorchScript CUDA
2. ONNX Runtime CUDA
3. TensorRT
4. 모델별 native backend
```

실제 프로젝트의 dependency와 대상 MLIP에 따라 첫 backend를 선택해 주세요.

## Phase 5: 실제 model adapter

우선 한 개 모델에 대해 adapter를 완성한 뒤 두 번째 모델을 추가하여 공통 interface가 실제로 재사용 가능한지 검증해 주세요.

두 번째 모델을 추가할 때 MD 코어 수정이 발생하지 않아야 합니다.

## Phase 6: 성능 최적화

다음을 제거하거나 최적화해 주세요.

- 매 step allocation
- 중복 neighbor 변환
- 암시적 synchronization
- CPU staging
- 불필요한 dtype conversion
- 불필요한 edge-vector 재계산
- 불필요한 force buffer copy

---

# 21. 코드 작성 규칙

다음 규칙을 지켜 주세요.

1. MD 코어에 MLIP framework header를 포함하지 마세요.
2. PyTorch 관련 코드는 LibTorch backend 또는 adapter 디렉터리에만 위치시켜 주세요.
3. CUDA device pointer의 소유권과 lifetime을 주석과 타입으로 명확히 표현해 주세요.
4. raw pointer를 장기간 저장할 때 buffer 재할당 가능성을 고려해 주세요.
5. CUDA 오류를 모든 kernel launch와 runtime 호출에서 검사해 주세요.
6. MPI 오류도 검사해 주세요.
7. hot path에서 virtual dispatch 비용은 inference 비용에 비해 작지만, 필요하면 step당 한 번만 발생하게 해 주세요.
8. hot path에서 문자열 lookup을 하지 마세요.
9. configuration parsing은 초기화 시점에 완료해 주세요.
10. model output key 탐색도 가능하면 초기화 시 검증하고 index를 캐시해 주세요.
11. 매 step map, vector 또는 tensor 객체를 과도하게 생성하지 마세요.
12. thread safety와 rank별 독립 상태를 보장해 주세요.
13. global singleton에 GPU별 mutable 실행 상태를 저장하지 마세요.
14. 모델 weight는 해당 rank가 담당하는 GPU에 한 번만 로드해 주세요.
15. CUDA Graph 적용 가능성을 막는 구조를 만들지 마세요.

---

# 22. 권장 디렉터리 구조

현재 프로젝트 구조에 맞게 조정하되 다음과 유사하게 분리해 주세요.

```text
src/

  potential/
    potential_interface.*
    potential_manager.*

    mlip/
      mlip_interface.*
      mlip_types.*
      mlip_registry.*
      species_mapping.*
      unit_conversion.*

      adapters/
        mace_adapter.*
        nequip_adapter.*
        deepmd_adapter.*

      backends/
        libtorch_backend.*
        onnx_cuda_backend.*
        tensorrt_backend.*

      cuda/
        edge_conversion.cu
        force_accumulation.cu
        virial_conversion.cu
        validation.cu

tests/
  unit/
  integration/
  mpi/
  performance/
```

---

# 23. 최종 산출물

작업 결과로 다음을 제공해 주세요.

1. 현재 코드 구조 분석
2. 기존 데이터 흐름 다이어그램
3. 변경 후 데이터 흐름 다이어그램
4. 수정 대상 파일 목록
5. 새로 추가되는 interface 설명
6. 실제 코드 변경
7. 단위 테스트와 통합 테스트
8. build option 및 dependency 설정
9. 간단한 설정 파일 예제
10. 단일 GPU 실행 예제
11. MPI 멀티GPU 실행 예제
12. profiling 방법
13. 알려진 제약 사항
14. 다음 MLIP adapter를 추가하는 방법

---

# 24. 완료 조건

다음 조건을 모두 만족해야 작업 완료로 간주합니다.

- MD 코어가 특정 MLIP framework에 의존하지 않음
- 새 MLIP 추가 시 MD 코어 수정이 필요하지 않음
- position과 force가 step마다 CPU로 이동하지 않음
- neighbor list가 GPU에서 MLIP로 전달됨
- MLIP가 MD의 CUDA stream을 사용하거나 명시적으로 동기화됨
- 매 step tensor data copy가 없음
- 매 step CUDA allocation이 없음
- MPI domain boundary에서 force가 정확함
- ghost force reverse communication이 정확함
- 1 GPU와 멀티GPU 결과가 허용 오차 내에서 일치함
- 모델 cutoff와 species mapping이 초기화 단계에서 검증됨
- energy, force, virial의 convention이 문서화됨
- 두 개 이상의 서로 다른 MLIP adapter를 공통 interface로 실행 가능함

---

# 25. 작업 시작 시 Claude Code가 먼저 수행할 일

코드를 바로 수정하기 전에 다음 순서로 진행해 주세요.

2. atom, neighbor list, force, MPI, CUDA stream 관련 코드를 검색합니다.
3. 현재 한 MD step의 실제 호출 경로를 추적합니다.
4. CPU↔GPU 복사와 allocation 위치를 찾습니다.
5. potential 계산 interface를 확인합니다.
6. 위 설계와 현재 구조의 차이를 정리합니다.
7. 최소 침습적인 단계별 리팩터링 계획을 작성합니다.
8. 계획에 따라 작은 단위로 코드를 수정합니다.
9. 각 단계마다 build와 test를 수행합니다.
10. 기존 동작을 유지하면서 MLIP mock backend부터 연결합니다.

기존 코드의 구조와 명명 규칙을 최대한 존중하되, 성능상 문제가 되는 구조는 호환 계층을 둔 뒤 점진적으로 제거해 주세요.

코드를 분석하지 않은 상태에서 존재하지 않는 class, file 또는 API를 가정해 대규모 코드를 생성하지 마세요. 먼저 실제 코드베이스에서 대응되는 구성 요소를 찾고, 확인한 구조를 기준으로 구현해 주세요.
