# YAML 기반 Declarative MLIP Adapter 설계 및 구현 지시서

## 1. 목표

현재 구현 중인 C++/CUDA 기반 MD 코드에서 새로운 MLIP를 가능한 한 코드 재빌드 없이 추가할 수 있도록, 기존 MLIP 연동 구조를 다음 형태로 리팩터링해 주세요.

핵심 목표는 다음과 같습니다.

1. MLIP의 모델별 입출력 매핑을 YAML로 정의할 수 있어야 합니다.
2. 기존 backend와 전처리 연산의 조합만으로 지원 가능한 모델은 YAML 파일 추가만으로 실행할 수 있어야 합니다.
3. PyTorch, ONNX Runtime, TensorRT 등 실행 backend는 사전에 C++로 구현합니다.
4. YAML은 새로운 실행 코드를 정의하는 것이 아니라, 이미 구현된 연산과 backend를 선언적으로 조합해야 합니다.
5. YAML에서 임의 코드를 실행하거나 임의의 CUDA kernel을 정의하는 구조는 사용하지 않습니다.
6. MD 코어는 특정 MLIP 모델이나 프레임워크에 의존하지 않아야 합니다.
7. 모델별 C++ adapter의 수를 최소화하되, 필요한 경우 native plugin을 추가할 수 있어야 합니다.
8. GPU-resident 실행, CUDA stream 공유, MPI ghost atom 처리를 유지해야 합니다.

최종적으로 다음 구조를 구현해 주세요.

```text
MD Core
  ↓
Generic MLIP Runtime
  ├─ PyTorch Backend
  ├─ ONNX Runtime Backend
  ├─ TensorRT Backend
  └─ Native Plugin Backend
  ↓
YAML Model Specification
```

---

# 2. 설계 원칙

## 2.1 YAML이 담당하는 범위

YAML은 다음과 같은 모델별 설정과 데이터 매핑을 정의해야 합니다.

* backend 종류
* 모델 파일 경로
* 실행 device
* precision
* 모델 입력 tensor 이름
* 모델 출력 tensor 이름
* 모델 forward 인자 순서
* MD 데이터와 모델 입력 사이의 매핑
* tensor shape
* tensor dtype
* tensor layout
* species mapping
* neighbor cutoff
* full/half neighbor list
* directed/undirected edge
* self edge 포함 여부
* edge vector와 distance 사용 여부
* 단위 변환
* force 부호 convention
* virial 및 stress convention
* 출력 reduction 방식
* 출력 scatter 및 accumulation 정책
* 모델 capability 선언

YAML 변경만으로 다음 항목을 수정할 수 있어야 합니다.

```text
모델 파일
입력 tensor 이름
출력 tensor 이름
입력 인자 순서
dtype
shape
species mapping
cutoff
neighbor-list convention
단위계
force convention
virial convention
output reduction
```

---

## 2.2 YAML이 담당하지 않는 범위

다음 기능은 YAML만으로 구현하려 하지 말고, 반드시 사전에 C++/CUDA 코드로 구현해 주세요.

* PyTorch, ONNX Runtime, TensorRT loader
* 모델 runtime 초기화
* CUDA device pointer의 tensor wrapping
* CUDA stream 연결
* backend별 inference 실행
* neighbor-list 변환 kernel
* edge vector 계산
* edge distance 계산
* triplet 생성
* quadruplet 생성
* force scatter
* force reduction
* force accumulation
* virial reduction
* ghost atom force 처리
* MPI reverse communication
* CUDA event 및 stream synchronization
* custom CUDA operator 호출
* model-specific native descriptor
* backend별 메모리 allocator 관리

YAML은 이 연산들을 새로 구현하는 수단이 아니라, 이미 등록된 연산들을 선택하고 조합하는 수단이어야 합니다.

---

# 3. 전체 아키텍처

다음 계층으로 구현해 주세요.

```text
MD Core
  ↓
MlipRuntime
  ↓
MlipExecutionPlan
  ↓
Preprocessing Pipeline
  ↓
Backend
  ↓
Model
  ↓
Postprocessing Pipeline
  ↓
MD Force/Energy/Virial Buffers
```

구체적으로는 다음과 같습니다.

```text
MD atom/neighbor data
  ↓
YAML specification parsing
  ↓
initialization-time validation
  ↓
execution plan 생성 및 캐싱
  ↓
GPU preprocessing
  ↓
backend inference
  ↓
GPU postprocessing
  ↓
MD output buffer 반영
```

YAML을 매 MD step마다 해석하면 안 됩니다.

YAML은 초기화 시 한 번만 파싱하고, 실행에 필요한 정보를 고정된 execution plan으로 변환해 주세요.

---

# 4. 권장 YAML 모델 명세

다음과 유사한 YAML 형식을 지원해 주세요.

```yaml
schema_version: 1

model:
  name: example_mlip
  backend: pytorch
  path: models/model.pt
  device: cuda
  precision: float32

backend:
  format: torchscript
  entrypoint: forward
  share_cuda_stream: true
  allow_cpu_fallback: false

inputs:
  positions:
    source: md.positions
    tensor_name: positions
    argument_index: 0
    shape: [nall, 3]
    dtype: float32
    layout: contiguous
    requires_grad: true

  species:
    source: md.species
    tensor_name: species
    argument_index: 1
    shape: [nall]
    dtype: int64
    transform:
      - species_map

  edge_index:
    source: md.neighbor_list
    tensor_name: edge_index
    argument_index: 2
    shape: [2, nedges]
    dtype: int64
    transform:
      - full_directed_edge_list
      - transpose_edge_index

  edge_vectors:
    source: md.edge_vectors
    tensor_name: edge_vectors
    argument_index: 3
    shape: [nedges, 3]
    dtype: float32
    transform:
      - compute_edge_vectors

outputs:
  energy:
    tensor_name: energy
    target: md.energy
    reduction: sum
    scale: 1.0

  forces:
    tensor_name: forces
    target: md.forces
    operation: accumulate
    sign: 1.0
    scale: 1.0

  virial:
    tensor_name: virial
    target: md.virial
    convention: negative_outer_product
    reduction: sum
    scale: 1.0

neighbor:
  cutoff: 5.0
  skin: 0.5
  list: full
  directed: true
  self_edges: false
  include_ghost_atoms: true
  require_cell_shifts: true

species_map:
  input: atomic_symbol
  output: model_index
  values:
    H: 0
    C: 1
    O: 2

units:
  md:
    length: angstrom
    energy: eV
  model:
    length: angstrom
    energy: eV

capabilities:
  energy: true
  forces: true
  virial: true
  stress: false
  external_neighbor_list: true
  external_cuda_stream: true
```

실제 YAML schema는 현재 코드 구조에 맞게 조정할 수 있지만, 문자열 기반의 느슨한 해석보다 명확한 schema와 validation을 사용해 주세요.

---

# 5. Generic MLIP Runtime

다음과 유사한 generic runtime을 구현해 주세요.

```cpp
class MlipRuntime {
public:
    void initialize(
        const MlipModelSpecification& specification,
        const MlipExecutionContext& context
    );

    void compute(
        const MlipInputView& input,
        MlipOutputView& output
    );

    const MlipCapabilities& capabilities() const;
    double cutoff() const;

private:
    std::unique_ptr<IMlipBackend> backend_;
    MlipExecutionPlan execution_plan_;
    MlipCapabilities capabilities_;
};
```

`MlipRuntime`은 특정 모델 이름을 알면 안 됩니다.

예를 들어 다음과 같은 분기를 두지 마세요.

```cpp
if (model_name == "mace") {
    ...
} else if (model_name == "nequip") {
    ...
}
```

대신 YAML 명세를 읽어 generic execution plan을 구성해야 합니다.

---

# 6. Execution Plan

YAML 파싱 결과를 매 step에서 직접 사용하지 말고, 초기화 단계에서 다음과 같은 execution plan으로 변환해 주세요.

```cpp
struct MlipExecutionPlan {
    std::vector<InputBinding> input_bindings;
    std::vector<OutputBinding> output_bindings;

    std::vector<PreprocessOperation> preprocess_operations;
    std::vector<PostprocessOperation> postprocess_operations;

    NeighborRequirements neighbor_requirements;
    SpeciesMapping species_mapping;
    UnitConversion unit_conversion;

    bool requires_autograd;
    bool share_cuda_stream;
};
```

Execution plan에는 다음 정보가 미리 해결되어 있어야 합니다.

* backend input index
* backend output index
* tensor 이름
* source buffer
* target buffer
* shape resolver
* dtype
* stride
* transform 목록
* output reduction 방식
* 단위 변환 계수
* force 부호
* virial convention
* scratch buffer 요구량
* operation 실행 순서

hot path에서는 YAML key 탐색, 문자열 비교, map lookup을 하지 마세요.

초기화 시 문자열 기반 명세를 enum, index, function pointer 또는 precompiled operation object로 변환해 주세요.

---

# 7. Backend 인터페이스

다음과 같은 backend 추상화를 구현해 주세요.

```cpp
class IMlipBackend {
public:
    virtual ~IMlipBackend() = default;

    virtual void initialize(
        const BackendConfiguration& config,
        const MlipExecutionContext& context
    ) = 0;

    virtual BackendModelMetadata inspect_model() = 0;

    virtual void execute(
        std::span<const BackendTensorView> inputs,
        std::span<BackendTensorView> outputs,
        cudaStream_t stream
    ) = 0;

    virtual void synchronize() = 0;
};
```

최소한 다음 backend를 등록 가능한 구조로 설계해 주세요.

```text
pytorch
onnx
tensorrt
native_plugin
```

backend registry 예:

```cpp
REGISTER_MLIP_BACKEND("pytorch", PyTorchBackend);
REGISTER_MLIP_BACKEND("onnx", OnnxRuntimeBackend);
REGISTER_MLIP_BACKEND("tensorrt", TensorRtBackend);
REGISTER_MLIP_BACKEND("native_plugin", NativePluginBackend);
```

MD 코어는 이 backend들의 header에 의존하지 않아야 합니다.

---

# 8. PyTorch Backend 요구사항

PyTorch backend는 다음을 지원하도록 구현해 주세요.

* TorchScript 또는 지원 가능한 export format 로드
* model을 지정 GPU에 한 번만 로드
* `eval()` 적용
* model parameter gradient 비활성화
* 외부 CUDA stream 사용
* MD GPU pointer를 zero-copy tensor view로 wrapping
* positional argument 또는 dictionary input 지원
* named output 또는 tuple output 지원
* 선택적 autograd force 계산
* CPU fallback 감지
* output device 및 dtype 검증
* 매 step model 재로드 금지
* 매 step CUDA stream 생성 금지

외부 CUDA stream을 사용할 때는 MD stream을 PyTorch stream으로 연결해 주세요.

개념적으로 다음과 같은 구조를 사용합니다.

```cpp
at::cuda::CUDAStream torch_stream =
    at::cuda::getStreamFromExternal(md_stream, device_id);

at::cuda::CUDAStreamGuard stream_guard(torch_stream);
```

설치된 LibTorch 버전의 실제 API를 확인하여 구현해 주세요.

---

# 9. 입력 Source Registry

YAML의 `source` 필드는 임의 문자열을 직접 해석하지 말고, 사전에 등록된 MD 데이터 source만 참조할 수 있도록 해 주세요.

예:

```text
md.positions
md.velocities
md.species
md.atomic_numbers
md.cell
md.pbc
md.neighbor_list
md.edge_index
md.edge_vectors
md.edge_distances
md.cell_shifts
md.local_mask
md.ghost_mask
md.batch
```

각 source는 다음 정보를 반환해야 합니다.

```cpp
struct DataSourceDescriptor {
    ScalarType dtype;
    int rank;
    ShapeResolver shape_resolver;
    DevicePointerResolver pointer_resolver;
};
```

등록되지 않은 source를 YAML에서 참조하면 초기화 단계에서 오류를 발생시켜 주세요.

YAML이 임의 메모리 주소나 객체 필드에 접근할 수 있게 해서는 안 됩니다.

---

# 10. 사전 컴파일된 전처리 연산

YAML의 `transform`은 임의 함수 이름을 실행하는 방식이 아니라, 사전에 등록된 연산만 선택할 수 있어야 합니다.

초기 지원 연산 예:

```text
identity
cast_float32
cast_float64
cast_int32
cast_int64
species_map
half_to_full_neighbor_list
full_directed_edge_list
remove_self_edges
add_reverse_edges
transpose_edge_index
compute_edge_vectors
compute_edge_distances
compute_cell_shifts
local_atom_mask
ghost_atom_mask
contiguous_copy
unit_scale
```

연산 registry 예:

```cpp
REGISTER_MLIP_PREPROCESS_OP(
    "compute_edge_vectors",
    ComputeEdgeVectorsOperation
);
```

각 연산은 다음을 명시해야 합니다.

* 입력 dtype
* 출력 dtype
* 입력 shape
* 출력 shape
* 필요한 scratch memory
* in-place 가능 여부
* GPU 지원 여부
* CUDA stream 지원 여부
* graph capture 가능 여부

---

# 11. 후처리 연산

출력 처리도 사전 등록된 operation으로 구현해 주세요.

초기 지원 연산 예:

```text
identity
scale
negate
sum
mean
scatter_add
accumulate_force
local_atom_slice
ghost_force_extract
virial_sign_conversion
stress_to_virial
virial_to_stress
voigt_reorder
unit_conversion
nan_inf_check
```

예:

```yaml
outputs:
  forces:
    tensor_name: forces
    target: md.forces
    operations:
      - local_and_ghost_force_view
      - unit_conversion
      - accumulate_force
```

출력 처리 또한 매 step CPU에 데이터를 가져오지 않고 GPU에서 수행해 주세요.

---

# 12. Shape 표현식

YAML의 shape에는 제한된 symbol만 허용해 주세요.

예:

```text
nlocal
nghost
nall
nedges
ntriplets
3
6
9
```

허용 예:

```yaml
shape: [nall, 3]
shape: [2, nedges]
shape: [nlocal]
```

임의 수식 실행이나 범용 expression evaluator는 사용하지 마세요.

필요하다면 제한된 shape expression만 지원합니다.

```text
nall = nlocal + nghost
```

shape는 초기화 및 step 실행 시점에 실제 buffer 크기와 검증해야 합니다.

---

# 13. Tensor Binding

입력 binding은 다음 정보를 가져야 합니다.

```cpp
struct InputBinding {
    int backend_input_index;
    std::string tensor_name;

    DataSourceId source;
    ScalarType target_dtype;
    TensorLayout layout;

    ShapeDescriptor shape;
    std::vector<PreprocessOperationId> operations;

    bool requires_grad;
};
```

출력 binding은 다음 정보를 가져야 합니다.

```cpp
struct OutputBinding {
    int backend_output_index;
    std::string tensor_name;

    OutputTargetId target;
    ScalarType expected_dtype;
    ShapeDescriptor expected_shape;

    std::vector<PostprocessOperationId> operations;
};
```

문자열은 디버깅과 오류 메시지용으로만 유지하고, hot path에서는 정수 ID와 캐시된 operation을 사용해 주세요.

---

# 14. Forward 호출 방식

다음 forward 호출 형식을 지원할 수 있도록 설계해 주세요.

## Positional arguments

```yaml
backend:
  call_style: positional
```

```yaml
inputs:
  positions:
    argument_index: 0
  species:
    argument_index: 1
  edge_index:
    argument_index: 2
```

## Named dictionary

```yaml
backend:
  call_style: dictionary
```

```yaml
inputs:
  positions:
    tensor_name: pos
  species:
    tensor_name: atomic_numbers
```

## Tuple output

```yaml
outputs:
  energy:
    output_index: 0
  forces:
    output_index: 1
```

## Named dictionary output

```yaml
outputs:
  energy:
    tensor_name: total_energy
  forces:
    tensor_name: atomic_forces
```

모델 metadata를 초기화 시 inspect하여 YAML 명세와 실제 모델 입출력이 맞는지 가능한 범위에서 검증해 주세요.

---

# 15. YAML만으로 추가 가능한 모델의 조건

새로운 모델이 다음 조건을 만족하면 재빌드 없이 YAML 추가만으로 지원되어야 합니다.

* 이미 구현된 backend를 사용
* 모델 파일 형식이 해당 backend에서 지원됨
* 입력이 등록된 MD source의 조합
* 입력 전처리가 등록된 operation의 조합
* 출력이 energy, force, virial 또는 stress 중 하나
* 출력 처리가 등록된 operation의 조합
* 기존 neighbor representation으로 변환 가능
* 기존 CUDA stream 공유 방식 사용 가능
* 기존 ghost atom 처리 방식과 호환
* custom operator가 필요하지 않음

예:

```text
PyTorch backend
+
positions
+
species
+
directed edge index
+
edge vectors
+
energy/force output
```

위와 같은 모델은 YAML만 추가하여 사용할 수 있어야 합니다.

---

# 16. 새 코드 또는 plugin이 필요한 조건

다음 요구사항이 있는 모델은 YAML만으로 지원하려 하지 마세요.

* 새로운 graph representation
* triplet graph
* quadruplet graph
* angle list
* dihedral list
* spherical harmonics
* radial basis descriptor
* 모델 고유 descriptor
* 특수 neighbor ordering
* custom PyTorch operator
* custom CUDA operator
* custom C++ class가 포함된 TorchScript 모델
* backend가 지원하지 않는 dynamic control flow
* 특수 force decomposition
* 새로운 ghost communication 방식
* 모델 내부 distributed inference
* 새로운 domain decomposition 규칙
* 비표준 virial 계산
* 임의 Python callback

이 경우 다음 중 하나를 추가해야 합니다.

```text
새 preprocessing operation
새 postprocessing operation
새 backend
새 native plugin
새 communication strategy
```

새 기능을 구현한 뒤에는 YAML에서 해당 기능을 선택할 수 있도록 registry에 등록해 주세요.

---

# 17. Native Plugin 구조

YAML capability를 벗어나는 모델을 위해 native plugin backend를 제공해 주세요.

예:

```cpp
class INativeMlipPlugin {
public:
    virtual ~INativeMlipPlugin() = default;

    virtual void initialize(
        const PluginConfiguration& config,
        const MlipExecutionContext& context
    ) = 0;

    virtual void compute(
        const MlipInputView& input,
        MlipOutputView& output
    ) = 0;

    virtual MlipCapabilities capabilities() const = 0;
};
```

가능하면 shared library 형태로 로드할 수 있도록 설계해 주세요.

예:

```yaml
model:
  backend: native_plugin

plugin:
  library: libcustom_mlip.so
  factory_symbol: create_custom_mlip
```

plugin ABI는 versioning을 포함해야 합니다.

```cpp
constexpr int MLIP_PLUGIN_ABI_VERSION = 1;
```

ABI가 맞지 않으면 명확한 오류를 발생시켜 주세요.

---

# 18. Neighbor-list 명세

YAML에서 neighbor 요구사항을 선언할 수 있도록 해 주세요.

```yaml
neighbor:
  cutoff: 5.0
  skin: 0.5
  list: full
  directed: true
  self_edges: false
  include_ghost_atoms: true
  require_edge_vectors: true
  require_distances: false
  require_cell_shifts: true
```

지원할 enum 예:

```text
list:
  half
  full

directed:
  true
  false
```

초기화 시 다음을 검증해 주세요.

* model cutoff와 MD cutoff
* skin 적용 여부
* half/full 변환 지원 여부
* directed edge 생성 지원 여부
* ghost atom 포함 여부
* cell shift 지원 여부
* edge vector 생성 가능 여부
* periodic boundary 조건 호환 여부

---

# 19. Species Mapping

YAML에서 다음 매핑 방식을 지원해 주세요.

```text
MD type → atomic number
MD type → model index
atomic number → model index
atomic symbol → model index
```

예:

```yaml
species_map:
  input: md_type
  output: model_index
  values:
    0: 0
    1: 1
    2: 2
```

또는:

```yaml
species_map:
  input: atomic_symbol
  output: model_index
  values:
    H: 0
    C: 1
    O: 2
```

매핑은 초기화 시 GPU lookup table로 변환해 주세요.

매 step `std::unordered_map`을 사용하지 마세요.

지원하지 않는 species가 발견되면 inference 전에 오류를 발생시켜 주세요.

---

# 20. 단위 변환

YAML에서 MD와 모델의 단위계를 선언할 수 있도록 해 주세요.

```yaml
units:
  md:
    length: angstrom
    energy: eV
  model:
    length: nm
    energy: kJ_per_mol
```

초기화 시 다음 계수를 계산해 주세요.

```text
position scale
energy scale
force scale
virial scale
stress scale
```

계산된 계수는 execution plan에 숫자로 저장하고, 매 step 문자열 단위 해석을 하지 마세요.

단위 변환은 GPU 전처리 또는 후처리 kernel에서 수행해 주세요.

---

# 21. Force 및 Virial Convention

다음과 같은 convention을 YAML로 선언할 수 있게 해 주세요.

```yaml
outputs:
  forces:
    tensor_name: forces
    sign: 1.0
    operation: accumulate

  virial:
    tensor_name: virial
    convention: negative_outer_product
    ordering: xx_yy_zz_xy_xz_yz
```

지원할 convention 예:

```text
force:
  force
  negative_gradient
  positive_gradient

virial:
  virial_tensor
  negative_virial_tensor
  stress_tensor
  negative_stress_tensor
  position_force_outer_product

ordering:
  matrix_3x3
  xx_yy_zz_xy_xz_yz
  xx_yy_zz_yz_xz_xy
```

내부 MD convention으로 변환하는 코드는 사전 컴파일된 postprocessing operation으로 구현해 주세요.

---

# 22. Autograd Force

모델이 energy만 출력하고 force를 autograd로 계산해야 하는 경우 YAML에서 선언할 수 있게 해 주세요.

```yaml
autograd:
  enabled: true
  energy_output: energy
  differentiate_wrt: positions
  force_sign: -1.0
  create_graph: false
  retain_graph: false
```

PyTorch backend가 이 기능을 구현해야 합니다.

다음을 지켜 주세요.

* positions tensor에만 필요한 gradient 설정
* model parameter gradient 비활성화
* graph가 step 간 누적되지 않도록 처리
* inference mode를 잘못 사용하지 않도록 처리
* force 계산 완료 후 graph 해제
* CUDA stream 유지
* output force가 GPU에 남도록 처리

---

# 23. GPU Resident 실행

YAML 기반 구조를 도입하더라도 다음 데이터 흐름을 유지해야 합니다.

```text
GPU positions
  ↓
GPU neighbor list
  ↓
GPU preprocessing
  ↓
GPU tensor view
  ↓
GPU model inference
  ↓
GPU postprocessing
  ↓
GPU force accumulation
  ↓
GPU integration
```

다음 동작이 매 step 발생하지 않도록 해 주세요.

* YAML parsing
* model load
* tensor metadata 재구성
* CUDA stream 생성
* CUDA allocation
* CPU↔GPU 입력 복사
* CPU output reduction
* 문자열 lookup
* operation registry lookup
* backend input 이름 탐색
* backend output 이름 탐색

---

# 24. MPI 및 Ghost Atom

YAML은 ghost atom 처리 방식을 새로 구현하지 않습니다.

다음 기능은 MD 코어 또는 사전 구현된 communication layer가 담당해야 합니다.

* ghost position forward communication
* ghost species forward communication
* local/ghost index 관리
* ghost force reverse communication
* owner rank force accumulation
* atom migration
* CUDA-aware MPI
* pinned-host fallback

YAML은 모델이 ghost atom을 필요로 하는지만 선언할 수 있습니다.

```yaml
neighbor:
  include_ghost_atoms: true

outputs:
  forces:
    includes_ghost_atoms: true
```

모델 출력에 ghost force가 포함되는 경우 기존 reverse communication 경로를 사용해 owner rank에 합산해 주세요.

---

# 25. Configuration Schema 검증

YAML schema version을 반드시 포함해 주세요.

```yaml
schema_version: 1
```

다음 검증을 초기화 단계에서 수행해 주세요.

* 필수 key 존재 여부
* 알 수 없는 key
* 잘못된 enum
* backend 등록 여부
* preprocessing operation 등록 여부
* postprocessing operation 등록 여부
* source 등록 여부
* target 등록 여부
* dtype 호환 여부
* shape 호환 여부
* model input/output 존재 여부
* cutoff 유효성
* species mapping 완전성
* unit 지원 여부
* capability 충족 여부
* CUDA 지원 여부
* stream 공유 지원 여부

오타가 있는 key를 조용히 무시하지 마세요.

예를 들어 `edge_indx`를 `edge_index`로 추정해서 실행하지 말고 명확한 오류를 발생시켜 주세요.

---

# 26. 보안 및 안정성

YAML 명세에서 다음 기능을 허용하지 마세요.

* 임의 Python 코드 실행
* shell command 실행
* 임의 shared library 로드
* 허용되지 않은 경로 접근
* 임의 CUDA kernel source compilation
* 임의 메모리 주소 지정
* arbitrary expression evaluation
* 임의 C++ symbol 호출

native plugin loading은 명시적으로 활성화된 경우에만 허용하고, 허용된 plugin 경로 또는 allowlist를 적용할 수 있게 해 주세요.

---

# 27. 오류 메시지

YAML 또는 runtime 오류에는 다음 정보를 포함해 주세요.

* YAML 파일 경로
* schema version
* model name
* backend
* MPI rank
* GPU device ID
* 입력 또는 출력 이름
* source 또는 target
* operation 이름
* expected dtype
* actual dtype
* expected shape
* actual shape
* atom 수
* edge 수
* 오류 발생 단계

예:

```text
MLIP configuration error:
  file: models/mace.yaml
  input: edge_index
  operation: full_directed_edge_list
  expected source: md.neighbor_list
  actual source: md.edge_vectors
```

---

# 28. 캐싱

다음 항목은 초기화 시 해결하고 캐시해 주세요.

* backend creator
* backend input index
* backend output index
* source resolver
* target resolver
* shape resolver
* dtype conversion
* preprocessing operation sequence
* postprocessing operation sequence
* unit conversion factor
* species lookup table
* neighbor requirement
* scratch buffer size

동일한 YAML과 모델 파일을 여러 rank가 사용하는 경우에도 각 rank는 자신의 GPU context와 model instance를 독립적으로 가져야 합니다.

---

# 29. 메모리 관리

전처리와 후처리에 필요한 임시 buffer는 persistent scratch buffer로 관리해 주세요.

```cpp
class MlipScratchAllocator {
public:
    void reserve(std::size_t bytes);
    void* allocate_view(std::size_t bytes, std::size_t alignment);
    void reset_step();
};
```

`reset_step()`은 실제 메모리를 해제하지 않고 offset만 초기화해야 합니다.

필요한 최대 scratch 크기는 execution plan을 분석하여 초기화 단계에서 계산하거나, capacity 기반으로 증가시키세요.

매 step `cudaMalloc` 또는 `cudaFree`가 발생하지 않도록 해 주세요.

---

# 30. Operation Fusion

가능하면 연속된 단순 operation을 하나의 CUDA kernel로 합칠 수 있도록 설계해 주세요.

예:

```text
species_map
+
int32 → int64 cast
```

또는:

```text
force sign conversion
+
unit scaling
+
force accumulation
```

초기 구현에서 반드시 fusion할 필요는 없지만, execution plan이 향후 fusion을 적용할 수 있는 구조여야 합니다.

---

# 31. CUDA Graph 호환성

다음 조건을 만족하여 향후 CUDA Graph를 적용할 수 있도록 해 주세요.

* execution operation 순서가 고정됨
* 매 step allocation 없음
* stream이 고정됨
* host synchronization 최소화
* tensor pointer가 가능한 한 안정적임
* backend가 graph capture를 지원하는지 capability로 확인 가능
* dynamic shape 범위가 capacity 안에서 관리됨

YAML 명세 자체가 CUDA Graph 적용을 방해해서는 안 됩니다.

---

# 32. 권장 코드 구조

현재 repository 구조에 맞게 조정하되, 다음과 유사하게 분리해 주세요.

```text
src/
  mlip/
    runtime/
      mlip_runtime.*
      execution_plan.*
      model_specification.*
      schema_validator.*

    config/
      yaml_parser.*
      schema_v1.*
      unit_parser.*
      shape_parser.*

    backend/
      backend_interface.*
      backend_registry.*
      pytorch_backend.*
      onnx_backend.*
      tensorrt_backend.*
      native_plugin_backend.*

    binding/
      input_binding.*
      output_binding.*
      source_registry.*
      target_registry.*

    operations/
      operation_registry.*
      preprocess_operation.*
      postprocess_operation.*

      cuda/
        neighbor_conversion.cu
        edge_vector.cu
        species_mapping.cu
        dtype_conversion.cu
        force_postprocess.cu
        virial_postprocess.cu

    memory/
      scratch_allocator.*
      tensor_view.*

    plugin/
      plugin_interface.*
      plugin_loader.*
      plugin_abi.*

  mpi/
    ghost_exchange.*
    force_reverse_exchange.*

configs/
  mlip/
    examples/
      pytorch_graph_model.yaml
      onnx_force_model.yaml
```

---

# 33. 구현 단계

## Phase 1: 현재 코드 분석

먼저 다음을 분석해 주세요.

* 현재 potential interface
* atom 데이터 구조
* neighbor-list 구조
* GPU memory ownership
* CUDA stream 관리
* MPI domain decomposition
* ghost atom 처리
* force accumulation
* 모델 연동 코드가 이미 있는지
* YAML parser 또는 configuration system
* build system
* optional dependency 관리

분석 후 수정 대상 파일과 단계별 계획을 제시해 주세요.

---

## Phase 2: YAML Schema와 Parser

다음을 구현해 주세요.

* schema version
* strict validation
* typed enum parsing
* unknown-key rejection
* model specification 객체
* 명확한 오류 메시지

이 단계에서는 runtime 실행 없이 YAML을 C++ 객체로 변환하는 테스트를 먼저 작성해 주세요.

---

## Phase 3: Registry

다음 registry를 구현해 주세요.

* backend registry
* data source registry
* output target registry
* preprocessing operation registry
* postprocessing operation registry

등록되지 않은 항목은 초기화 단계에서 오류를 발생시켜 주세요.

---

## Phase 4: Execution Plan

YAML specification을 hot-path 친화적인 execution plan으로 컴파일해 주세요.

다음을 제거해야 합니다.

* step별 문자열 검색
* step별 map lookup
* step별 YAML node 접근
* step별 동적 operation 생성

---

## Phase 5: Mock Backend

실제 MLIP를 연결하기 전에 mock backend를 구현해 주세요.

예:

```text
positions
+
edge_index
→
harmonic energy/force
```

이를 이용해 다음을 검증합니다.

* YAML input binding
* neighbor transform
* GPU preprocessing
* output binding
* force accumulation
* CUDA stream
* ghost force 처리

---

## Phase 6: PyTorch Backend

다음을 구현해 주세요.

* model load
* GPU binding
* external CUDA stream
* raw pointer tensor wrapping
* positional input
* dictionary input
* tuple output
* dictionary output
* optional autograd
* device/dtype validation

---

## Phase 7: 두 개 이상의 YAML 모델 검증

서로 다른 입출력 형식을 가진 두 모델을 YAML만 변경하여 실행해 주세요.

예:

```text
모델 A:
positions + species + edge_index
→ energy + forces

모델 B:
positions + atomic_numbers + edge_vectors
→ atomic_energy
→ reduction 및 autograd force
```

두 번째 모델 추가 시 MD 코어 또는 PyTorch backend 코드가 수정되지 않아야 합니다.

---

## Phase 8: Native Plugin 검증

기존 declarative capability로 표현할 수 없는 mock 모델을 native plugin으로 연결해 주세요.

이를 통해 다음 경계를 명확히 검증합니다.

```text
YAML로 가능한 모델
vs.
새 plugin이 필요한 모델
```

---

# 34. 테스트 요구사항

## YAML Parser 테스트

* 정상 YAML
* 필수 key 누락
* unknown key
* 잘못된 enum
* 지원하지 않는 schema version
* 중복 input index
* 중복 output index
* 잘못된 shape symbol
* 등록되지 않은 source
* 등록되지 않은 operation
* 등록되지 않은 backend

## Binding 테스트

* position binding
* species mapping
* edge index binding
* named input
* positional input
* tuple output
* dictionary output
* dtype conversion
* shape validation

## Neighbor 테스트

* half to full 변환
* reverse edge 생성
* self edge 제거
* periodic shift
* edge vector
* edge distance
* ghost atom 포함

## Output 테스트

* force sign
* force scale
* force accumulate
* local atom slice
* ghost force 처리
* energy sum
* virial ordering
* stress-to-virial 변환

## GPU 테스트

* 입력이 GPU에 유지되는지
* 출력이 GPU에 유지되는지
* 매 step allocation이 없는지
* 매 step D2H/H2D가 없는지
* 외부 CUDA stream이 사용되는지
* 불필요한 synchronization이 없는지

## MPI 테스트

* single-rank와 multi-rank 결과 비교
* domain boundary force
* ghost reverse communication
* atom migration 이후 결과
* rank별 GPU binding

---

# 35. 성능 측정

다음 구간에 NVTX range를 추가해 주세요.

```text
MLIP::BindInputs
MLIP::Preprocess
MLIP::BackendExecute
MLIP::Autograd
MLIP::Postprocess
MLIP::ForceAccumulate
```

다음을 측정할 수 있게 해 주세요.

* input binding 시간
* preprocessing 시간
* inference 시간
* autograd 시간
* postprocessing 시간
* force accumulation 시간
* step별 allocation 횟수
* CPU↔GPU 전송 횟수
* synchronization 횟수
* scratch memory high-water mark

---

# 36. 완료 조건

다음 조건을 모두 만족해야 완료로 간주합니다.

* YAML이 모델별 입출력 mapping을 정의함
* YAML이 backend를 선택함
* YAML이 neighbor convention을 정의함
* YAML이 species mapping을 정의함
* YAML이 단위 및 force/virial convention을 정의함
* YAML은 초기화 시 한 번만 파싱됨
* hot path에 YAML node 접근이 없음
* hot path에 문자열 lookup이 없음
* 기존 capability 내의 새 모델은 YAML만 추가하여 실행 가능
* 새로운 backend 또는 descriptor가 필요한 모델은 명확히 plugin 경로를 사용함
* MD 코어가 PyTorch, ONNX, TensorRT에 의존하지 않음
* position, neighbor, force가 GPU에 유지됨
* 매 step CPU↔GPU 복사가 없음
* 매 step CUDA allocation이 없음
* MD CUDA stream과 backend 실행이 올바르게 연결됨
* ghost atom force가 올바르게 owner rank에 합산됨
* 지원하지 않는 YAML 구성이 초기화 단계에서 명확히 거부됨
* 두 개 이상의 서로 다른 모델을 YAML 변경만으로 실행함

---

# 37. 작업 시 주의사항

완전한 범용 YAML 프로그래밍 언어를 만들지 마세요.

이 설계의 목적은 다음입니다.

```text
YAML
=
이미 구현된 backend와 연산을 선택하고 조합하는 선언적 명세
```

다음과 같이 설계해서는 안 됩니다.

```text
YAML
=
새로운 모델 실행 로직과 CUDA 연산을 자유롭게 구현하는 언어
```

기능이 부족할 경우 YAML 표현식을 복잡하게 확장하기보다, 명확한 C++/CUDA operation 또는 native plugin을 추가해 주세요.

가장 현실적이고 유지보수 가능한 구조는 다음입니다.

```text
YAML 기반 declarative adapter
+
소수의 사전 컴파일된 backend
+
등록된 GPU preprocessing/postprocessing operation
+
예외적인 모델을 위한 native plugin
```

---

# 38. Claude Code 작업 시작 지시

코드를 바로 수정하기 전에 다음 순서로 진행해 주세요.

1. repository 구조를 조사합니다.
2. 현재 potential 호출 경로를 추적합니다.
3. atom, neighbor, force buffer의 ownership을 확인합니다.
4. CUDA stream과 MPI communication 구조를 확인합니다.
5. 기존 configuration parser를 확인합니다.
6. YAML로 선언 가능한 부분과 C++ 구현이 필요한 부분을 구분합니다.
7. 최소한의 schema 초안을 작성합니다.
8. 수정 파일 목록과 단계별 구현 계획을 제시합니다.
9. parser와 execution plan부터 작은 단위로 구현합니다.
10. mock backend로 전체 경로를 검증합니다.
11. 이후 실제 backend를 연결합니다.
12. 각 단계마다 build와 test를 수행합니다.

현재 코드에 존재하지 않는 class, API 또는 파일을 임의로 가정하여 대규모 코드를 먼저 작성하지 마세요.

반드시 실제 코드 구조를 분석한 뒤, 기존 구조를 최대한 유지하면서 점진적으로 리팩터링해 주세요.
