## Code Review: spark-rapids-jni — (protobuf-jni-3b1-3b2-nested-scalar)

**📊 Found across 12 files:**
- 🔴 1 must-fix
- 🟡 10 should-fix
- 🟢 22 suggestions

---

### 📄 `src/main/cpp/src/SparkResourceAdaptorJni.cpp`

#### 1. [🟢 SUGGESTION] The async H2D copy at line 1029 uses `h_scan_descs` (a `std::vector` / `cudf::detail::host_vector`, pageable host memory) as the source

**Line:** 2193
**Issue:** The async H2D copy at line 1029 uses `h_scan_descs` (a `std::vector` / `cudf::detail::host_vector`, pageable host memory) as the source. The copy is enqueued on `stream`, but the host vector goes out of scope at the end of the enclosing `if (!h_scan_descs.empty())` block (line 1060) without a prior `stream.synchronize()`. Stream ordering guarantees the kernel reads the destination after the copy, but it does not guarantee the host source stays alive until the (possibly truly-async) copy drains. For pageable memory `cudaMemcpyAsync` H2D is frequently synchronous in practice, which would make this safe today, but that is implementation-defined. Is the lifetime guaranteed across all CUDA versions here? This pattern recurs in the file (e.g. line 1044) and is primarily a memory-stream concern; flagging only as a defensive note.
**Confidence:** 🟢❓ POSSIBLE
**Code:**
```cpp
|
CUDF_CUDA_TRY(cudaMemcpyAsync(d_scan_descs.data(),
                              h_scan_descs.data(),
                              h_scan_descs.size() * sizeof(h_scan_descs[0]),
                              cudaMemcpyHostToDevice,
                              stream.value()));
```
**Fix:** If the host buffer can be destroyed before the copy completes, consider staging through pinned memory (e.g. `cudf::detail::make_pinned_vector_async`) or copying via a typed helper that documents the lifetime contract. Defer to the memory-stream reviewer for the authoritative call.
**Diff:**
```diff
|
- CUDF_CUDA_TRY(cudaMemcpyAsync(d_scan_descs.data(),
-                               h_scan_descs.data(),
-                               h_scan_descs.size() * sizeof(h_scan_descs[0]),
-                               cudaMemcpyHostToDevice,
-                               stream.value()));
+ // Stage through pinned host memory so the source outlives the async copy without a sync.
+ auto pinned_descs = cudf::detail::make_pinned_vector_async<repeated_field_scan_desc>(
+   h_scan_descs.size(), stream);
+ std::copy(h_scan_descs.begin(), h_scan_descs.end(), pinned_descs.begin());
+ CUDF_CUDA_TRY(cudaMemcpyAsync(d_scan_descs.data(),
+                               pinned_descs.data(),
+                               h_scan_descs.size() * sizeof(h_scan_descs[0]),
+                               cudaMemcpyHostToDevice,
+                               stream.value()));
```
**Reference:** cudf.md memory-stream section: "NOT safe: std::vector as source for async copy without stream.synchronize() before destruction (pageable memory is not stream-ordered)". Verified the code at /home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/protobuf/protobuf.cu:1026-1060 — no stream.synchronize() before h_scan_descs leaves scope.
*(Reviewer: correctness)*

#### 2. [🟡 SHOULD FIX] The new `decode_charset` signature defaults its MR parameter to `rmm::mr::get_current_device_resource_ref()`, while it defaults the stream to the cudf wrapper `cudf::get_default_stream()` on the line right above

**Line:** 2200
**Issue:** The new `decode_charset` signature defaults its MR parameter to `rmm::mr::get_current_device_resource_ref()`, while it defaults the stream to the cudf wrapper `cudf::get_default_stream()` on the line right above. The sibling new public function added in this same change set — `parse_timestamp_strings_with_format` in `/home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/cast_string.hpp:202` — uses `cudf::get_current_device_resource_ref()`. The project conventions (Section memory-stream) state `cudf::get_current_device_resource_ref()` should be used. Could you align the MR default with the cudf wrapper to keep the new APIs consistent with each other? That also lets you drop the extra `<rmm/mr/per_device_resource.hpp>` include at line 25, since `<cudf/utilities/memory_resource.hpp>` (already included at line 22) provides `cudf::get_current_device_resource_ref()`.
**Confidence:** 🟣❗ CERTAIN
**Code:**
```cpp
|
rmm::cuda_stream_view stream      = cudf::get_default_stream(),
rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());
```
**Fix:** Change the MR default to `cudf::get_current_device_resource_ref()` and remove the now-unneeded `#include <rmm/mr/per_device_resource.hpp>`.
**Diff:**
```diff
|
--- a/src/main/cpp/src/charset_decode.hpp
+++ b/src/main/cpp/src/charset_decode.hpp
@@
 #include <rmm/cuda_stream_view.hpp>
-#include <rmm/mr/per_device_resource.hpp>
@@
   rmm::cuda_stream_view stream      = cudf::get_default_stream(),
-  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());
+  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
```
**Reference:** Sibling new API at /home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/cast_string.hpp:202 uses `cudf::get_current_device_resource_ref()`; conventions.md Section memory-stream ("`rmm::mr::get_current_device_resource_ref()` is DEPRECATED — MUST use `cudf::get_current_device_resource_ref()`"). Note: I verified at /mnt/nvme/home/haoyangl/cc_exp/spark-rapids-jni/target/libcudf-install/include/rmm/mr/per_device_resource.hpp:187 that the rmm function carries no `[[deprecated]]` attribute in the pinned rmm version, and that this `rmm::mr::` form is used widely elsewhere in the existing codebase (e.g. cast_string.hpp:82-143, row_conversion.hpp:34-52). Hence this is a consistency concern (SHOULD FIX), not a compiler-enforced deprecation.
*(Reviewer: architecture)*

#### 3. [🔴 MUST FIX] The MR default parameter uses the deprecated `rmm::mr::get_current_device_resource_ref()` instead of `cudf::get_current_device_resource_ref()`

**Line:** 2220
**Issue:** The MR default parameter uses the deprecated `rmm::mr::get_current_device_resource_ref()` instead of `cudf::get_current_device_resource_ref()`. The cudf and spark-rapids-jni memory-stream conventions explicitly mark this a blocking MUST FIX: "rmm::mr::get_current_device_resource_ref() is DEPRECATED — MUST use cudf::get_current_device_resource_ref()". The `<cudf/utilities/memory_resource.hpp>` header is already included at line 22, so the correct symbol is already available. Additionally, `<rmm/mr/per_device_resource.hpp>` (line 25) is only needed to provide `rmm::mr::get_current_device_resource_ref()`; once that call is replaced, this include should be removed to keep the header lean.
**Confidence:** 🟣❗ CERTAIN
**Code:**
```cpp
rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());
```
**Fix:** Replace `rmm::mr::get_current_device_resource_ref()` with `cudf::get_current_device_resource_ref()`, and remove the now-unused `#include <rmm/mr/per_device_resource.hpp>`.
**Diff:**
```diff
- #include <rmm/mr/per_device_resource.hpp>
-
  (remove line 25)
- rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());
+ rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
```
**Reference:** Convention (memory-stream section): "rmm::mr::get_current_device_resource_ref() is DEPRECATED — MUST use cudf::get_current_device_resource_ref() (MUST FIX)". Correct pattern used in same file at cast_string.hpp:234 (changed line): `rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref()`.
*(Reviewer: jni-boundary)*

#### 4. [🟡 SHOULD FIX] The default MR parameter uses the deprecated `rmm::mr::get_current_device_resource_ref()` instead of the project-mandated `cudf::get_current_device_resource_ref()`

**Line:** 2390
**Issue:** The default MR parameter uses the deprecated `rmm::mr::get_current_device_resource_ref()` instead of the project-mandated `cudf::get_current_device_resource_ref()`. Per conventions (memory-stream section): "`rmm::mr::get_current_device_resource_ref()` is DEPRECATED — MUST use `cudf::get_current_device_resource_ref()`". After fixing line 81, the `<rmm/mr/per_device_resource.hpp>` include on line 25 becomes unused (the cudf wrapper is already available via the `<cudf/utilities/memory_resource.hpp>` include on line 22).
**Confidence:** 🟣❗ CERTAIN
**Code:**
```cpp
rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());
```
**Fix:** Replace the default with `cudf::get_current_device_resource_ref()` and remove the now-unused `<rmm/mr/per_device_resource.hpp>` include.
**Diff:**
```diff
- #include <rmm/mr/per_device_resource.hpp>
+
...
- rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());
+ rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
```
**Reference:** Convention (memory-stream section): "rmm::mr::get_current_device_resource_ref() is DEPRECATED — MUST use cudf::get_current_device_resource_ref() (MUST FIX)". The cudf wrapper is defined at /home/haoyangl/code/spark-rapids-jni/thirdparty/cudf/cpp/include/cudf/utilities/memory_resource.hpp:27 and is already included via charset_decode.hpp:22. Other new functions in the same PR use the correct form, e.g., /home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/cast_string.hpp:234.
*(Reviewer: jni-boundary)*

---

### 📄 `src/main/cpp/src/cast_string_to_float.cu`

#### 5. [🟢 SUGGESTION] The `requires` constraint on the `__device__ __forceinline__` `lookup_field` template uses `std::is_invocable_r_v` rather than the `cuda::std::` equivalent

**Line:** 140
**Issue:** The `requires` constraint on the `__device__ __forceinline__` `lookup_field` template uses `std::is_invocable_r_v` rather than the `cuda::std::` equivalent. The project's gpu-kernel-correctness convention marks the `cuda::std::` mandate as review-blocking "in all device code". This constraint is a compile-time-only check (it never lowers to a device instruction), so the practical impact is nil, and it is not 100% certain that `cuda::std::is_invocable_r_v` is exposed by the CCCL version this repo pins (no local CCCL checkout was available to confirm, and the convention's substitution table lists `is_same_v`/`is_signed_v`/etc. but not `is_invocable_r_v`). For consistency with the device-code trait mandate, would it make sense to switch to `cuda::std::is_invocable_r_v` if CCCL exposes it (it requires `#include <cuda/std/type_traits>`, already included at line 27)? If CCCL does not expose it, leaving `std::` here is the only option and this can be ignored.
**Confidence:** 🟢❓ POSSIBLE
**Code:**
```cpp
template <typename Match>
  requires std::is_invocable_r_v<bool, Match, int, int>
__device__ __forceinline__ int lookup_field(int field_number, ...)
```
**Fix:** If `cuda::std::is_invocable_r_v` exists in the pinned CCCL, replace `std::is_invocable_r_v` with `cuda::std::is_invocable_r_v` here and at protobuf_kernels.cu:183. Otherwise keep as-is — this is compile-time-only and does not affect device execution.
**Diff:**
```diff
-  requires std::is_invocable_r_v<bool, Match, int, int>
+  requires cuda::std::is_invocable_r_v<bool, Match, int, int>
```
**Reference:** Convention "Section: gpu-kernel-correctness > cuda::std:: Mandate" in /home/haoyangl/code/spark-rapids-jni/.review/conventions.md:639-678 (trait substitutions); `#include <cuda/std/type_traits>` already present at /home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/protobuf/protobuf_device_helpers.cuh:27.
*(Reviewer: gpu-kernel-correctness)*

---

### 📄 `src/main/cpp/src/charset_decode.cu`

#### 6. [🟢 SUGGESTION] **Outside review scope:** The `@return` tag says "7 columns" but the `@brief` immediately above (line 147) says "6 children," and the implementation at `/home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/cast_string_to_datetime.cu:934-940` pushes exactly 6 children into `output_columns` via 6 `emplace_back` calls

**Line:** 108
**Issue:** **Outside review scope:** The `@return` tag says "7 columns" but the `@brief` immediately above (line 147) says "6 children," and the implementation at `/home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/cast_string_to_datetime.cu:934-940` pushes exactly 6 children into `output_columns` via 6 `emplace_back` calls. The count "7" in `@return` is stale, making the documentation internally contradictory.
**Confidence:** 🟣❗ CERTAIN
**Code:**
```cpp
@return a struct column constains 7 columns described above.
```
**Fix:** Correct the `@return` count to match the `@brief` and implementation — change "7 columns" to "6 columns". Also note that "constains" should be "contains".
**Diff:**
```diff
- * @return a struct column constains 7 columns described above.
+ * @return a struct column contains 6 columns described above.
```
**Reference:** /home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/cast_string_to_datetime.cu:934-940 (6 emplace_back calls); /home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/cast_string.hpp:147 (@brief says "6 children")
*(Reviewer: comment-compliance)*

---

### 📄 `src/main/cpp/src/protobuf/protobuf.cu`

#### 7. [🟢 SUGGESTION] The pattern "allocate device_uvector, cudaMemcpyAsync HostToDevice, CUDF_CUDA_TRY" is repeated at least eight times in the changed lines of `decode_protobuf_to_struct` (lines 549-588, 624-641, 784-788, 1029-1034, 1044-1049)

**Line:** 548
**Issue:** The pattern "allocate device_uvector, cudaMemcpyAsync HostToDevice, CUDF_CUDA_TRY" is repeated at least eight times in the changed lines of `decode_protobuf_to_struct` (lines 549-588, 624-641, 784-788, 1029-1034, 1044-1049). Each call site writes three lines for what is logically a single "upload host vector to device" operation. `cudf::detail::make_device_uvector_async` already wraps this pattern and is used in protobuf_builders.cu and protobuf_kernels.cu; applying it consistently here would shrink the function considerably.
**Confidence:** 🟣❗ CERTAIN
**Code:**
```cpp
CUDF_CUDA_TRY(cudaMemcpyAsync(d_repeated_indices.data(),
repeated_field_indices.data(),
num_repeated * sizeof(int),
cudaMemcpyHostToDevice,
stream.value()));
```
**Fix:** Can you replace the allocate+memcpy pairs with `cudf::detail::make_device_uvector_async`? For example:
    auto d_repeated_indices = cudf::detail::make_device_uvector_async(
      repeated_field_indices, stream, scratch_mr);
  This is already the convention used in protobuf_builders.cu:162-191 for the same kind of host-to-device copy, so adopting it here would make the style consistent.
**Diff:**
```diff
- if (num_repeated > 0) {
-   CUDF_CUDA_TRY(cudaMemcpyAsync(d_repeated_indices.data(),
-                                 repeated_field_indices.data(),
-                                 num_repeated * sizeof(int),
-                                 cudaMemcpyHostToDevice,
-                                 stream.value()));
- }
+ auto d_repeated_indices = num_repeated > 0
+   ? cudf::detail::make_device_uvector_async(repeated_field_indices, stream, scratch_mr)
+   : rmm::device_uvector<int>(0, stream, scratch_mr);
```
**Reference:** /home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/protobuf/protobuf_builders.cu:162 — `cudf::detail::make_device_uvector_async` already used for exactly the same upload pattern.
*(Reviewer: refactor)*

#### 8. [🟢 SUGGESTION] The "build host lookup table, conditionally upload to device, pass pointer+size to kernel" idiom is duplicated three times in the changed lines (lines 563-606 for `fn_to_rep`/`fn_to_nested`, lines 633-641 for `field_lookup`, and lines 1035-1050 for `fn_to_scan`)

**Line:** 563
**Issue:** The "build host lookup table, conditionally upload to device, pass pointer+size to kernel" idiom is duplicated three times in the changed lines (lines 563-606 for `fn_to_rep`/`fn_to_nested`, lines 633-641 for `field_lookup`, and lines 1035-1050 for `fn_to_scan`). Each block checks `!h_table.empty()`, allocates a device vector, copies, and then passes either `data()` or `nullptr` to the kernel. Could this be extracted into a small host helper, say `upload_lookup_table(h_table, stream, mr) -> std::pair<rmm::device_uvector<int>, int>`, which returns the device vector and its logical size (0 when the table is empty)?
**Confidence:** 🟣❗ CERTAIN
**Code:**
```cpp
// Three independent sites all do:
auto h_table = build_*(...);
rmm::device_uvector<int> d_table(0, stream, scratch_mr);
int table_size = 0;
if (!h_table.empty()) {
  d_table = rmm::device_uvector<int>(h_table.size(), stream, scratch_mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(...));
  table_size = static_cast<int>(h_table.size());
}
// ... launch kernel with (table_size > 0 ? d_table.data() : nullptr, table_size)
```
**Fix:** Extract into a helper in `protobuf_host_helpers.hpp` (which already hosts `build_lookup_table` and friends):
    struct device_lookup_table {
      rmm::device_uvector<int> data;
      int size() const { return static_cast<int>(data.size()); }
      int const* ptr() const { return size() > 0 ? data.data() : nullptr; }
    };
    device_lookup_table upload_lookup_table(std::vector<int> const& h, rmm::cuda_stream_view s, rmm::device_async_resource_ref mr);
**Diff:**
```diff
- rmm::device_uvector<int> d_fn_to_rep(0, stream, scratch_mr);
- rmm::device_uvector<int> d_fn_to_nested(0, stream, scratch_mr);
- ...
- if (!h_fn_to_rep.empty()) {
-   d_fn_to_rep = rmm::device_uvector<int>(h_fn_to_rep.size(), stream, scratch_mr);
-   CUDF_CUDA_TRY(cudaMemcpyAsync(d_fn_to_rep.data(), h_fn_to_rep.data(), ...));
- }
+ auto d_fn_to_rep    = upload_lookup_table(h_fn_to_rep, stream, scratch_mr);
+ auto d_fn_to_nested = upload_lookup_table(h_fn_to_nested, stream, scratch_mr);
...
- d_fn_to_rep.data(),
- static_cast<int>(d_fn_to_rep.size()),
+ d_fn_to_rep.ptr(),
+ d_fn_to_rep.size(),
```
**Reference:** `build_lookup_table` and `build_index_lookup_table` already centralised at /home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/protobuf/protobuf_host_helpers.hpp:48-73; the upload counterpart is currently missing, causing the repeated boilerplate in protobuf.cu:563-606, 633-642, 1035-1050.
*(Reviewer: refactor)*

#### 9. [🟢 SUGGESTION] The "allocate device vector + cudaMemcpyAsync" pattern for host->device copies is repeated at least 8 times in the changed lines (lines 549-553, 556-560, 575-580, 583-588, 624-628, 637-642, 784-789, 1029-1033, 1044-1049)

**Line:** 573
**Issue:** The "allocate device vector + cudaMemcpyAsync" pattern for host->device copies is repeated at least 8 times in the changed lines (lines 549-553, 556-560, 575-580, 583-588, 624-628, 637-642, 784-789, 1029-1033, 1044-1049). Each site manually manages an `rmm::device_uvector`, then issues a `cudaMemcpyAsync` with `CUDF_CUDA_TRY`. The codebase already has `cudf::detail::make_device_uvector_async`, which wraps exactly this pattern in a single call (as seen at `protobuf_builders.cu:162`, `185`, and `190`). Could these raw copy blocks be replaced with `cudf::detail::make_device_uvector_async` to make the intent clearer and reduce the boilerplate?
**Confidence:** 🟣❗ CERTAIN
**Code:**
```cpp
d_fn_to_rep = rmm::device_uvector<int>(h_fn_to_rep.size(), stream, scratch_mr);
CUDF_CUDA_TRY(cudaMemcpyAsync(d_fn_to_rep.data(),
                              h_fn_to_rep.data(),
                              h_fn_to_rep.size() * sizeof(int),
                              cudaMemcpyHostToDevice,
                              stream.value()));
```
**Fix:** Where `scratch_mr == cudf::get_current_device_resource_ref()` (as on line 460) the extra-allocate-then-copy sequence can be replaced with `cudf::detail::make_device_uvector_async`. Note that `make_device_uvector_async` requires the source to be a pinned host vector (or a `host_span`); for the plain `std::vector` sources used here you would first need to pin the allocation (e.g., via `cudf::detail::make_pinned_vector_async`), or use the span overload. Where the destination MR is intentionally `scratch_mr`, passing `cudf::get_current_device_resource_ref()` as the MR to `make_device_uvector_async` keeps the semantics identical. Even if the source must remain a `std::vector`, wrapping in a helper (say `upload_to_device`) would centralise the pattern.
**Diff:**
```diff
- d_fn_to_rep = rmm::device_uvector<int>(h_fn_to_rep.size(), stream, scratch_mr);
- CUDF_CUDA_TRY(cudaMemcpyAsync(d_fn_to_rep.data(),
-                               h_fn_to_rep.data(),
-                               h_fn_to_rep.size() * sizeof(int),
-                               cudaMemcpyHostToDevice,
-                               stream.value()));
+ // Option A: if h_fn_to_rep is changed to a pinned vector
+ auto d_fn_to_rep = cudf::detail::make_device_uvector_async(h_fn_to_rep, stream, scratch_mr);
+ // Option B: helper for std::vector sources
+ template <typename T>
+ rmm::device_uvector<T> upload_to_device(std::vector<T> const& h, rmm::cuda_stream_view s, rmm::device_async_resource_ref mr) {
+   rmm::device_uvector<T> d(h.size(), s, mr);
+   CUDF_CUDA_TRY(cudaMemcpyAsync(d.data(), h.data(), h.size()*sizeof(T), cudaMemcpyHostToDevice, s.value()));
+   return d;
+ }
```
**Reference:** `cudf::detail::make_device_uvector_async` used at /home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/protobuf/protobuf_builders.cu:162,185,190; raw pattern repeated at protobuf.cu:549-588, 624-641, 784-789, 1029-1049.
*(Reviewer: refactor)*

#### 10. [🟢 SUGGESTION] The `switch (element_type.id())` block for repeated scalar element types (lines 1094-1257) dispatches the same `build_repeated_scalar_column<T>(...)` template with identical arguments across 8 cases (INT32, INT64, UINT32, UINT64, FLOAT32, FLOAT64, BOOL8, INT32-fixed, INT64-fixed)

**Line:** 1094
**Issue:** The `switch (element_type.id())` block for repeated scalar element types (lines 1094-1257) dispatches the same `build_repeated_scalar_column<T>(...)` template with identical arguments across 8 cases (INT32, INT64, UINT32, UINT64, FLOAT32, FLOAT64, BOOL8, INT32-fixed, INT64-fixed). All cases pass the same 11 parameters; only the template type `T` changes. Could `cudf::type_dispatcher` (or a local `type_to_id` map) replace the switch and collapse the duplicated argument lists into a single dispatch site?
**Confidence:** 🟣❗ CERTAIN
**Code:**
```cpp
switch (element_type.id()) {
  case cudf::type_id::INT32:
    column_map[schema_idx] = build_repeated_scalar_column<int32_t>(binary_input, message_data, list_offsets, base_offset, h_device_schema[schema_idx], std::move(w.offsets), d_occurrences, total_count, num_rows, d_error, stream, mr);
    break;
  case cudf::type_id::INT64:
    column_map[schema_idx] = build_repeated_scalar_column<int64_t>(...same args...);
    break;
  // ... 6 more near-identical cases
}
```
**Fix:** A small functor combined with `cudf::type_dispatcher` would avoid the repeated argument lists. Alternatively, a lambda that captures the common arguments and calls the template can reduce the duplication:
    auto build_col = [&]<typename T>() {
      column_map[schema_idx] = build_repeated_scalar_column<T>(
        binary_input, message_data, list_offsets, base_offset,
        h_device_schema[schema_idx], std::move(w.offsets),
        d_occurrences, total_count, num_rows, d_error, stream, mr);
    };
    switch (element_type.id()) {
      case cudf::type_id::INT32:  build_col.template operator()<int32_t>(); break;
      case cudf::type_id::INT64:  build_col.template operator()<int64_t>(); break;
      ...
    }
**Diff:**
```diff
- case cudf::type_id::INT32:
-   column_map[schema_idx] =
-     build_repeated_scalar_column<int32_t>(binary_input, message_data, list_offsets,
-                                           base_offset, h_device_schema[schema_idx],
-                                           std::move(w.offsets), d_occurrences,
-                                           total_count, num_rows, d_error, stream, mr);
-   break;
- case cudf::type_id::INT64:
-   column_map[schema_idx] =
-     build_repeated_scalar_column<int64_t>(binary_input, message_data, list_offsets,
-                                           base_offset, h_device_schema[schema_idx],
-                                           std::move(w.offsets), d_occurrences,
-                                           total_count, num_rows, d_error, stream, mr);
-   break;
- // (6 more cases)
+ auto dispatch_build = [&]<typename T>() {
+   column_map[schema_idx] = build_repeated_scalar_column<T>(
+     binary_input, message_data, list_offsets, base_offset,
+     h_device_schema[schema_idx], std::move(w.offsets),
+     d_occurrences, total_count, num_rows, d_error, stream, mr);
+ };
+ switch (element_type.id()) {
+   case cudf::type_id::INT32:   dispatch_build.operator()<int32_t>(); break;
+   case cudf::type_id::INT64:   dispatch_build.operator()<int64_t>(); break;
+   case cudf::type_id::UINT32:  dispatch_build.operator()<uint32_t>(); break;
+   case cudf::type_id::UINT64:  dispatch_build.operator()<uint64_t>(); break;
+   case cudf::type_id::FLOAT32: dispatch_build.operator()<float>(); break;
+   case cudf::type_id::FLOAT64: dispatch_build.operator()<double>(); break;
+   case cudf::type_id::BOOL8:   dispatch_build.operator()<uint8_t>(); break;
```
**Reference:** `cudf::type_dispatcher` pattern documented in conventions `Section: correctness` "Type Dispatching"; similar argument-list collapse used in protobuf.cu lines 804-830 via `LAUNCH_VARINT_BATCH`/`LAUNCH_FIXED_BATCH` macros.
*(Reviewer: refactor)*

---

### 📄 `src/main/cpp/src/protobuf/protobuf_builders.cu`

#### 11. [🟢 SUGGESTION] `make_null_list_column_with_child` (lines 113-143) and `make_empty_list_column` (lines 131-143) both allocate a single-element (or `num_rows+1` element) INT32 offsets buffer, zero it with `thrust::fill`, then wrap it in a column

**Line:** 113
**Issue:** `make_null_list_column_with_child` (lines 113-143) and `make_empty_list_column` (lines 131-143) both allocate a single-element (or `num_rows+1` element) INT32 offsets buffer, zero it with `thrust::fill`, then wrap it in a column. The `make_empty_list_column` function uses `cudaMemsetAsync` on a 1-element buffer (line 140-141) rather than the same `thrust::fill` used in `make_null_list_column_with_child`. Using different approaches for the same operation (zeroing an INT32 buffer) across the same file makes code harder to maintain. Both could use the same approach consistently.
**Confidence:** 🟣❗ CERTAIN
**Code:**
```cpp
// make_null_list_column_with_child (line 119-120):
rmm::device_uvector<int32_t> offsets(num_rows + 1, stream, mr);
thrust::fill(rmm::exec_policy_nosync(stream), offsets.begin(), offsets.end(), 0);
// make_empty_list_column (line 136-141):
auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                  1,
                                                  rmm::device_buffer(sizeof(int32_t), stream, mr),
                                                  rmm::device_buffer{},
                                                  0);
CUDF_CUDA_TRY(cudaMemsetAsync(
  offsets_col->mutable_view().data<int32_t>(), 0, sizeof(int32_t), stream.value()));
```
**Fix:** Unify by factoring out a `make_zero_int32_offsets(n, stream, mr)` helper that creates and zeroes an n-element device offset buffer, or at least use the same `thrust::fill` pattern in both places.
**Diff:**
```diff
- auto offsets_col = std::make_unique<cudf::column>(...);
- CUDF_CUDA_TRY(cudaMemsetAsync(offsets_col->mutable_view().data<int32_t>(), 0, sizeof(int32_t), stream.value()));
+ rmm::device_uvector<int32_t> offsets(1, stream, mr);
+ thrust::fill(rmm::exec_policy_nosync(stream), offsets.begin(), offsets.end(), 0);
+ auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
+                                                   1, offsets.release(), rmm::device_buffer{}, 0);
```
**Reference:** Zeroing pattern with `thrust::fill` used in `make_null_list_column_with_child` at /home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/protobuf/protobuf_builders.cu:120; `cudaMemsetAsync` variant at protobuf_builders.cu:97-98, 140-141.
*(Reviewer: refactor)*

#### 12. [🟡 SHOULD FIX] The default MR parameter in the new `decode_charset` public API uses the deprecated `rmm::mr::get_current_device_resource_ref()`

**Line:** 386
**Issue:** The default MR parameter in the new `decode_charset` public API uses the deprecated `rmm::mr::get_current_device_resource_ref()`. The spark-rapids-jni conventions (inherited from cudf) require `cudf::get_current_device_resource_ref()` (MUST FIX in cudf, though the project uses both forms). Notably the newly-added functions in `cast_string.hpp` at lines 174, 196, and 234 correctly use `cudf::get_current_device_resource_ref()`, making the inconsistency visible within the same PR.
**Confidence:** 🟣❗ CERTAIN
**Code:**
```cpp
[[nodiscard]] decode_result decode_charset(
cudf::column_view const& input,
charset_type charset,
error_action action,
rmm::cuda_stream_view stream      = cudf::get_default_stream(),
rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());
```
**Fix:** Replace `rmm::mr::get_current_device_resource_ref()` with `cudf::get_current_device_resource_ref()` to match the correct form used everywhere in this PR and to be consistent with the newly-added functions in `cast_string.hpp`.
**Diff:**
```diff
- rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());
+ rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
```
**Reference:** /home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/cast_string.hpp:196 — uses `cudf::get_current_device_resource_ref()` in the newly-added `parse_timestamp_strings_with_format` declaration; cudf memory-stream conventions: "MUST use `cudf::get_current_device_resource_ref()` (MUST FIX)"
*(Reviewer: compliant)*

---

### 📄 `src/main/cpp/src/protobuf/protobuf_kernels.cu`

#### 13. [🟢 SUGGESTION] `count_repeated_fields_kernel` (lines 281-409) is a long device kernel (~130 lines)

**Line:** 302
**Issue:** `count_repeated_fields_kernel` (lines 281-409) is a long device kernel (~130 lines). It does three logically distinct things: (a) initialise the `repeated_info` and `nested_locations` arrays, (b) parse the message and count/record repeated-field occurrences, and (c) record nested-field locations. The two inner `lookup_field_idx` lambda calls (lines 349-363 and 367-399) differ only in which index table (`fn_to_rep_idx` vs `fn_to_nested_idx`) and output buffer (`repeated_info` vs `nested_locations`) they use. Could the initialisation loops (lines 301-313) be moved to a separate initialization kernel to reduce kernel size and improve testability? Similarly, the two lookup branches share enough structure that a single parameterized helper (perhaps passed as a template lambda) might reduce nesting.
**Confidence:** 🟣❗ CERTAIN
**Code:**
```cpp
CUDF_KERNEL void count_repeated_fields_kernel(...) {
  // 1. Init repeated_info array
  for (int f = 0; f < num_repeated_fields; f++) { ... }
  // 2. Init nested_locations array
  for (int f = 0; f < num_nested_fields; f++) { ... }
  // 3. null check / message bounds
  // 4. parse loop with two similar lookup branches
}
```
**Fix:** The initialisation loops (items 1 and 2 above) could be replaced by a `thrust::fill` call on the host side before launching the kernel (since `repeated_field_info{0}` and `field_location{-1,0}` are fixed constants), eliminating ~12 lines of device-side per-thread initialisation work. The parse loop itself is appropriate for a kernel.
**Diff:**
```diff
+ // Before launching count_repeated_fields_kernel:
+ thrust::fill(rmm::exec_policy_nosync(stream), d_repeated_info.begin(), d_repeated_info.end(), repeated_field_info{0});
+ thrust::fill(rmm::exec_policy_nosync(stream), d_nested_locations.begin(), d_nested_locations.end(), field_location{-1, 0});
- // Remove init loops inside kernel:
- for (int f = 0; f < num_repeated_fields; f++) {
-   repeated_info[flat_index(...)] = {0};
- }
- for (int f = 0; f < num_nested_fields; f++) {
-   nested_locations[flat_index(...)] = {-1, 0};
- }
```
**Reference:** `thrust::fill` with `rmm::exec_policy_nosync` used throughout the file (e.g. protobuf_builders.cu:314-317, protobuf_kernels.cu:902).
*(Reviewer: refactor)*

#### 14. [🟢 SUGGESTION] The scalar-field group classification block (lines 682-733) is a sequence of 12 mutually exclusive `if/else if` branches encoding a mapping from `(type_id, encoding)` → group index

**Line:** 439
**Issue:** The scalar-field group classification block (lines 682-733) is a sequence of 12 mutually exclusive `if/else if` branches encoding a mapping from `(type_id, encoding)` → group index. This is in a `for` loop over up to `num_scalar` fields and runs on the host once per batch. The logic is dense; a lookup table (a `std::unordered_map` or a small inline function) would make it easier to verify that the mapping is complete and correct, and would simplify adding new types in the future.
**Confidence:** 🟢❓ POSSIBLE
**Code:**
```cpp
int g = GRP_FALLBACK;
if (tid == cudf::type_id::INT32 && is_fixed) {
  g = 9;
} else if (tid == cudf::type_id::INT64 && is_fixed) {
  g = 10;
} else if (tid == cudf::type_id::UINT32 && is_fixed) {
  g = 9;
...
```
**Fix:** Consider replacing the chain with a helper function or a small struct-keyed table:
    auto classify_scalar_group = [](cudf::type_id tid, proto_encoding enc) -> int {
      bool const zz = (enc == proto_encoding::ZIGZAG);
      bool const fx = (enc == proto_encoding::FIXED);
      if (tid == cudf::type_id::INT32  && !zz && !fx) return 0;
      if (tid == cudf::type_id::UINT32 && !zz && !fx) return 1;
      ...
    };
  This is a POSSIBLE suggestion because the current code works correctly; it is purely a readability concern.
**Diff:**
```diff
// (representative sketch of helper extraction)
+ auto classify_scalar_group = [](cudf::type_id tid, proto_encoding enc) -> int {
+   bool const zz = (enc == proto_encoding::ZIGZAG);
+   bool const fx = (enc == proto_encoding::FIXED);
+   if (tid == cudf::type_id::INT32  && !zz && !fx) return 0;
+   if (tid == cudf::type_id::UINT32 && !fx)         return 1;
+   if (tid == cudf::type_id::INT64  && !zz && !fx) return 2;
+   if (tid == cudf::type_id::UINT64 && !fx)         return 3;
+   if (tid == cudf::type_id::BOOL8)                 return 4;
+   if (tid == cudf::type_id::INT32  && zz)          return 5;
+   if (tid == cudf::type_id::INT64  && zz)          return 6;
+   if (tid == cudf::type_id::FLOAT32)               return 7;
+   if (tid == cudf::type_id::FLOAT64)               return 8;
+   if ((tid == cudf::type_id::INT32 || tid == cudf::type_id::UINT32) && fx) return 9;
+   if ((tid == cudf::type_id::INT64 || tid == cudf::type_id::UINT64) && fx) return 10;
+   return 11; // GRP_FALLBACK
+ };
```
**Reference:** Conventions Section: refactor — "Complexity assessment: Functions with deep nesting (3+ levels), many branches (5+ if/case paths)… 10+ branch points warrants a SUGGESTION for decomposition."
*(Reviewer: refactor)*

#### 15. [🟢 SUGGESTION] `enum_binary_search` (lines 555-573) is a standard lower-bound-then-equality-check binary search

**Line:** 555
**Issue:** `enum_binary_search` (lines 555-573) is a standard lower-bound-then-equality-check binary search. The same logic is run from three different kernel call sites (lines 602, 631, 654). The implementation is correct, but could simply delegate to `cuda::std::lower_bound` (available in `<cuda/std/algorithm>`) to reduce custom code. This is especially valuable from an edge-case perspective because the hand-rolled version initialises `left` and `right` as `int`, so a `num_valid_values` of 0 would cause `right = -1` at line 559 — which is safe since `left <= right` is false immediately, but is an implicit assumption worth making explicit.
**Confidence:** 🟣❗ CERTAIN
**Code:**
```cpp
__device__ inline int enum_binary_search(int32_t const* valid_enum_values,
                                         int num_valid_values,
                                         int32_t val)
{
  int left  = 0;
  int right = num_valid_values - 1;
  while (left <= right) {
    int mid         = left + (right - left) / 2;
    ...
```
**Fix:** Could you replace the hand-rolled search with `cuda::std::lower_bound` and an equality check? For example:
    __device__ inline int enum_binary_search(int32_t const* valid_enum_values,
                                             int num_valid_values,
                                             int32_t val)
    {
      auto const* end = valid_enum_values + num_valid_values;
      auto const* it  = cuda::std::lower_bound(valid_enum_values, end, val);
      if (it == end || *it != val) return -1;
      return static_cast<int>(it - valid_enum_values);
    }
  This removes the edge-case risk from `right = -1` and delegates correctness to a well-tested standard primitive.
**Diff:**
```diff
- __device__ inline int enum_binary_search(int32_t const* valid_enum_values,
-                                          int num_valid_values,
-                                          int32_t val)
- {
-   int left  = 0;
-   int right = num_valid_values - 1;
-   while (left <= right) {
-     int mid         = left + (right - left) / 2;
-     int32_t mid_val = valid_enum_values[mid];
-     if (mid_val == val) {
-       return mid;
-     } else if (mid_val < val) {
-       left = mid + 1;
-     } else {
-       right = mid - 1;
-     }
-   }
-   return -1;
- }
+ __device__ inline int enum_binary_search(int32_t const* valid_enum_values,
+                                          int num_valid_values,
+                                          int32_t val)
+ {
+   auto const* end = valid_enum_values + num_valid_values;
+   auto const* it  = cuda::std::lower_bound(valid_enum_values, end, val);
+   if (it == end || *it != val) return -1;
+   return static_cast<int>(it - valid_enum_values);
+ }
```
**Reference:** `<cuda/std/algorithm>` provides `cuda::std::lower_bound` for device code per conventions Section: gpu-kernel-correctness.
*(Reviewer: refactor)*

#### 16. [🟢 SUGGESTION] The blocks-count expression `static_cast<int>((N + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK)` is copy-pasted verbatim in at least ten host-side wrapper functions (lines 687, 717, 746, 760, 775, 792, 830, 896)

**Line:** 687
**Issue:** The blocks-count expression `static_cast<int>((N + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK)` is copy-pasted verbatim in at least ten host-side wrapper functions (lines 687, 717, 746, 760, 775, 792, 830, 896). Extracting it to a small inline helper would make the launch arithmetic self-documenting and easier to audit.
**Confidence:** 🟣❗ CERTAIN
**Code:**
```cpp
auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
```
**Fix:** Could you extract this into a one-liner helper near the top of the anonymous namespace (or in a shared header), for example:
    inline int grid_size(int n) { return static_cast<int>((n + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK); }
  Then every call site becomes `auto const blocks = grid_size(num_rows);`, which reads more clearly and centralises the rounding logic.
**Diff:**
```diff
// Add near THREADS_PER_BLOCK definition (e.g. protobuf_types.cuh or protobuf_kernels.cu preamble):
+ inline int grid_size(int n) { return static_cast<int>((static_cast<unsigned>(n) + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK); }
//
// Then at every call site replace, e.g.:
- auto const blocks = static_cast<int>((num_rows + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK);
+ auto const blocks = grid_size(num_rows);
```
**Reference:** Same pattern used in many RAPIDS kernels via `cudf::util::div_rounding_up_safe`; alternatively `cuda::ceil_div(n, THREADS_PER_BLOCK)` from `<cuda/std/utility>` is already available.
*(Reviewer: refactor)*

#### 17. [🟢 SUGGESTION] The two new helper functions `is_supported_row_conversion_type` (line 2044) and `check_supported_columns` (line 2052) are defined inside an anonymous namespace at the bottom of the file (lines 2037-2086), far from the public API functions they validate (`convert_to_rows` at line 2096, `convert_to_rows_fixed_width_optimized` at line 2159, `convert_from_rows_fixed_width_optimized` at line 2558)

**Line:** 865
**Issue:** The two new helper functions `is_supported_row_conversion_type` (line 2044) and `check_supported_columns` (line 2052) are defined inside an anonymous namespace at the bottom of the file (lines 2037-2086), far from the public API functions they validate (`convert_to_rows` at line 2096, `convert_to_rows_fixed_width_optimized` at line 2159, `convert_from_rows_fixed_width_optimized` at line 2558). Placing validation helpers at the bottom of a 2600-line file means a reader must scroll past hundreds of unrelated lines to find them. Consider moving these to near the top of the `spark_rapids_jni` namespace (around the existing private `detail` helpers), or at least before the first public function that uses them.
**Confidence:** 🟣❗ CERTAIN
**Code:**
```cpp
// Lines 2037-2086: anonymous namespace with helpers
namespace {
inline bool is_supported_row_conversion_type(data_type t, bool fixed_width_only) { ... }
inline bool check_supported_columns(table_view const& tbl, bool fixed_width_only) { ... }
inline void check_supported_schema(std::vector<data_type> const& schema, bool fixed_width_only) { ... }
}
// Lines 2096, 2160, 2558: public functions that use the helpers above
```
**Fix:** Move the anonymous namespace block with these three helpers to just before the first function that uses them (`convert_to_rows` at line 2096), or into the existing `detail` namespace region. This is a structural reorganization with no behavior change.
**Diff:**
```diff
- // At line 2037 (after convert_to_rows body)
- namespace {
- inline bool is_supported_row_conversion_type(...) { ... }
- ...
- }
+ // Move to before line 2096 (before convert_to_rows)
+ namespace {
+ inline bool is_supported_row_conversion_type(...) { ... }
+ ...
+ }
```
**Reference:** Other file-local helpers in row_conversion.cu are placed near their first use; the placement at line 2037 breaks this pattern.
*(Reviewer: refactor)*

#### 18. [🟢 SUGGESTION] `device_tokens` is filled by `cudaMemcpyAsync` whose source is `host_tokens`, a `std::vector<format_token>` (pageable host memory)

**Line:** 867
**Issue:** `device_tokens` is filled by `cudaMemcpyAsync` whose source is `host_tokens`, a `std::vector<format_token>` (pageable host memory). For pageable host->device transfers CUDA stages the source into an internal pinned buffer before the call returns, so this is functionally safe even though the vector is destroyed when the function returns — there is no use-after-free. The downside is the copy is not truly asynchronous (the host pays the staging cost). For consistency with the rest of the protobuf/charset code that builds device buffers from host vectors (e.g. `cudf::detail::make_device_uvector_async`), would it be worth using a stream-ordered helper here? It would let the H2D copy overlap with surrounding work and removes the manual `cudaMemcpyAsync` bookkeeping.
**Confidence:** 🟣❗ CERTAIN
**Code:**
```cpp
|
rmm::device_uvector<format_token> device_tokens(
  host_tokens.size(), stream, cudf::get_current_device_resource_ref());
CUDF_CUDA_TRY(cudaMemcpyAsync(device_tokens.data(),
                              host_tokens.data(),
                              sizeof(format_token) * host_tokens.size(),
                              cudaMemcpyHostToDevice,
                              stream.value()));
```
**Fix:** Optionally replace the explicit allocation + `cudaMemcpyAsync` with `auto device_tokens = cudf::detail::make_device_uvector_async(host_tokens, stream, cudf::get_current_device_resource_ref());`, which performs pinned staging and a stream-ordered async copy. (Note: the spark-rapids-jni convention also accepts raw `cudaMemcpyAsync` to reduce `cudf::detail::` coupling, so this is optional.)
**Diff:**
```diff
|
-  rmm::device_uvector<format_token> device_tokens(
-    host_tokens.size(), stream, cudf::get_current_device_resource_ref());
-  CUDF_CUDA_TRY(cudaMemcpyAsync(device_tokens.data(),
-                                host_tokens.data(),
-                                sizeof(format_token) * host_tokens.size(),
-                                cudaMemcpyHostToDevice,
-                                stream.value()));
+  auto device_tokens = cudf::detail::make_device_uvector_async(
+    host_tokens, stream, cudf::get_current_device_resource_ref());
```
**Reference:** CUDA Runtime API synchronization behavior (https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html): pageable H2D copies stage the source before returning, so the pageable source need not outlive the call. Codebase precedent: /home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/protobuf/protobuf_kernels.cu:827-828 and protobuf_builders.cu:162-163 use `cudf::detail::make_device_uvector_async`.
*(Reviewer: memory-stream)*

#### 19. [🟢 SUGGESTION] `validate_enum_and_propagate_rows` (lines 885-913) is a composite function that (1) allocates an `item_invalid` buffer, (2) fills it with false, (3) launches `validate_enum_values_kernel`, then (4) calls `propagate_invalid_enum_flags_to_rows`

**Line:** 885
**Issue:** `validate_enum_and_propagate_rows` (lines 885-913) is a composite function that (1) allocates an `item_invalid` buffer, (2) fills it with false, (3) launches `validate_enum_values_kernel`, then (4) calls `propagate_invalid_enum_flags_to_rows`. In `protobuf_builders.cu:239-273`, `build_enum_string_column` performs nearly the same sequence: allocate `d_item_has_invalid_enum`, fill to false, call `launch_validate_enum_values`, then call `propagate_invalid_enum_flags_to_rows`. The composite helper `validate_enum_and_propagate_rows` was presumably added to unify this, yet `build_enum_string_column` does not use it. Could `build_enum_string_column` delegate to `validate_enum_and_propagate_rows` instead of reimplementing the sequence?
**Confidence:** 🟣❗ CERTAIN
**Code:**
```cpp
// In build_enum_string_column (protobuf_builders.cu:252-271):
rmm::device_uvector<bool> d_item_has_invalid_enum(num_rows, stream, ...);
thrust::fill(..., false);
launch_validate_enum_values(enum_values.data(), valid.data(), d_item_has_invalid_enum.data(), ...);
propagate_invalid_enum_flags_to_rows(d_item_has_invalid_enum, d_row_force_null, ...);
// Also independently implemented in validate_enum_and_propagate_rows (protobuf_kernels.cu:885):
// (same sequence with slightly different parameter names)
```
**Fix:** Replace the duplicated sequence inside `build_enum_string_column` with a call to `validate_enum_and_propagate_rows`. The signatures differ slightly (`validate_enum_and_propagate_rows` takes `values` and `valid_enums` and calls `validate_enum_values_kernel` directly, while `build_enum_string_column` calls the host wrapper `launch_validate_enum_values`) — adjust one to match the other.
**Diff:**
```diff
- rmm::device_uvector<bool> d_item_has_invalid_enum(num_rows, stream, cudf::get_current_device_resource_ref());
- thrust::fill(rmm::exec_policy_nosync(stream), d_item_has_invalid_enum.begin(), d_item_has_invalid_enum.end(), false);
- launch_validate_enum_values(enum_values.data(), valid.data(), d_item_has_invalid_enum.data(),
-                             lookup.d_valid_enums.data(), static_cast<int>(valid_enums.size()),
-                             num_rows, stream);
- propagate_invalid_enum_flags_to_rows(d_item_has_invalid_enum, d_row_force_null, num_rows, top_row_indices, propagate_invalid_rows, stream);
+ validate_enum_and_propagate_rows(enum_values, valid, valid_enums, d_row_force_null, num_rows, top_row_indices, propagate_invalid_rows, stream);
```
**Reference:** `validate_enum_and_propagate_rows` declared at /home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/protobuf/protobuf_host_helpers.hpp:168-175; duplicate sequence at /home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/protobuf/protobuf_builders.cu:252-271.
*(Reviewer: refactor)*

---

### 📄 `src/main/cpp/src/protobuf/protobuf_kernels.cuh`

#### 20. [🟡 SHOULD FIX] The default MR parameter for `decode_charset` uses the deprecated `rmm::mr::get_current_device_resource_ref()` rather than the project-preferred `cudf::get_current_device_resource_ref()`

**Line:** 423
**Issue:** The default MR parameter for `decode_charset` uses the deprecated `rmm::mr::get_current_device_resource_ref()` rather than the project-preferred `cudf::get_current_device_resource_ref()`. The header already includes `<cudf/utilities/memory_resource.hpp>` at line 22 which provides `cudf::get_current_device_resource_ref()`, making the explicit `<rmm/mr/per_device_resource.hpp>` include at line 25 unnecessary once the call site is updated. The new functions added in this PR in `cast_string.hpp` (e.g. `parse_timestamp_strings_with_format` at line 234) correctly use `cudf::get_current_device_resource_ref()`, so this is inconsistent with the PR's own new code.
**Confidence:** 🟣❗ CERTAIN
**Code:**
```cpp
rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());
```
**Fix:** Replace `rmm::mr::get_current_device_resource_ref()` with `cudf::get_current_device_resource_ref()`, and remove the now-unneeded `#include <rmm/mr/per_device_resource.hpp>` at line 25.
**Diff:**
```diff
- #include <rmm/mr/per_device_resource.hpp>
+ // (remove this include — cudf/utilities/memory_resource.hpp already provides cudf::get_current_device_resource_ref())
...
-   rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource_ref());
+   rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
```
**Reference:** /home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/cast_string.hpp:234 — new `parse_timestamp_strings_with_format` function added in this PR uses `cudf::get_current_device_resource_ref()`. The memory-stream convention (cudf.md Section: memory-stream) states: "`rmm::mr::get_current_device_resource_ref()` is DEPRECATED — MUST use `cudf::get_current_device_resource_ref()` (MUST FIX)".
*(Reviewer: compliant)*

#### 21. [🟢 SUGGESTION] `extract_and_build_scalar_column` (lines 445-460) takes a `LaunchFn` callback that receives `out.data()` and `valid.data()`, which it uses to launch a kernel

**Line:** 445
**Issue:** `extract_and_build_scalar_column` (lines 445-460) takes a `LaunchFn` callback that receives `out.data()` and `valid.data()`, which it uses to launch a kernel. This API requires the caller to know about the internal buffer layout to fill in `output` and `valid` pointers. When there are now many callers (scalar path, repeated path, etc.) this tight coupling could make it harder to change the buffer strategy (e.g., switching from `bool*` validity to a different representation). The function's `num_rows == 0` branch (line 454) also silently returns a column with `rmm::device_buffer{}` as the null mask, meaning the caller cannot tell if the column is nullable or not from the return value alone — which differs from the standard `make_fixed_width_column(dtype, 0, mask_state::UNINITIALIZED)` path.
**Confidence:** 🟣❗ CERTAIN
**Code:**
```cpp
template <typename T, typename LaunchFn>
std::unique_ptr<cudf::column> extract_and_build_scalar_column(cudf::data_type dt,
                                                              int num_rows,
                                                              LaunchFn&& launch_extract,
                                                              rmm::cuda_stream_view stream,
                                                              rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<T> out(num_rows, stream, mr);
  rmm::device_uvector<bool> valid((num_rows > 0 ? num_rows : 1), stream, mr);
  if (num_rows == 0) {
    return std::make_unique<cudf::column>(dt, 0, out.release(), rmm::device_buffer{}, 0);
  }
```
**Fix:** Consider whether the `num_rows == 0` early return should use `cudf::make_empty_column(dt)` instead of manually constructing the column, which would align with `make_empty_column_safe` used elsewhere. Also consider whether the `LaunchFn` signature should be narrowed to a more concrete form (e.g., a struct with `output_type`, `output_ptr`, `valid_ptr`) to reduce implicit coupling.
**Diff:**
```diff
-     if (num_rows == 0) {
-       return std::make_unique<cudf::column>(dt, 0, out.release(), rmm::device_buffer{}, 0);
-     }
+     if (num_rows == 0) {
+       return cudf::make_empty_column(dt);
+     }
```
**Reference:** `cudf::make_empty_column` / `make_empty_column_safe` used at /home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/protobuf/protobuf_builders.cu:109; `cudf::make_empty_column()` mentioned in conventions Section: refactor under "cuDF Reusable Utilities".
*(Reviewer: refactor)*

#### 22. [🟡 SHOULD FIX] In the `as_bytes` (num_rows > 0) branch of `extract_and_build_string_or_bytes_column`, the bytes child column is built with `rmm::device_buffer(chars.data(), total_size, stream, mr)`

**Line:** 673
**Issue:** In the `as_bytes` (num_rows > 0) branch of `extract_and_build_string_or_bytes_column`, the bytes child column is built with `rmm::device_buffer(chars.data(), total_size, stream, mr)`. That is `rmm::device_buffer`'s copy-from-pointer constructor, so it allocates a *second* `total_size`-byte device buffer against `mr` and does a device-to-device copy of the chars, after which the original `chars` uvector is freed at scope exit. The parallel STRING branch right below (line 681) and the equivalent code in protobuf_builders.cu:431 both use `chars.release()` to transfer ownership with no copy. Can the bytes path do the same? `rmm::device_uvector<char>::release()` returns an `rmm::device_buffer`, so it plugs straight into the column constructor and avoids the redundant allocation + D2D copy on this hot path.
**Confidence:** 🟣❗ CERTAIN
**Code:**
```cpp
|
auto bytes_child =
  std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT8},
                                 total_size,
                                 rmm::device_buffer(chars.data(), total_size, stream, mr),
                                 rmm::device_buffer{},
                                 0);
```
**Fix:** Replace the copying `rmm::device_buffer(chars.data(), total_size, stream, mr)` with `chars.release()` to move ownership of the already-allocated buffer into the child column, matching the STRING branch and protobuf_builders.cu:431.
**Diff:**
```diff
|
-    auto bytes_child =
-      std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT8},
-                                     total_size,
-                                     rmm::device_buffer(chars.data(), total_size, stream, mr),
-                                     rmm::device_buffer{},
-                                     0);
+    auto bytes_child = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::UINT8},
+                                                      total_size,
+                                                      chars.release(),
+                                                      rmm::device_buffer{},
+                                                      0);
```
**Reference:** /home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/protobuf/protobuf_builders.cu:430-431 ("Transfer ownership of the chars buffer instead of copying — the strings path below uses `chars.release()` for the same reason."); same-function STRING branch at /home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/protobuf/protobuf_kernels.cuh:681 uses `chars.release()`.
*(Reviewer: memory-stream)*

#### 23. [🟢 SUGGESTION] `extract_typed_column` (lines 684-846 of the .cuh) has a very long parameter list (18 parameters)

**Line:** 684
**Issue:** `extract_typed_column` (lines 684-846 of the .cuh) has a very long parameter list (18 parameters). It is called from two call sites and several parameters always travel together: `(message_data, loc_provider, num_items, blocks, threads_per_block)` and `(has_default, default_int, default_float, default_bool, default_string)`. Grouping these into small structs (e.g. a `field_extract_context` and a `field_default_values`) would make call sites readable at a glance and reduce the chance of parameter-order mistakes.
**Confidence:** 🟣❗ CERTAIN
**Code:**
```cpp
template <typename LocationProvider>
inline std::unique_ptr<cudf::column> extract_typed_column(
  cudf::data_type dt,
  int encoding,
  uint8_t const* message_data,
  LocationProvider const& loc_provider,
  int num_items,
  int blocks,
  int threads_per_block,
  bool has_default,
  int64_t default_int,
  double default_float,
  bool default_bool,
  cudf::detail::host_vector<uint8_t> const& default_string,
  int schema_idx,
  ...
```
**Fix:** Consider grouping the five default-value parameters into a struct, for example:
    struct field_defaults {
      bool     has_default;
      int64_t  default_int;
      double   default_float;
      bool     default_bool;
      cudf::detail::host_vector<uint8_t> const* default_string;
    };
  The call sites already have `field_meta` available which already bundles these — it might even be simpler to pass `protobuf_field_meta_view const&` directly to `extract_typed_column` and let it unpack what it needs.
**Diff:**
```diff
// (sketch — exact diff depends on chosen grouping)
- inline std::unique_ptr<cudf::column> extract_typed_column(
-   cudf::data_type dt, int encoding, uint8_t const* message_data,
-   LocationProvider const& loc_provider, int num_items, int blocks, int threads_per_block,
-   bool has_default, int64_t default_int, double default_float, bool default_bool,
-   cudf::detail::host_vector<uint8_t> const& default_string, int schema_idx, ...)
+ struct field_defaults {
+   bool has_default = false;
+   int64_t default_int = 0;
+   double default_float = 0.0;
+   bool default_bool = false;
+   cudf::detail::host_vector<uint8_t> const* default_string = nullptr;
+ };
+ inline std::unique_ptr<cudf::column> extract_typed_column(
+   cudf::data_type dt, int encoding, uint8_t const* message_data,
+   LocationProvider const& loc_provider, int num_items, int blocks, int threads_per_block,
+   field_defaults const& defaults, int schema_idx, ...)
```
**Reference:** Refactoring convention from conventions.md Section: refactor — "Long parameter lists: Functions with 5+ parameters — suggest grouping into config/options struct."
*(Reviewer: refactor)*

#### 24. [🟢 SUGGESTION] `build_repeated_scalar_column` (lines 848-919) ends with an inline null-mask / `make_lists_column` sequence that duplicates exactly the logic in `make_list_column_with_input_nulls` (protobuf_builders.cu:24-42)

**Line:** 848
**Issue:** `build_repeated_scalar_column` (lines 848-919) ends with an inline null-mask / `make_lists_column` sequence that duplicates exactly the logic in `make_list_column_with_input_nulls` (protobuf_builders.cu:24-42). Both branch on `input_null_count > 0` and call `cudf::copy_bitmask` in the true arm and `rmm::device_buffer{}` in the false arm. The builder function already exists and is called from `build_repeated_string_column` and `build_repeated_enum_string_column`; using it here would remove the duplication.
**Confidence:** 🟣❗ CERTAIN
**Code:**
```cpp
(protobuf_kernels.cuh:908-918)
if (input_null_count > 0) {
  auto null_mask = cudf::copy_bitmask(binary_input, stream, mr);
  return cudf::make_lists_column(num_rows,
                                 std::move(offsets_col),
                                 std::move(child_col),
                                 input_null_count,
                                 std::move(null_mask));
}
return cudf::make_lists_column(
  num_rows, std::move(offsets_col), std::move(child_col), 0, rmm::device_buffer{});
```
**Fix:** Can you replace the inline branch with a call to the existing helper?
    return make_list_column_with_input_nulls(
      num_rows, std::move(offsets_col), std::move(child_col), binary_input, stream, mr);
**Diff:**
```diff
- if (input_null_count > 0) {
-   auto null_mask = cudf::copy_bitmask(binary_input, stream, mr);
-   return cudf::make_lists_column(num_rows,
-                                  std::move(offsets_col),
-                                  std::move(child_col),
-                                  input_null_count,
-                                  std::move(null_mask));
- }
- return cudf::make_lists_column(
-   num_rows, std::move(offsets_col), std::move(child_col), 0, rmm::device_buffer{});
+ return make_list_column_with_input_nulls(
+   num_rows, std::move(offsets_col), std::move(child_col), binary_input, stream, mr);
```
**Reference:** /home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/protobuf/protobuf_builders.cu:24-42 — `make_list_column_with_input_nulls` provides identical logic. Already used at protobuf_builders.cu:352 and :446.
*(Reviewer: refactor)*

---

### 📄 `src/main/cpp/tests/protobuf_helpers.cu`

#### 25. [🟢 SUGGESTION] `NullMaskFromPaddedValidUsesZeroLogicalRows` passes `num_rows=0` but the `valid` buffer still contains one `false` element

**Line:** 36
**Issue:** `NullMaskFromPaddedValidUsesZeroLogicalRows` passes `num_rows=0` but the `valid` buffer still contains one `false` element. There is no test for `num_rows=0` with an empty `valid` buffer (size 0), which is the edge case that exercises the early-return path for an entirely empty column. The CUDF_EXPECTS guard at protobuf_kernels.cuh:434 checks `valid.size() >= num_rows`, so a zero-size buffer with num_rows=0 is valid but untested.
**Confidence:** 🟣❗ CERTAIN
**Code:**
```cpp
|
std::array<bool, 1> h_valid{false};  // size=1, num_rows=0
// valid.size() is 1, num_rows is 0 — the guard allows this but
// the truly-empty path (valid.size()==0, num_rows==0) is not tested
```
**Fix:** Add a third test case that constructs an empty `rmm::device_uvector<bool>` (size 0) and calls `make_null_mask_from_valid` with `num_rows=0`, asserting that it returns an empty mask without a CUDF_EXPECTS failure.
**Diff:**
```diff
+ TEST_F(ProtobufHelpersTest, NullMaskFromEmptyValidAndZeroRows)
+ {
+   auto stream = cudf::get_default_stream();
+   rmm::device_uvector<bool> valid(0, stream);
+   auto [mask, null_count] = spark_rapids_jni::protobuf::detail::make_null_mask_from_valid(
+     valid, 0, stream, cudf::get_current_device_resource_ref());
+   EXPECT_EQ(0u, mask.size());
+   EXPECT_EQ(nullptr, mask.data());
+   EXPECT_EQ(0, null_count);
+ }
```
**Reference:** /home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/protobuf/protobuf_kernels.cuh:434-435 (CUDF_EXPECTS guard that is exercised here)
*(Reviewer: test-coverage)*

---

### 📄 `src/main/java/com/nvidia/spark/rapids/jni/OrcTimezoneInfo.java`

#### 26. [🟡 SHOULD FIX] `host_tokens` is a `std::vector<format_token>` (pageable host memory) used as the source of a `cudaMemcpyAsync` H2D copy, and `parse_timestamp_strings_with_format` returns at line 387 without ever synchronizing `stream`

**Line:** 244
**Issue:** `host_tokens` is a `std::vector<format_token>` (pageable host memory) used as the source of a `cudaMemcpyAsync` H2D copy, and `parse_timestamp_strings_with_format` returns at line 387 without ever synchronizing `stream`. `cudf::bools_to_mask` (line 384) returns a device buffer and does not sync the host. When the function returns, `host_tokens` is destroyed while the async copy from its pageable buffer may still be in flight. Per the CUDA Runtime API sync-behavior docs, an H2D copy from pageable memory "might be synchronous" but is not guaranteed to be — the driver may stage from the user buffer after the call returns. The project conventions explicitly flag this exact pattern (conventions.md:614: "NOT safe: `std::vector` as source for async copy without `stream.synchronize()` before destruction"). The idiomatic, already-used fix in this same PR is `cudf::detail::make_device_uvector_async`, which stages through a pinned bounce buffer (see protobuf_builders.cu:162 using `make_device_uvector_async`).
**Confidence:** 🟠❗ LIKELY
**Code:**
```java
|
auto const host_tokens = compile_format(format, legacy);            // std::vector
...
rmm::device_uvector<format_token> device_tokens(
  host_tokens.size(), stream, cudf::get_current_device_resource_ref());
CUDF_CUDA_TRY(cudaMemcpyAsync(device_tokens.data(),
                              host_tokens.data(),                    // pageable source
                              sizeof(format_token) * host_tokens.size(),
                              cudaMemcpyHostToDevice,
                              stream.value()));
...
return result;   // no stream.synchronize(); host_tokens destroyed here
```
**Fix:** Use the pinned-staging helper `cudf::detail::make_device_uvector_async(host_tokens, stream, cudf::get_current_device_resource_ref())` to build `device_tokens`. This both removes the manual `cudaMemcpyAsync` and guarantees the source data is safely staged regardless of stream completion. (Alternatively, stage `host_tokens` into a pinned vector via `cudf::detail::make_pinned_vector_async` before the copy, matching the protobuf code in this PR.)
**Diff:**
```diff
|
-  rmm::device_uvector<format_token> device_tokens(
-    host_tokens.size(), stream, cudf::get_current_device_resource_ref());
-  CUDF_CUDA_TRY(cudaMemcpyAsync(device_tokens.data(),
-                                host_tokens.data(),
-                                sizeof(format_token) * host_tokens.size(),
-                                cudaMemcpyHostToDevice,
-                                stream.value()));
+  auto device_tokens = cudf::detail::make_device_uvector_async(
+    host_tokens, stream, cudf::get_current_device_resource_ref());
```
**Reference:** conventions.md:614 ("NOT safe: `std::vector` as source for async copy without `stream.synchronize()` before destruction"); CUDA Runtime API sync-behavior doc (https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html): "For transfers between device memory and pageable host memory, the function might be synchronous with respect to host"; in-PR correct pattern at /home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/protobuf/protobuf_builders.cu:162.
*(Reviewer: correctness)*

---

### 📄 `src/test/java/com/nvidia/spark/rapids/jni/CastStringsTest.java`

#### 27. [🟢 SUGGESTION] The new timestamp parsing tests (`parseTimestampWithFormat_correctedDateOnlyFormats` etc.) at lines 1503-1747 do not cover a null-only input column (all rows null)

**Line:** 1520
**Issue:** The new timestamp parsing tests (`parseTimestampWithFormat_correctedDateOnlyFormats` etc.) at lines 1503-1747 do not cover a null-only input column (all rows null). The underlying GPU kernel in parse_timestamp_with_format.cu calls `for_each_n` over all rows; a column where every row is null should produce an all-null output, and the validity buffer must not be uninitialized for null positions (the comment at line 369 says "Every code path … writes validity[idx]" but this only holds if threads for null rows still write).
**Confidence:** 🟣❗ CERTAIN
**Code:**
```java
|
// parse_timestamp_with_format.cu:372-381
thrust::for_each_n(rmm::exec_policy_nosync(stream), ..., num_rows,
                   parse_with_format_fn{...});
// No null-input guard before the kernel; relies on operator() handling nulls
```
**Fix:** Add a test with a column of all-null strings using `parseTimestampWithFormat` and verify all output rows are null, matching the contract for null-input rows.
**Diff:**
```diff
+ @Test
+ void parseTimestampWithFormat_allNullInput() {
+   try (ColumnVector in = ColumnVector.fromStrings((String)null, (String)null);
+        ColumnVector actual = CastStrings.parseTimestampWithFormat(in, "yyyy-MM-dd", false);
+        ColumnVector exp = ColumnVector.timestampMicroSecondsFromBoxedLongs(null, null)) {
+     AssertUtils.assertColumnsAreEqual(exp, actual);
+   }
+ }
```
**Reference:** /home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/parse_timestamp_with_format.cu:368-381
*(Reviewer: test-coverage)*

---

### 📄 `src/test/java/com/nvidia/spark/rapids/jni/CharsetDecodeTest.java`

#### 28. [🟢 SUGGESTION] `testZeroLengthNestedMessage` (lines 3072-3099) verifies that decoding a zero-length nested message produces a STRUCT result, but the only assertion is type-level (`assertEquals(DType.STRUCT, result.getType())`)

**Line:** 263
**Issue:** `testZeroLengthNestedMessage` (lines 3072-3099) verifies that decoding a zero-length nested message produces a STRUCT result, but the only assertion is type-level (`assertEquals(DType.STRUCT, result.getType())`). The child INT32 field inside the inner struct is missing from the wire data, so it should surface as null (or default). Verifying null/default behavior for a missing nested scalar child would more precisely catch regressions in the "nested struct present but child absent" code path.
**Confidence:** 🟣❗ CERTAIN
**Code:**
```java
assertNotNull(result);
assertEquals(DType.STRUCT, result.getType());
// no assertion on child column nullness
```
**Fix:** Consider adding a check that the child STRUCT's INT32 field is null (no value on wire).
**Diff:**
```diff
+ try (ColumnVector innerStruct = result.getChildColumnView(0).copyToColumnVector();
+      ColumnVector xCol = innerStruct.getChildColumnView(0).copyToColumnVector();
+      HostColumnVector hostX = xCol.copyToHost()) {
+   assertTrue(hostX.isNull(0), "zero-length nested message should yield null child field");
+ }
```
**Reference:** /home/haoyangl/code/spark-rapids-jni/src/test/java/com/nvidia/spark/rapids/jni/ProtobufTest.java:2843-2849 (testNullInputRowProducesNullStructRow verifies nullness on the struct row level — analogous pattern for field nullness).
*(Reviewer: test-coverage)*

---

### 📄 `src/test/java/com/nvidia/spark/rapids/jni/ProtobufTest.java`

#### 29. [🟡 SHOULD FIX] `testUnpackedRepeatedInt32` uses a single-row input with three identical field_number=1 occurrences

**Line:** 1750
**Issue:** `testUnpackedRepeatedInt32` uses a single-row input with three identical field_number=1 occurrences. It never tests the multi-row case where repeated field values span different rows and the per-row LIST offsets must partition them correctly. A bug in the exclusive_scan that builds list offsets (protobuf.cu:1004) would not be caught.
**Confidence:** 🟣❗ CERTAIN
**Code:**
```java
|
try (Table input = new Table.TestBuilder().column(new Byte[][]{row}).build()) {
  // only one row tested
```
**Fix:** The test at line 1947 (`testPermissiveRepeatedWrongWireTypeDoesNotCorruptFollowingRow`) uses two rows but only in a permissive-mode error scenario. Add a simple two-row happy-path assertion: row0 has [1,2,3] and row1 has [4,5] and verify the offsets partition them into two lists.
**Diff:**
```diff
+ // Additional multi-row verification for correct list offsets
+ Byte[] r0 = concat(
+     box(tag(1, WT_VARINT)), box(encodeVarint(1)),
+     box(tag(1, WT_VARINT)), box(encodeVarint(2)),
+     box(tag(1, WT_VARINT)), box(encodeVarint(3)));
+ Byte[] r1 = concat(
+     box(tag(1, WT_VARINT)), box(encodeVarint(4)),
+     box(tag(1, WT_VARINT)), box(encodeVarint(5)));
+ try (Table multiInput = new Table.TestBuilder().column(new Byte[][]{r0, r1}).build();
+      ColumnVector multiResult = decodeRaw(multiInput.getColumn(0),
+          new int[]{1}, new int[]{-1}, new int[]{0},
+          new int[]{Protobuf.WT_VARINT}, new int[]{DType.INT32.getTypeId().getNativeId()},
+          new int[]{Protobuf.ENC_DEFAULT}, new boolean[]{true}, new boolean[]{false},
+          new boolean[]{false}, new long[]{0}, new double[]{0.0}, new boolean[]{false},
+          new byte[][]{null}, new int[][]{null}, false)) {
+   try (ColumnVector list = multiResult.getChildColumnView(0).copyToColumnVector()) {
+     HostColumnVector hList = list.copyToHost();
+     // row 0 must have 3 elements, row 1 must have 2 elements
+     assertEquals(3, hList.getList(0).size());
+     assertEquals(2, hList.getList(1).size());
+   }
+ }
```
**Reference:** /home/haoyangl/code/spark-rapids-jni/src/main/cpp/src/protobuf/protobuf.cu:1003-1008 (exclusive_scan builds per-row LIST offsets)
*(Reviewer: test-coverage)*

#### 30. [🟡 SHOULD FIX] `testNestedMessage` asserts only that the result is non-null and has type STRUCT, but never reads the decoded child value (inner.x = 42)

**Line:** 1910
**Issue:** `testNestedMessage` asserts only that the result is non-null and has type STRUCT, but never reads the decoded child value (inner.x = 42). A bug that decodes inner.x to 0 or any wrong value would not be caught. Compare with `verifyDeepNesting` (line 2945) which correctly reads the leaf value.
**Confidence:** 🟣❗ CERTAIN
**Code:**
```java
|
assertNotNull(result);
assertEquals(DType.STRUCT, result.getType());
// closes here — inner.x = 42 never verified
```
**Fix:** Add assertions that drill into result's child (the inner STRUCT), then into its INT32 child and verify the value equals 42, following the same pattern used in verifyDeepNesting at line 3003-3007.
**Diff:**
```diff
- assertNotNull(result);
- assertEquals(DType.STRUCT, result.getType());
+ assertNotNull(result);
+ assertEquals(DType.STRUCT, result.getType());
+ try (ColumnVector innerStruct = result.getChildColumnView(0).copyToColumnVector();
+      ColumnVector xCol = innerStruct.getChildColumnView(0).copyToColumnVector();
+      HostColumnVector hostX = xCol.copyToHost()) {
+   assertEquals(DType.STRUCT, innerStruct.getType());
+   assertEquals(DType.INT32, xCol.getType());
+   assertEquals(42, hostX.getInt(0));
+ }
```
**Reference:** /home/haoyangl/code/spark-rapids-jni/src/test/java/com/nvidia/spark/rapids/jni/ProtobufTest.java:3003-3007 (verifyDeepNesting reads and asserts the leaf value)
*(Reviewer: test-coverage)*

#### 31. [🟡 SHOULD FIX] `testNestedMessage` (lines 1909-1945) encodes `inner.x = 42` but the assertions only check the top-level type (`DType.STRUCT`) — the actual decoded value 42 in the child STRUCT's INT32 field is never read or verified

**Line:** 1941
**Issue:** `testNestedMessage` (lines 1909-1945) encodes `inner.x = 42` but the assertions only check the top-level type (`DType.STRUCT`) — the actual decoded value 42 in the child STRUCT's INT32 field is never read or verified. A mutation that zeroed all nested scalar values or returned the default instead of the decoded value would pass this test undetected.
**Confidence:** 🟣❗ CERTAIN
**Code:**
```java
assertNotNull(result);
assertEquals(DType.STRUCT, result.getType());
// inner.x decoded value 42 is never checked
```
**Fix:** After the type checks, drill into the child column and assert the decoded integer value equals 42.
**Diff:**
```diff
- assertNotNull(result);
- assertEquals(DType.STRUCT, result.getType());
+ assertNotNull(result);
+ assertEquals(DType.STRUCT, result.getType());
+ try (ColumnVector innerStruct = result.getChildColumnView(0).copyToColumnVector();
+      ColumnVector xCol = innerStruct.getChildColumnView(0).copyToColumnVector();
+      HostColumnVector hostX = xCol.copyToHost()) {
+   assertEquals(DType.INT32, xCol.getType());
+   assertEquals(42, hostX.getInt(0));
+ }
```
**Reference:** /home/haoyangl/code/spark-rapids-jni/src/test/java/com/nvidia/spark/rapids/jni/ProtobufTest.java:2936-2941 (testLargeFieldNumber correctly asserts the decoded integer value after drilling into the child).
*(Reviewer: test-coverage)*

#### 32. [🟢 SUGGESTION] `testPackedRepeatedDoubleWithMultipleFields` verifies the double column's decoded values but only spot-checks int_values count and bool_values indirectly (via the comment at line 1874)

**Line:** 2607
**Issue:** `testPackedRepeatedDoubleWithMultipleFields` verifies the double column's decoded values but only spot-checks int_values count and bool_values indirectly (via the comment at line 1874). A mutation that reverses the is_packed flag for int_values (so packed ints decode as a single LENGTH-DELIMITED string instead of individual ints) would not be caught for that column, because its content is never asserted.
**Confidence:** 🟠❗ LIKELY
**Code:**
```java
|
// Focus of this test is … exact decoded values are spot-checked only on the double column.
// Per-element-type round-trips are covered separately …
```
**Fix:** The comment is accurate — the per-type coverage exists elsewhere — but the multi-row packed-varint count for int_values at row 1 (15 occurrences, 150 bytes) is the primary stress case for the 2-byte length varint. Consider asserting at minimum the row counts for the int_values LIST: row 0 → 3 elements, row 1 → 15 elements, row 2 → 0 elements.
**Diff:**
```diff
+ // Verify int_values list sizes (especially row 1 with 2-byte length varint)
+ try (ColumnVector intListCol = result.getChildColumnView(1).copyToColumnVector()) {
+   assertEquals(DType.LIST, intListCol.getType());
+   try (HostColumnVector hIntList = intListCol.copyToHost()) {
+     assertEquals(3, ((List<?>)hIntList.getList(0)).size(), "row 0 int count");
+     assertEquals(15, ((List<?>)hIntList.getList(1)).size(), "row 1 int count (2-byte varint length)");
+     assertEquals(0, ((List<?>)hIntList.getList(2)).size(), "row 2 int count (omitted field)");
+   }
+ }
```
**Reference:** /home/haoyangl/code/spark-rapids-jni/src/test/java/com/nvidia/spark/rapids/jni/ProtobufTest.java:1874-1876 (comment identifying the gap)
*(Reviewer: test-coverage)*

#### 33. [🟡 SHOULD FIX] The private helper `verifyDeepNesting(int numLevels)` is defined at line 2945 but is never called by any `@Test` method

**Line:** 2945
**Issue:** The private helper `verifyDeepNesting(int numLevels)` is defined at line 2945 but is never called by any `@Test` method. Deep nesting decode — a key feature of the nested struct path added in `protobuf.cu` lines 953-1138 — is therefore not exercised by the test suite at all. The helper builds realistic multi-level nested protobuf messages and already contains meaningful value assertions (e.g., `assertEquals(1, hostChild.getInt(0))` at line 3006). Activating it as a parameterized test would immediately cover the depth=2, depth=3, and depth=N code paths.
**Confidence:** 🟣❗ CERTAIN
**Code:**
```java
private void verifyDeepNesting(int numLevels) {   // line 2945 — never called
```
**Fix:** Add one or more `@Test` methods that call `verifyDeepNesting` with representative depth values (e.g., 2, 3, and the maximum supported depth).
**Diff:**
```diff
+ @Test
+ void testDeepNestingLevel2() { verifyDeepNesting(2); }
+
+ @Test
+ void testDeepNestingLevel3() { verifyDeepNesting(3); }
```
**Reference:** /home/haoyangl/code/spark-rapids-jni/src/test/java/com/nvidia/spark/rapids/jni/ProtobufTest.java:2945-3009 (method body with assertions); the analogous single-level nested decode is exercised by testNestedMessage at line 1909.
*(Reviewer: test-coverage)*

---

### Skipped Domains
refactor-completeness (false-positive substitution signal: numeric_limits/is_same_v tokens appear independently, no systematic rename), impact-surface (no changed public symbol has callers in unchanged files; new structs are leaf types with no inheritors)

---

### 🔍 Validation Notes

✅ **All 1 findings verified** — no corrections needed.

### Files Reviewed
- 📄 `src/main/cpp/src/SparkResourceAdaptorJni.cpp`
- 📄 `src/main/cpp/src/cast_string_to_float.cu`
- 📄 `src/main/cpp/src/charset_decode.cu`
- 📄 `src/main/cpp/src/protobuf/protobuf.cu`
- 📄 `src/main/cpp/src/protobuf/protobuf_builders.cu`
- 📄 `src/main/cpp/src/protobuf/protobuf_kernels.cu`
- 📄 `src/main/cpp/src/protobuf/protobuf_kernels.cuh`
- 📄 `src/main/cpp/tests/protobuf_helpers.cu`
- 📄 `src/main/java/com/nvidia/spark/rapids/jni/OrcTimezoneInfo.java`
- 📄 `src/test/java/com/nvidia/spark/rapids/jni/CastStringsTest.java`
- 📄 `src/test/java/com/nvidia/spark/rapids/jni/CharsetDecodeTest.java`
- 📄 `src/test/java/com/nvidia/spark/rapids/jni/ProtobufTest.java`
