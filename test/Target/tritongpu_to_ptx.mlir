// RUN: %PYTHON -m triton.tools.aot %s --target=ptx --sm=80 --ptx-version=63 | FileCheck %s
// CHECK-LABEL: // Generated by LLVM NVPTX Back-End
// CHECK: .version 6.3
// CHECK: .target sm_80
// CHECK: .address_size 64

#blocked0 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
module attributes {"triton_gpu.num-warps" = 2 : i32} {
  // CHECK-LABEL: global_load_store_vec4
  func @global_load_store_vec4(%arg0: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 4 : i32}, %arg3: i32) {
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    %3 = tt.splat %1 : (i32) -> tensor<256xi32, #blocked0>
    %4 = arith.addi %3, %2 : tensor<256xi32, #blocked0>
    %5 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked0>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>, #blocked0>
    %7 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked0>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>, #blocked0>

    // Load 4 elements from A with single one vectorized load instruction
    // CHECK: @${{.*}} ld.global.v4.b32 { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    // Load 4 elements from B with single one vectorized load instruction
    // CHECK: @${{.*}} ld.global.v4.b32 { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} }, [ ${{.*}} + 0 ];

    %9 = tt.load %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #blocked0>
    %10 = tt.load %8 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #blocked0>
    %11 = arith.addf %9, %10 : tensor<256xf32, #blocked0>
    %12 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked0>
    %13 = tt.addptr %12, %4 : tensor<256x!tt.ptr<f32>, #blocked0>

    // Store 4 elements to global with single one vectorized store instruction
    // CHECK: @$5 st.global.v4.b32 [ ${{.*}} + 0 ], { ${{.*}}, ${{.*}}, ${{.*}}, ${{.*}} };
    tt.store %13, %11 : tensor<256xf32, #blocked0>
    return
  }
}
