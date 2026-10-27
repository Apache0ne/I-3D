// I-3D original compact ternary GEMV, 2 bits per weight.
// Encoding: 0=>0, 1=>+1, 2=>-1, 3=>invalid/treated as zero.
// 16 values/u32, row padded to ceil(K/16) words. Per-row f32 scale.
// Not byte-compatible with PrismML GGUF PQ2_0 or PTQ1_0.
struct Shape {rows:u32, cols:u32, row_words:u32, reserved:u32};
@group(0) @binding(0) var<storage,read> packed:array<u32>;
@group(0) @binding(1) var<storage,read> x:array<f32>;
@group(0) @binding(2) var<storage,read> scales:array<f32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
@group(0) @binding(4) var<uniform> shape:Shape;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
 let row=gid.x; if(row>=shape.rows){return;}
 var sum=0.0;
 for(var k:u32=0u;k<shape.cols;k=k+1u){
  let word=packed[row*shape.row_words+k/16u];
  let trit=(word>>((k%16u)*2u))&3u;
  let w=select(0.0,1.0,trit==1u)-select(0.0,1.0,trit==2u);
  sum=sum+w*x[k];
 }
 y[row]=sum*scales[row];
}
