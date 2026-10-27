// Signed int4 in 2's-complement nibbles; row-padded to ceil(K/8) u32.
// Y[row]=sum_k scale[row]*W_q4[row,k]*X[k].
struct Shape { rows:u32, cols:u32, row_words:u32, reserved:u32 };
@group(0) @binding(0) var<storage,read> packed:array<u32>;
@group(0) @binding(1) var<storage,read> x:array<f32>;
@group(0) @binding(2) var<storage,read> scales:array<f32>;
@group(0) @binding(3) var<storage,read_write> y:array<f32>;
@group(0) @binding(4) var<uniform> shape:Shape;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
  let row=gid.x;if(row>=shape.rows){return;}
  var acc=0.0;
  for(var k:u32=0u;k<shape.cols;k=k+1u){
    let word=packed[row*shape.row_words+k/8u];
    let nibble=(word>>((k%8u)*4u))&15u;
    let signed=i32(nibble)-select(0,16,nibble>=8u);
    acc=acc+f32(signed)*x[k];
  }
  y[row]=acc*scales[row];
}
