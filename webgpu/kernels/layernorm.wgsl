// Per-row LayerNorm(X, gamma, beta, epsilon) (f32).
struct Shape {rows:u32, cols:u32, epsilon:f32, reserved:u32};
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read> gamma:array<f32>;
@group(0) @binding(2) var<storage,read> beta:array<f32>;
@group(0) @binding(3) var<storage,read_write> out:array<f32>;
@group(0) @binding(4) var<uniform> shape:Shape;
var<workgroup> sums:array<f32,64>;
var<workgroup> squares:array<f32,64>;
var<workgroup> mu:f32;
var<workgroup> invstd:f32;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
  let row=wg.x;let lane=lid.x;if(row>=shape.rows){return;}
  var s=0.0;var s2=0.0;
  for(var col=lane;col<shape.cols;col=col+64u){
    let v=x[row*shape.cols+col];s=s+v;s2=s2+v*v;
  }
  sums[lane]=s;squares[lane]=s2;workgroupBarrier();
  if(lane==0u){
    var all=0.0;var all2=0.0;
    for(var i:u32=0u;i<64u;i=i+1u){all=all+sums[i];all2=all2+squares[i];}
    mu=all/f32(shape.cols);
    invstd=inverseSqrt(max(0.0,all2/f32(shape.cols)-mu*mu)+shape.epsilon);
  }
  workgroupBarrier();
  for(var col=lane;col<shape.cols;col=col+64u){
    out[row*shape.cols+col]=(x[row*shape.cols+col]-mu)*invstd*gamma[col]+beta[col];
  }
}
