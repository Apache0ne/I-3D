// Stable row-wise softmax: each workgroup cooperates across one row.
struct Shape {rows:u32, cols:u32, reserved0:u32, reserved1:u32};
@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read_write> out:array<f32>;
@group(0) @binding(2) var<uniform> shape:Shape;
var<workgroup> partial:array<f32,64>;
var<workgroup> vmax:f32;
var<workgroup> vsum:f32;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>) {
  let row=wg.x;let lane=lid.x;if(row>=shape.rows){return;}
  var maxv=-3.402823e38;
  for(var col=lane;col<shape.cols;col=col+64u){maxv=max(maxv,x[row*shape.cols+col]);}
  partial[lane]=maxv;workgroupBarrier();
  if(lane==0u){
    var maxall=-3.402823e38;
    for(var i:u32=0u;i<64u;i=i+1u){maxall=max(maxall,partial[i]);}
    vmax=maxall;
  }
  workgroupBarrier();
  var sum=0.0;
  for(var col=lane;col<shape.cols;col=col+64u){sum=sum+exp(x[row*shape.cols+col]-vmax);}
  partial[lane]=sum;workgroupBarrier();
  if(lane==0u){
    var s=0.0;for(var i:u32=0u;i<64u;i=i+1u){s=s+partial[i];}vsum=s;
  }
  workgroupBarrier();
  for(var col=lane;col<shape.cols;col=col+64u){
    out[row*shape.cols+col]=exp(x[row*shape.cols+col]-vmax)/vsum;
  }
}
