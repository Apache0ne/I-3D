// I-3D WebGPU f32 tiled row-major GEMM. C[M,N] = A[M,K] * B[K,N].
struct Shape { m:u32, n:u32, k:u32, reserved:u32 };
@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read> b:array<f32>;
@group(0) @binding(2) var<storage,read_write> c:array<f32>;
@group(0) @binding(3) var<uniform> shape:Shape;
var<workgroup> ta:array<array<f32,8>,8>;
var<workgroup> tb:array<array<f32,8>,8>;
@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>,
        @builtin(local_invocation_id) lid:vec3<u32>) {
  let row=gid.y; let col=gid.x;
  var value=0.0;
  for(var base:u32=0u;base<shape.k;base=base+8u){
    let ka=base+lid.x; let kb=base+lid.y;
    ta[lid.y][lid.x]=select(0.0,a[row*shape.k+ka],row<shape.m && ka<shape.k);
    tb[lid.y][lid.x]=select(0.0,b[kb*shape.n+col],kb<shape.k && col<shape.n);
    workgroupBarrier();
    for(var j:u32=0u;j<8u;j=j+1u){value=value+ta[lid.y][j]*tb[j][lid.x];}
    workgroupBarrier();
  }
  if(row<shape.m && col<shape.n){c[row*shape.n+col]=value;}
}
