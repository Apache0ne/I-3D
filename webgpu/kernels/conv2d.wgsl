// NHWC image convolution with KHKWCI x CO filters, f32, stride 1, same padding.
// Shape includes explicit dimensions so models can match PyTorch/ONNX layouts.
struct Shape {
  width:u32,height:u32,cin:u32,cout:u32,
  kh:u32,kw:u32,pad_y:u32,pad_x:u32
};
@group(0) @binding(0) var<storage,read> image:array<f32>;
@group(0) @binding(1) var<storage,read> filter:array<f32>;
@group(0) @binding(2) var<storage,read> bias:array<f32>;
@group(0) @binding(3) var<storage,read_write> output:array<f32>;
@group(0) @binding(4) var<uniform> shape:Shape;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
 let idx=gid.x;let count=shape.width*shape.height*shape.cout;
 if(idx>=count){return;}
 let out_c=idx%shape.cout;
 let pixel=idx/shape.cout;
 let x=pixel%shape.width;let y=pixel/shape.width;
 var sum=bias[out_c];
 for(var ky:u32=0u;ky<shape.kh;ky=ky+1u){
   let yy=i32(y)+i32(ky)-i32(shape.pad_y);
   if(yy<0||yy>=i32(shape.height)){continue;}
   for(var kx:u32=0u;kx<shape.kw;kx=kx+1u){
     let xx=i32(x)+i32(kx)-i32(shape.pad_x);
     if(xx<0||xx>=i32(shape.width)){continue;}
     let in_base=(u32(yy)*shape.width+u32(xx))*shape.cin;
     let weight_base=(ky*shape.kw+kx)*shape.cin*shape.cout;
     for(var c:u32=0u;c<shape.cin;c=c+1u){
       sum=sum+image[in_base+c]*filter[weight_base+c*shape.cout+out_c];
     }
   }
 }
 output[idx]=sum;
}
