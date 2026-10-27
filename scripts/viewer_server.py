"""Serve I-3D GLB outputs and pose authoring controls inside the Colab kernel proxy.

Only files in the configured workspace are exposed. No external service,
Blender GUI, authentication tokens, or second 3D reconstruction are required.
"""
from __future__ import annotations
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse, unquote
import argparse, json, os, mimetypes, re, math, hashlib, struct, zlib

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from i3d.artifacts import validate_glb
HTML=ROOT/"viewer"/"index.html"
EXT={".glb",".gltf",".bin",".png",".jpg",".jpeg",".webp",".hdr",".exr",".json",
     ".blend",".npz",".obj",".usda",".usd",".mp4"}
STAGES=[
    ("geometry","01_geometry.glb","Hi3D geometry"),
    ("paint","02_textured.glb","PBR textured character"),
    ("tokenrig","03_tokenrig.glb","TokenRig auto-rigged character"),
    ("soma","04_soma_character.glb","SOMA 30-joint character"),
    ("animated","07_animated.glb","MotionBricks animation"),
]

# Browser RMBG artifacts are allowed only from explicit same-origin POSTs.
RMBG_FILES={
    "source":"00_source_input.png",
    "mask":"00_alpha_mask.png",
    "cutout":"00_background_removed.png",
}
RMBG_MODELS={"patched_fp16","kn4666_fp16","kn4666_q4"}
RMBG_MAX_BYTES=48*1024*1024
PNG_SIGNATURE=b"\\x89PNG\\r\\n\\x1a\\n"


def validate_rmbg_png(blob):
    """Bounded PNG header and IHDR CRC validation without image decoding."""
    if not isinstance(blob,bytes) or len(blob)<45:
        raise ValueError("Empty or truncated PNG")
    if len(blob)>RMBG_MAX_BYTES:
        raise ValueError("PNG exceeds 48 MiB upload limit")
    if blob[:8]!=PNG_SIGNATURE:
        raise ValueError("Invalid PNG signature")
    length,chunk=struct.unpack(">I4s",blob[8:16])
    if length!=13 or chunk!=b"IHDR":
        raise ValueError("Missing PNG IHDR")
    header=blob[16:29]
    crc=struct.unpack(">I",blob[29:33])[0]
    if (zlib.crc32(b"IHDR"+header)&0xffffffff)!=crc:
        raise ValueError("PNG IHDR checksum mismatch")
    width,height,depth,color,comp,filtr,interlace=struct.unpack(">IIBBBBB",header)
    if not (1<=width<=8192 and 1<=height<=8192 and width*height<=40_000_000):
        raise ValueError("PNG resolution too large or invalid")
    if depth!=8 or color not in (2,6) or comp!=0 or filtr!=0 or interlace!=0:
        raise ValueError("Expected 8-bit RGB/RGBA noninterlaced PNG")
    return width,height


def save_rmbg_png(work,kind,blob,model):
    if kind not in RMBG_FILES or model not in RMBG_MODELS:
        raise ValueError("Unknown RMBG output/model selection")
    width,height=validate_rmbg_png(blob)
    work=Path(work).resolve()
    work.mkdir(parents=True,exist_ok=True)
    file=work/RMBG_FILES[kind]
    # A new source invalidates all prior outputs, never combine models/images.
    if kind=="source":
        for stale in ("mask","cutout"):
            (work/RMBG_FILES[stale]).unlink(missing_ok=True)
        (work/"00_rmbg_metadata.json").unlink(missing_ok=True)
    if kind=="cutout":
        for expected in ("source","mask"):
            existing=work/RMBG_FILES[expected]
            if not existing.is_file():
                raise ValueError("Source and alpha mask must be uploaded before cutout")
            if validate_rmbg_png(existing.read_bytes())!=(width,height):
                raise ValueError("RMBG image and alpha mask dimensions do not match")
    tmp=file.with_name(file.name+".tmp")
    try:
        with tmp.open("wb") as handle:
            handle.write(blob)
        os.replace(tmp,file)
    finally:
        tmp.unlink(missing_ok=True)
    if kind=="cutout":
        manifest={"version":1,"model":model,"width":width,"height":height,
                  "generated_at":datetime.now(timezone.utc).isoformat(),
                  "sha256":{name:hashlib.sha256((work/filename).read_bytes()).hexdigest()
                             for name,filename in RMBG_FILES.items()}}
        meta=work/"00_rmbg_metadata.json"
        temporary=meta.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest,indent=2)+"\\n",encoding="utf-8")
        os.replace(temporary,meta)
    return {"ok":True,"kind":kind,"file":file.name,"bytes":len(blob),
            "width":width,"height":height,"ready":(work/"00_rmbg_metadata.json").is_file()}

def safe_asset(work,name):
    if "/" in name or "\\" in name or not name or name.startswith("."):
        raise ValueError("Only direct output files are allowed")
    path=(work/name).resolve()
    if path.parent!=work.resolve() or path.suffix.lower() not in EXT:
        raise ValueError("Invalid asset path or extension")
    return path

def info(path):
    s=path.stat()
    return dict(name=path.name,url="/assets/"+path.name,size=s.st_size,
                mtime=s.st_mtime,modified=datetime.fromtimestamp(s.st_mtime,timezone.utc).isoformat(),
                extension=path.suffix.lower())

def _ready_asset(path):
    """A generated model is not ready merely because a file appeared."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if path.suffix.lower() == ".glb":
        try:
            validate_glb(path)
        except (OSError, ValueError, TypeError):
            return False
    return True


def manifest(work):
    assets=sorted([info(p) for p in work.iterdir() if _ready_asset(p) and not p.name.startswith(".")
                   and p.suffix.lower() in EXT],key=lambda item:item["name"])
    stages=[]
    for id,filename,label in STAGES:
        p=work/filename
        ready=_ready_asset(p)
        stages.append(dict(id=id,name=label,filename=filename,
                           available=ready,asset=info(p) if ready else None))
    reference=work/"soma_reference.json"
    names=[]
    if reference.exists():
        try:
            names=json.loads(reference.read_text(encoding="utf-8")).get("joint_names",[])
            if len(names)!=30:names=[]
        except (OSError,ValueError):
            names=[]
    poses=[]
    try:
        poses=list(json.loads((work/"viewer_poses.json").read_text())["poses"])
    except (KeyError,ValueError,OSError):
        pass
    source=next((a for a in assets if a["name"]=="00_source_input.png"),None)
    if source is None:
        source=next((a for a in assets if a["name"].startswith("original_input")),None)
    if source is None:
        source=next((a for a in assets if a["extension"] in
                     (".png",".jpeg",".jpg",".webp") and not
                     a["name"].startswith(("00_","screenshot_"))),None)
    rmbg_ready=(work/"00_rmbg_metadata.json").is_file() and all(
        (work/name).is_file() for name in RMBG_FILES.values())
    return dict(project="I-3D_web",coordinate_system="glTF right-handed Y-up",
                stages=stages,assets=assets,original_image=source,bone_names=names,
                background_removal_ready=rmbg_ready,
                background_cutout="/assets/00_background_removed.png" if rmbg_ready else None,
                saved_poses=sorted(poses),can_pose=stages[3]["available"] and bool(names))

def checked_transform(bone,name):
    for key in ("world","rest_world"):
        value=bone.get(key)
        if not isinstance(value,list) or len(value)!=16 or not all(
            isinstance(v,(int,float)) and math.isfinite(v) and abs(v)<1e6 for v in value
        ):
            raise ValueError(f"Invalid {key} matrix for {name}")
        # Three.js matrices are column-major. Their last row must be affine.
        if not all(abs(value[i]-(1 if i==15 else 0))<1e-4 for i in (3,7,11,15)):
            raise ValueError(f"Non-affine joint matrix for {name}")
    local=bone.get("local")
    if not isinstance(local,dict):raise ValueError("Missing local joint values")
    for key,n in (("position",3),("quaternion",4),("scale",3)):
        vec=local.get(key)
        if not isinstance(vec,list) or len(vec)!=n or not all(
            isinstance(v,(int,float)) and math.isfinite(v) for v in vec
        ):
            raise ValueError(f"Invalid {key} on {name}")
    if any(x<=0 for x in local["scale"]):
        raise ValueError(f"Negative or zero scale rejected on {name}")
    if abs(sum(v*v for v in local["quaternion"])-1)>0.02:
        raise ValueError(f"Invalid quaternion on {name}")

def save_pose(work,payload):
    if not isinstance(payload,dict) or payload.get("slot") not in ("start","target"):
        raise ValueError("Expected pose slot start or target")
    ref=work/"soma_reference.json"
    glb=work/"04_soma_character.glb"
    if not ref.is_file() or not glb.is_file():
        raise ValueError("SOMA model and reference are required")
    names=json.loads(ref.read_text(encoding="utf-8"))["joint_names"]
    bones=payload.get("bones")
    if len(names)!=30 or names[0]!="Hips" or not isinstance(bones,dict) or set(bones)!=set(names):
        raise ValueError("Require the canonical Hips-rooted 30 SOMA bones")
    for name,bone in bones.items():
        if not isinstance(bone,dict):raise ValueError("Bone data must be a dictionary")
        checked_transform(bone,name)
    file=work/"viewer_poses.json"
    data={"version":1,"coordinate_system":"GLTF_Y_UP_RH","poses":{}}
    if file.is_file():
        try:
            old=json.loads(file.read_text(encoding="utf-8"))
            if old.get("version")==1 and isinstance(old.get("poses"),dict):data=old
        except (OSError,ValueError):pass
    slot=payload["slot"]
    data["poses"][slot]={"bones":bones,"source":glb.name,"source_bytes":glb.stat().st_size,
                         "source_mtime":glb.stat().st_mtime,
                         "coordinate_system":"GLTF_Y_UP_RH",
                         "saved_at":datetime.now(timezone.utc).isoformat()}
    tmp=file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,separators=(",",":"),allow_nan=False),encoding="utf-8")
    os.replace(tmp,file)
    return dict(ok=True,saved=sorted(data["poses"]),file=file.name)

def handler_for(work):
    work=work.resolve()
    class Handler(BaseHTTPRequestHandler):
        server_version="I3D-Studio/1.0"
        def log_message(self,fmt,*args):
            print("[I-3D viewer]",fmt%args,flush=True)
        def respond(self,data,status=200):
            blob=json.dumps(data,allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type","application/json; charset=utf-8")
            self.send_header("Cache-Control","no-store")
            self.send_header("Content-Length",str(len(blob)))
            self.end_headers()
            if self.command!="HEAD":self.wfile.write(blob)
        def serve(self,file,head=False,content_type=None):
            if not file.is_file():return self.respond({"error":"File missing"},404)
            size=file.stat().st_size
            if size<=0:return self.respond({"error":"Empty file"},404)
            start,end,status=0,size-1,200
            rng=self.headers.get("Range","") if file!=HTML else ""
            if rng:
                match=re.fullmatch(r"bytes=(\d+)-(\d*)",rng)
                if not match:return self.respond({"error":"Invalid range"},416)
                start=int(match.group(1))
                end=min(int(match.group(2)),size-1) if match.group(2) else size-1
                if start>=size or end<start:return self.respond({"error":"Range unsatisfiable"},416)
                status=206
            self.send_response(status)
            self.send_header("Content-Type",content_type or mimetypes.guess_type(file.name)[0] or "application/octet-stream")
            self.send_header("Content-Length",str(end-start+1))
            self.send_header("Cache-Control","no-cache")
            self.send_header("Accept-Ranges","bytes")
            self.send_header("X-Content-Type-Options","nosniff")
            if status==206:self.send_header("Content-Range",f"bytes {start}-{end}/{size}")
            self.end_headers()
            if head:return
            with file.open("rb") as f:
                f.seek(start)
                remain=end-start+1
                while remain:
                    chunk=f.read(min(1024*1024,remain))
                    if not chunk:break
                    self.wfile.write(chunk)
                    remain-=len(chunk)
        def route(self,head=False):
            url=urlparse(self.path).path
            if url in ("/","/index.html"):
                return self.serve(HTML,head,content_type="text/html; charset=utf-8")
            if url in ("/webgpu", "/webgpu/"):
                return self.serve(ROOT/"webgpu"/"index.html",head,
                                  content_type="text/html; charset=utf-8")
            if url.startswith("/webgpu/"):
                # Branch-only static WGSL/JS kernel lab. No parent traversal.
                rel=unquote(url[len("/webgpu/"):])
                if (not rel or "\\" in rel or
                    any(part in ("", ".", "..") for part in rel.split("/"))):
                    return self.respond({"error":"Invalid WebGPU asset path"},400)
                candidate=(ROOT/"webgpu"/rel).resolve()
                lab=(ROOT/"webgpu").resolve()
                if lab not in candidate.parents or candidate.suffix.lower() not in (".js",".wgsl",".html",".css",".json"):
                    return self.respond({"error":"Unallowed WebGPU asset"},403)
                mime={".js":"text/javascript; charset=utf-8",
                      ".wgsl":"text/plain; charset=utf-8",
                      ".html":"text/html; charset=utf-8",
                      ".json":"application/json; charset=utf-8"}.get(candidate.suffix.lower())
                return self.serve(candidate,head,content_type=mime)
            if url=="/api/manifest":return self.respond(manifest(work))
            if url=="/api/poses":
                file=work/"viewer_poses.json"
                try:
                    return self.respond(json.loads(file.read_text(encoding="utf-8")) if file.is_file()
                                        else {"version":1,"poses":{}})
                except (OSError,ValueError) as e:return self.respond({"error":str(e)},500)
            if url.startswith("/assets/"):
                try:file=safe_asset(work,unquote(url[len("/assets/"):]))
                except ValueError as e:return self.respond({"error":str(e)},400)
                return self.serve(file,head)
            return self.respond({"error":"Unknown route"},404)
        def do_GET(self):return self.route()
        def do_HEAD(self):return self.route(head=True)
        def do_POST(self):
            url=urlparse(self.path)
            if url.path=="/api/rmbg":
                if self.headers.get("X-I3D-Action")!="rmbg":
                    return self.respond({"error":"Missing RMBG action header"},403)
                if self.headers.get_content_type()!="image/png":
                    return self.respond({"error":"Expected Content-Type: image/png"},415)
                length=int(self.headers.get("Content-Length","0") or 0)
                if length<45 or length>RMBG_MAX_BYTES:
                    return self.respond({"error":"PNG size outside allowed range"},413)
                from urllib.parse import parse_qs
                kind=parse_qs(url.query).get("kind",[""])[0]
                model=self.headers.get("X-I3D-Model","")
                try:
                    return self.respond(save_rmbg_png(work,kind,self.rfile.read(length),model))
                except (ValueError,OSError) as e:
                    return self.respond({"error":str(e)},400)
            if url.path!="/api/pose":return self.respond({"error":"Unknown route"},404)
            if self.headers.get("X-I3D-Action")!="pose":return self.respond({"error":"Missing action header"},403)
            length=int(self.headers.get("Content-Length",0))
            if length<2 or length>1024*1024:return self.respond({"error":"Pose payload exceeds 1 MB"},413)
            try:
                payload=json.loads(self.rfile.read(length))
                return self.respond(save_pose(work,payload))
            except (OSError,ValueError,TypeError,KeyError) as e:
                return self.respond({"error":str(e)},400)
    return Handler

def launch(workspace,port=8765,host="0.0.0.0"):
    from threading import Thread
    work=Path(workspace).resolve()
    work.mkdir(parents=True,exist_ok=True)
    server=ThreadingHTTPServer((host,port),handler_for(work))
    Thread(target=server.serve_forever,daemon=True).start()
    return server

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--workspace",type=Path,required=True)
    p.add_argument("--port",type=int,default=8765)
    a=p.parse_args()
    print(f"I-3D viewer listening on port {a.port} for {a.workspace}",flush=True)
    server=ThreadingHTTPServer(("0.0.0.0",a.port),handler_for(a.workspace))
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()

if __name__=="__main__":main()
