from pyray import (
    Vector2, Vector3, Vector4, Transform, Matrix, Camera3D, 
    Color, Rectangle, Model, ModelAnimation, Mesh, BoneInfo, 
    Texture, RenderTexture)
from raylib import *
from raylib.defines import *

import bvh
import quat
import numpy as np
import struct
import cffi
import re
import sys
from pathlib import Path
ffi = cffi.FFI()

BASE_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = BASE_DIR / "resources"
DEFAULT_HUMANML3D_BVH_PATH = BASE_DIR.parent / "humanml3d_272" / "bvh" / "000000.bvh"
DEFAULT_MODEL_BIN = "SMPL.bin"
DEFAULT_SMPL_MODEL_PATH = BASE_DIR.parent / "272-dim-Motion-Representation" / "body_models" / "human_model_files"
SMPL_MESH_TINT = Color(230, 172, 128, 255)
SMPL_MODEL_NAMES = [
    "Pelvis",
    "Left_hip",
    "Right_hip",
    "Spine1",
    "Left_knee",
    "Right_knee",
    "Spine2",
    "Left_ankle",
    "Right_ankle",
    "Spine3",
    "Left_foot",
    "Right_foot",
    "Neck",
    "Left_collar",
    "Right_collar",
    "Head",
    "Left_shoulder",
    "Right_shoulder",
    "Left_elbow",
    "Right_elbow",
    "Left_wrist",
    "Right_wrist",
    "Left_palm",
    "Right_palm",
]


def resource_path_bytes(name: str) -> bytes:
    return str(RESOURCES_DIR / name).encode()


def load_shader_source(path: Path) -> str:
    source = path.read_text(encoding="utf-8")

    # The bundled shaders target GLSL ES 3.0. Convert them at runtime for
    # desktop OpenGL contexts such as macOS's 4.1 core profile.
    if source.startswith("#version 300 es"):
        desktop_version = "410" if sys.platform == "darwin" else "330"
        source = source.replace("#version 300 es", f"#version {desktop_version} core", 1)
        source = re.sub(r"^\s*precision\s+\w+\s+float\s*;\s*$", "", source, flags=re.MULTILINE)
    return source


def LoadShaderCompat(vertex_name: str, fragment_name: str):
    return LoadShaderFromMemory(
        load_shader_source(RESOURCES_DIR / vertex_name).encode(),
        load_shader_source(RESOURCES_DIR / fragment_name).encode(),
    )


def quat_wxyz_to_raylib_vector4(rotation):
    return Vector4(rotation[1], rotation[2], rotation[3], rotation[0])


def infer_bvh_position_scale(bvh_data) -> float:
    offsets = np.asarray(bvh_data["offsets"], dtype=np.float32)
    if len(offsets) <= 1:
        return 1.0

    bone_lengths = np.linalg.norm(offsets[1:], axis=1)
    mean_bone_length = float(np.mean(bone_lengths))

    # HumanML3D BVH exported from SMPL is already in meters (~0.1-0.4 per bone).
    # Legacy BVH files in centimeters (~8-40 per bone) are converted to meters.
    return 0.01 if mean_bone_length > 2.0 else 1.0

#----------------------------------------------------------------------------------
# Camera
#----------------------------------------------------------------------------------

class Camera:

    def __init__(self):
        self.cam3d = Camera3D()
        self.cam3d.position = Vector3(2.0, 3.0, 5.0)
        self.cam3d.target = Vector3(-0.5, 1.0, 0.0)
        self.cam3d.up = Vector3(0.0, 1.0, 0.0)
        self.cam3d.fovy = 45.0
        self.cam3d.projection = CAMERA_PERSPECTIVE
        self.azimuth = 0.0
        self.altitude = 0.4
        self.distance = 4.0
        self.offset = Vector3Zero()
    
    def update(
        self,
        target,
        azimuthDelta,
        altitudeDelta,
        offsetDeltaX,
        offsetDeltaY,
        mouseWheel,
        dt):

        self.azimuth = self.azimuth + 1.0 * dt * -azimuthDelta
        self.altitude = Clamp(self.altitude + 1.0 * dt * altitudeDelta, 0.0, 0.4 * PI)
        self.distance = Clamp(self.distance +  20.0 * dt * -mouseWheel, 0.1, 100.0)
        
        rotationAzimuth = QuaternionFromAxisAngle(Vector3(0, 1, 0), self.azimuth)
        position = Vector3RotateByQuaternion(Vector3(0, 0, self.distance), rotationAzimuth)
        axis = Vector3Normalize(Vector3CrossProduct(position, Vector3(0, 1, 0)))

        rotationAltitude = QuaternionFromAxisAngle(axis, self.altitude)

        localOffset = Vector3(dt * offsetDeltaX, dt * -offsetDeltaY, 0.0)
        localOffset = Vector3RotateByQuaternion(localOffset, rotationAzimuth)

        self.offset = Vector3Add(self.offset, Vector3RotateByQuaternion(localOffset, rotationAltitude))

        cameraTarget = Vector3Add(self.offset, target)
        eye = Vector3Add(cameraTarget, Vector3RotateByQuaternion(position, rotationAltitude))

        self.cam3d.target = cameraTarget
        self.cam3d.position = eye        

#----------------------------------------------------------------------------------
# Shadow Maps
#----------------------------------------------------------------------------------

class ShadowLight:
    
    def __init__(self):
        
        self.target = Vector3Zero()
        self.position = Vector3Zero()
        self.up = Vector3(0.0, 1.0, 0.0)
        self.target = Vector3Zero()
        self.width = 0
        self.height = 0
        self.near = 0.0
        self.far = 1.0


def LoadShadowMap(width, height):

    target = RenderTexture()
    target.id = rlLoadFramebuffer()
    target.texture.width = width
    target.texture.height = height
    assert target.id != 0
    
    rlEnableFramebuffer(target.id)

    target.depth.id = rlLoadTextureDepth(width, height, False)
    target.depth.width = width
    target.depth.height = height
    target.depth.format = 19       #DEPTH_COMPONENT_24BIT?
    target.depth.mipmaps = 1
    rlFramebufferAttach(target.id, target.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0)
    assert rlFramebufferComplete(target.id)

    rlDisableFramebuffer()

    return target

def UnloadShadowMap(target):
    
    if target.id > 0:
        rlUnloadFramebuffer(target.id)
        

def BeginShadowMap(target, shadowLight):
    
    BeginTextureMode(target)
    ClearBackground(WHITE)
    
    rlDrawRenderBatchActive()      # Update and draw internal render batch

    rlMatrixMode(RL_PROJECTION)    # Switch to projection matrix
    rlPushMatrix()                 # Save previous matrix, which contains the settings for the 2d ortho projection
    rlLoadIdentity()               # Reset current matrix (projection)

    rlOrtho(
        -shadowLight.width/2, shadowLight.width/2, 
        -shadowLight.height/2, shadowLight.height/2, 
        shadowLight.near, shadowLight.far)

    rlMatrixMode(RL_MODELVIEW)     # Switch back to modelview matrix
    rlLoadIdentity()               # Reset current matrix (modelview)

    # Setup Camera view
    matView = MatrixLookAt(shadowLight.position, shadowLight.target, shadowLight.up)
    rlMultMatrixf(MatrixToFloatV(matView).v)      # Multiply modelview matrix by view matrix (camera)

    rlEnableDepthTest()            # Enable DEPTH_TEST for 3D    


def EndShadowMap():
    rlDrawRenderBatchActive()       # Update and draw internal render batch

    rlMatrixMode(RL_PROJECTION)     # Switch to projection matrix
    rlPopMatrix()                   # Restore previous matrix (projection) from matrix stack

    rlMatrixMode(RL_MODELVIEW)      # Switch back to modelview matrix
    rlLoadIdentity()                # Reset current matrix (modelview)

    rlDisableDepthTest()            # Disable DEPTH_TEST for 2D

    EndTextureMode()

def SetShaderValueShadowMap(shader, locIndex, target):
    if locIndex > -1:
        rlEnableShader(shader.id)
        slotPtr = ffi.new('int*'); slotPtr[0] = 10  # Can be anything 0 to 15, but 0 will probably be taken up
        rlActiveTextureSlot(slotPtr[0])
        rlEnableTexture(target.depth.id)
        rlSetUniform(locIndex, slotPtr, SHADER_UNIFORM_INT, 1)

#----------------------------------------------------------------------------------
# GBuffer
#----------------------------------------------------------------------------------

class GBuffer:
    
    def __init__(self):
        self.id = 0              # OpenGL framebuffer object id
        self.color = Texture()   # Color buffer attachment texture 
        self.normal = Texture()  # Normal buffer attachment texture 
        self.depth = Texture()   # Depth buffer attachment texture


def LoadGBuffer(width, height):
    
    target = GBuffer()
    target.id = rlLoadFramebuffer()
    assert target.id
    
    rlEnableFramebuffer(target.id)

    target.color.id = rlLoadTexture(ffi.NULL, width, height, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8, 1)
    target.color.width = width
    target.color.height = height
    target.color.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    target.color.mipmaps = 1
    rlFramebufferAttach(target.id, target.color.id, RL_ATTACHMENT_COLOR_CHANNEL0, RL_ATTACHMENT_TEXTURE2D, 0)
    
    target.normal.id = rlLoadTexture(ffi.NULL, width, height, PIXELFORMAT_UNCOMPRESSED_R16G16B16A16, 1)
    target.normal.width = width
    target.normal.height = height
    target.normal.format = PIXELFORMAT_UNCOMPRESSED_R16G16B16A16
    target.normal.mipmaps = 1
    rlFramebufferAttach(target.id, target.normal.id, RL_ATTACHMENT_COLOR_CHANNEL1, RL_ATTACHMENT_TEXTURE2D, 0)
    
    target.depth.id = rlLoadTextureDepth(width, height, False)
    target.depth.width = width
    target.depth.height = height
    target.depth.format = 19       #DEPTH_COMPONENT_24BIT?
    target.depth.mipmaps = 1
    rlFramebufferAttach(target.id, target.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0)

    assert rlFramebufferComplete(target.id)

    rlDisableFramebuffer()

    return target


def UnloadGBuffer(target):

    if target.id > 0:
        rlUnloadFramebuffer(target.id)


def BeginGBuffer(target, camera):
    
    rlDrawRenderBatchActive()       # Update and draw internal render batch

    rlEnableFramebuffer(target.id)  # Enable render target
    rlActiveDrawBuffers(2) 

    # Set viewport and RLGL internal framebuffer size
    rlViewport(0, 0, target.color.width, target.color.height)
    rlSetFramebufferWidth(target.color.width)
    rlSetFramebufferHeight(target.color.height)

    ClearBackground(BLACK)

    rlMatrixMode(RL_PROJECTION)    # Switch to projection matrix
    rlPushMatrix()                 # Save previous matrix, which contains the settings for the 2d ortho projection
    rlLoadIdentity()               # Reset current matrix (projection)

    aspect = float(target.color.width)/float(target.color.height)

    # NOTE: zNear and zFar values are important when computing depth buffer values
    if camera.projection == CAMERA_PERSPECTIVE:

        # Setup perspective projection
        top = rlGetCullDistanceNear()*np.tan(camera.fovy*0.5*DEG2RAD)
        right = top*aspect

        rlFrustum(-right, right, -top, top, rlGetCullDistanceNear(), rlGetCullDistanceFar())

    elif camera.projection == CAMERA_ORTHOGRAPHIC:

        # Setup orthographic projection
        top = camera.fovy/2.0
        right = top*aspect

        rlOrtho(-right, right, -top,top, rlGetCullDistanceNear(), rlGetCullDistanceFar())

    rlMatrixMode(RL_MODELVIEW)     # Switch back to modelview matrix
    rlLoadIdentity()               # Reset current matrix (modelview)

    # Setup Camera view
    matView = MatrixLookAt(camera.position, camera.target, camera.up)
    rlMultMatrixf(MatrixToFloatV(matView).v)      # Multiply modelview matrix by view matrix (camera)

    rlEnableDepthTest()            # Enable DEPTH_TEST for 3D


def EndGBuffer(windowWidth, windowHeight):
    
    rlDrawRenderBatchActive()       # Update and draw internal render batch
    
    rlDisableDepthTest()            # Disable DEPTH_TEST for 2D
    rlActiveDrawBuffers(1) 
    rlDisableFramebuffer()          # Disable render target (fbo)

    rlMatrixMode(RL_PROJECTION)         # Switch to projection matrix
    rlPopMatrix()                   # Restore previous matrix (projection) from matrix stack
    rlLoadIdentity()                    # Reset current matrix (projection)
    rlOrtho(0, windowWidth, windowHeight, 0, 0.0, 1.0)

    rlMatrixMode(RL_MODELVIEW)          # Switch back to modelview matrix
    rlLoadIdentity()                    # Reset current matrix (modelview)


#----------------------------------------------------------------------------------
# Character Model and Animation
#----------------------------------------------------------------------------------

def FileRead(out, size, f):
    ffi.memmove(out, f.read(size), size)

def LoadCharacterModel(fileName):

    model = Model()
    model.transform = MatrixIdentity()
  
    with open(fileName, "rb") as f:
        
        model.materialCount = 1
        model.materials = MemAlloc(model.materialCount * ffi.sizeof(Mesh()))
        model.materials[0] = LoadMaterialDefault()

        model.meshCount = 1
        model.meshMaterial = MemAlloc(model.meshCount * ffi.sizeof(Mesh()))
        model.meshMaterial[0] = 0

        model.meshes = MemAlloc(model.meshCount * ffi.sizeof(Mesh()))
        model.meshes[0].vertexCount = struct.unpack('I', f.read(4))[0]
        model.meshes[0].triangleCount = struct.unpack('I', f.read(4))[0]
        model.boneCount = struct.unpack('I', f.read(4))[0]

        model.meshes[0].boneCount = model.boneCount
        model.meshes[0].vertices = MemAlloc(model.meshes[0].vertexCount * 3 * ffi.sizeof("float"))
        model.meshes[0].texcoords = MemAlloc(model.meshes[0].vertexCount * 2 * ffi.sizeof("float"))
        model.meshes[0].normals = MemAlloc(model.meshes[0].vertexCount * 3 * ffi.sizeof("float"))
        model.meshes[0].boneIds = MemAlloc(model.meshes[0].vertexCount * 4 * ffi.sizeof("unsigned char"))
        model.meshes[0].boneWeights = MemAlloc(model.meshes[0].vertexCount * 4 * ffi.sizeof("float"))
        model.meshes[0].indices = MemAlloc(model.meshes[0].triangleCount * 3 * ffi.sizeof("unsigned short"))
        model.meshes[0].colors = MemAlloc(model.meshes[0].vertexCount * 4 * ffi.sizeof("unsigned char"))
        model.meshes[0].animVertices = MemAlloc(model.meshes[0].vertexCount * 3 * ffi.sizeof("float"))
        model.meshes[0].animNormals = MemAlloc(model.meshes[0].vertexCount * 3 * ffi.sizeof("float"))
        model.bones =  MemAlloc(model.boneCount * ffi.sizeof(BoneInfo()))
        model.bindPose =  MemAlloc(model.boneCount * ffi.sizeof(Transform()))
        
        FileRead(model.meshes[0].vertices, ffi.sizeof("float") * model.meshes[0].vertexCount * 3, f)
        FileRead(model.meshes[0].texcoords, ffi.sizeof("float") * model.meshes[0].vertexCount * 2, f)
        FileRead(model.meshes[0].normals, ffi.sizeof("float") * model.meshes[0].vertexCount * 3, f)
        FileRead(model.meshes[0].boneIds, ffi.sizeof("unsigned char") * model.meshes[0].vertexCount * 4, f)
        FileRead(model.meshes[0].boneWeights, ffi.sizeof("float") * model.meshes[0].vertexCount * 4, f)
        FileRead(model.meshes[0].indices, ffi.sizeof("unsigned short") * model.meshes[0].triangleCount * 3, f)
        vertexColors = np.full((model.meshes[0].vertexCount, 4), 255, dtype=np.uint8)
        ffi.memmove(model.meshes[0].colors, ffi.from_buffer(vertexColors), vertexColors.nbytes)
        ffi.memmove(model.meshes[0].animVertices, model.meshes[0].vertices, ffi.sizeof("float") * model.meshes[0].vertexCount * 3)
        ffi.memmove(model.meshes[0].animNormals, model.meshes[0].normals, ffi.sizeof("float") * model.meshes[0].vertexCount * 3)
        FileRead(model.bones, ffi.sizeof(BoneInfo()) * model.boneCount, f)
        FileRead(model.bindPose, ffi.sizeof(Transform()) * model.boneCount, f)
        
        model.meshes[0].boneMatrices = MemAlloc(model.boneCount * ffi.sizeof(Matrix()))
        for i in range(model.boneCount):
            model.meshes[0].boneMatrices[i] = MatrixIdentity()
    
    UploadMesh(ffi.addressof(model.meshes[0]), True)
    
    return model


#----------------------------------------------------------------------------------
# Debug Draw
#----------------------------------------------------------------------------------

def DrawTransform(position, rotation, scale):
    
    rotMatrix = QuaternionToMatrix(quat_wxyz_to_raylib_vector4(rotation))
  
    DrawLine3D(
        Vector3(*position),
        Vector3Add(Vector3(*position), Vector3(scale * rotMatrix.m0, scale * rotMatrix.m1, scale * rotMatrix.m2)),
        RED)
        
    DrawLine3D(
        Vector3(*position),
        Vector3Add(Vector3(*position), Vector3(scale * rotMatrix.m4, scale * rotMatrix.m5, scale * rotMatrix.m6)),
        GREEN)
        
    DrawLine3D(
        Vector3(*position),
        Vector3Add(Vector3(*position), Vector3(scale * rotMatrix.m8, scale * rotMatrix.m9, scale * rotMatrix.m10)),
        BLUE)

def DrawSkeleton(positions, rotations, parents, color):
    
    for i in range(len(positions)):
    
        DrawSphereWires(
            Vector3(*positions[i]),
            0.01,
            4,
            6,
            color)

        DrawTransform(positions[i], rotations[i], 0.1)

        if parents[i] != -1:
        
            DrawLine3D(
                Vector3(*positions[i]),
                Vector3(*positions[parents[i]]),
                color)


def LoadBVHAnimation(path: Path):
    data = bvh.load(str(path))
    positionScale = infer_bvh_position_scale(data)
    frameTime = max(float(data.get("frametime", 1.0 / 60.0)), 1e-6)
    parents = data['parents']
    localPositions = positionScale * data['positions'].copy().astype(np.float32)
    localRotations = quat.unroll(quat.from_euler(np.radians(data['rotations']), order=data['order']))
    globalRotations, globalPositions = quat.fk(localRotations, localPositions, parents)

    print(f"Loaded BVH: {path}")
    print(f"  position scale: {positionScale}")
    print(f"  playback fps: {1.0 / frameTime:.2f}")

    return {
        "names": data["names"],
        "parents": parents,
        "localPositions": localPositions,
        "localRotations": localRotations,
        "globalRotations": globalRotations,
        "globalPositions": globalPositions,
        "frameTime": frameTime,
    }


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float32)
    tris = vertices[faces]
    faceNormals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    faceNormals /= np.linalg.norm(faceNormals, axis=1, keepdims=True) + 1e-8
    for corner in range(3):
        np.add.at(normals, faces[:, corner], faceNormals)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    return normals.astype(np.float32)


def quat_wxyz_to_axis_angle(rotations: np.ndarray) -> np.ndarray:
    q = np.asarray(rotations, dtype=np.float32).copy()
    q /= np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8
    q[q[..., 0] < 0.0] *= -1.0

    xyz = q[..., 1:4]
    sinHalfAngle = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sinHalfAngle, q[..., 0:1])
    axisAngle = xyz * (angle / (sinHalfAngle + 1e-8))
    axisAngle = np.where(sinHalfAngle > 1e-6, axisAngle, 2.0 * xyz)
    return axisAngle.astype(np.float32)


def LoadSMPLRuntimeAnimation(bvhAnimation):
    import smplx
    import torch

    names = bvhAnimation["names"]
    nameToIndex = {name: i for i, name in enumerate(names)}
    missing = [name for name in SMPL_MODEL_NAMES if name not in nameToIndex]
    if missing:
        raise ValueError(f"BVH is missing SMPL joints required by smplx: {missing}")

    bvhToSmplOrder = np.asarray([nameToIndex[name] for name in SMPL_MODEL_NAMES], dtype=np.int32)
    axisAngles = quat_wxyz_to_axis_angle(bvhAnimation["localRotations"])[:, bvhToSmplOrder]

    model = smplx.create(
        model_path=str(DEFAULT_SMPL_MODEL_PATH),
        model_type="smpl",
        gender="NEUTRAL",
        batch_size=1,
    )
    model.eval()

    print("Loaded runtime SMPL model.")
    print(f"  model path: {DEFAULT_SMPL_MODEL_PATH}")

    return {
        "model": model,
        "torch": torch,
        "axisAngles": axisAngles,
        "translations": bvhAnimation["localPositions"][:, 0].astype(np.float32),
        "faces": model.faces.astype(np.int32),
    }


def EvaluateSMPLRuntimeFrame(runtimeAnimation, frame: int):
    torch = runtimeAnimation["torch"]
    axisAngles = runtimeAnimation["axisAngles"][frame:frame + 1]
    translation = runtimeAnimation["translations"][frame:frame + 1]
    zeroTranslation = torch.zeros((1, 3), dtype=torch.float32)

    with torch.no_grad():
        output = runtimeAnimation["model"](
            global_orient=torch.from_numpy(axisAngles[:, 0]),
            body_pose=torch.from_numpy(axisAngles[:, 1:24].reshape(1, -1)),
            transl=zeroTranslation,
        )

    vertices = output.vertices[0].detach().cpu().numpy().astype(np.float32)
    pelvis = output.joints[0, 0].detach().cpu().numpy().astype(np.float32)
    vertices += translation[0] - pelvis
    normals = vertex_normals(vertices, runtimeAnimation["faces"])
    return vertices, normals


def UpdateModelStaticMesh(model, vertices: np.ndarray, normals: np.ndarray):
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    normals = np.ascontiguousarray(normals, dtype=np.float32)

    ffi.memmove(model.meshes[0].vertices, ffi.from_buffer(vertices), vertices.nbytes)
    ffi.memmove(model.meshes[0].normals, ffi.from_buffer(normals), normals.nbytes)

    UpdateMeshBuffer(model.meshes[0], 0, ffi.from_buffer(vertices), vertices.nbytes, 0)
    UpdateMeshBuffer(model.meshes[0], 2, ffi.from_buffer(normals), normals.nbytes, 0)
    

#----------------------------------------------------------------------------------
# App
#----------------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Init Window
    
    screenWidth = 1280
    screenHeight = 720
    
    SetConfigFlags(FLAG_VSYNC_HINT)
    InitWindow(screenWidth, screenHeight, b"GenoViewPython")
    SetTargetFPS(60)

    # Shaders
    
    shadowShader = LoadShaderCompat("shadow.vs", "shadow.fs")
    shadowShaderLightClipNear = GetShaderLocation(shadowShader, b"lightClipNear")
    shadowShaderLightClipFar = GetShaderLocation(shadowShader, b"lightClipFar")
    
    basicShader = LoadShaderCompat("basic.vs", "basic.fs")
    basicShaderSpecularity = GetShaderLocation(basicShader, b"specularity")
    basicShaderGlossiness = GetShaderLocation(basicShader, b"glossiness")
    basicShaderCamClipNear = GetShaderLocation(basicShader, b"camClipNear")
    basicShaderCamClipFar = GetShaderLocation(basicShader, b"camClipFar")
    
    lightingShader = LoadShaderCompat("post.vs", "lighting.fs")
    lightingShaderGBufferColor = GetShaderLocation(lightingShader, b"gbufferColor")
    lightingShaderGBufferNormal = GetShaderLocation(lightingShader, b"gbufferNormal")
    lightingShaderGBufferDepth = GetShaderLocation(lightingShader, b"gbufferDepth")
    lightingShaderSSAO = GetShaderLocation(lightingShader, b"ssao")
    lightingShaderCamPos = GetShaderLocation(lightingShader, b"camPos")
    lightingShaderCamInvViewProj = GetShaderLocation(lightingShader, b"camInvViewProj")
    lightingShaderLightDir = GetShaderLocation(lightingShader, b"lightDir")
    lightingShaderSunColor = GetShaderLocation(lightingShader, b"sunColor")
    lightingShaderSunStrength = GetShaderLocation(lightingShader, b"sunStrength")
    lightingShaderSkyColor = GetShaderLocation(lightingShader, b"skyColor")
    lightingShaderSkyStrength = GetShaderLocation(lightingShader, b"skyStrength")
    lightingShaderGroundStrength = GetShaderLocation(lightingShader, b"groundStrength")
    lightingShaderAmbientStrength = GetShaderLocation(lightingShader, b"ambientStrength")
    lightingShaderExposure = GetShaderLocation(lightingShader, b"exposure")
    lightingShaderCamClipNear = GetShaderLocation(lightingShader, b"camClipNear")
    lightingShaderCamClipFar = GetShaderLocation(lightingShader, b"camClipFar")
    
    ssaoShader = LoadShaderCompat("post.vs", "ssao.fs")
    ssaoShaderGBufferNormal = GetShaderLocation(ssaoShader, b"gbufferNormal")
    ssaoShaderGBufferDepth = GetShaderLocation(ssaoShader, b"gbufferDepth")
    ssaoShaderCamView = GetShaderLocation(ssaoShader, b"camView")
    ssaoShaderCamProj = GetShaderLocation(ssaoShader, b"camProj")
    ssaoShaderCamInvProj = GetShaderLocation(ssaoShader, b"camInvProj")
    ssaoShaderCamInvViewProj = GetShaderLocation(ssaoShader, b"camInvViewProj")
    ssaoShaderLightViewProj = GetShaderLocation(ssaoShader, b"lightViewProj")
    ssaoShaderShadowMap = GetShaderLocation(ssaoShader, b"shadowMap")
    ssaoShaderShadowInvResolution = GetShaderLocation(ssaoShader, b"shadowInvResolution")
    ssaoShaderCamClipNear = GetShaderLocation(ssaoShader, b"camClipNear")
    ssaoShaderCamClipFar = GetShaderLocation(ssaoShader, b"camClipFar")
    ssaoShaderLightClipNear = GetShaderLocation(ssaoShader, b"lightClipNear")
    ssaoShaderLightClipFar = GetShaderLocation(ssaoShader, b"lightClipFar")
    ssaoShaderLightDir = GetShaderLocation(ssaoShader, b"lightDir")
    
    blurShader = LoadShaderCompat("post.vs", "blur.fs")
    blurShaderGBufferNormal = GetShaderLocation(blurShader, b"gbufferNormal")
    blurShaderGBufferDepth = GetShaderLocation(blurShader, b"gbufferDepth")
    blurShaderInputTexture = GetShaderLocation(blurShader, b"inputTexture")
    blurShaderCamInvProj = GetShaderLocation(blurShader, b"camInvProj")
    blurShaderCamClipNear = GetShaderLocation(blurShader, b"camClipNear")
    blurShaderCamClipFar = GetShaderLocation(blurShader, b"camClipFar")
    blurShaderInvTextureResolution = GetShaderLocation(blurShader, b"invTextureResolution")
    blurShaderBlurDirection = GetShaderLocation(blurShader, b"blurDirection")

    fxaaShader = LoadShaderCompat("post.vs", "fxaa.fs")
    fxaaShaderInputTexture = GetShaderLocation(fxaaShader, b"inputTexture")
    fxaaShaderInvTextureResolution = GetShaderLocation(fxaaShader, b"invTextureResolution")
    
    # Objects
    
    groundMesh = GenMeshPlane(20.0, 20.0, 10, 10)
    groundModel = LoadModelFromMesh(groundMesh)
    groundPosition = Vector3(0.0, -0.01, 0.0)
    
    characterModel = LoadCharacterModel(resource_path_bytes(DEFAULT_MODEL_BIN))
    characterPosition = Vector3(0.0, 0.0, 0.0)
    characterTint = SMPL_MESH_TINT
    characterModel.materials[0].maps[MATERIAL_MAP_DIFFUSE].color = WHITE
    
    # Animation
    
    # bvhData = bvh.load(str(RESOURCES_DIR / "ground1_subject1.bvh"))
    bvhAnimation = LoadBVHAnimation(DEFAULT_HUMANML3D_BVH_PATH)
    smplRuntimeAnimation = LoadSMPLRuntimeAnimation(bvhAnimation)

    parents = bvhAnimation["parents"]
    localPositions = bvhAnimation["localPositions"]
    globalRotations = bvhAnimation["globalRotations"]
    globalPositions = bvhAnimation["globalPositions"]
    bvhFrameTime = bvhAnimation["frameTime"]

    print("Using runtime SMPL playback.")
    sceneFloorY = float(np.min(EvaluateSMPLRuntimeFrame(smplRuntimeAnimation, 0)[0][:, 1]))
    groundPosition.y = sceneFloorY - 0.01
    print(f"Using scene floor y: {groundPosition.y:.4f}")
    
    animationFrame = 0
    animationTime = 0.0
    
    # Camera
    
    camera = Camera()
    
    rlSetClipPlanes(0.01, 50.0)
    
    # Shadows
    
    lightDir = Vector3Normalize(Vector3(0.35, -1.0, -0.35))
    
    shadowLight = ShadowLight()
    shadowLight.target = Vector3Zero()
    shadowLight.position = Vector3Scale(lightDir, -5.0)
    shadowLight.up = Vector3(0.0, 1.0, 0.0)
    shadowLight.width = 5.0
    shadowLight.height = 5.0
    shadowLight.near = 0.01
    shadowLight.far = 10.0
    
    shadowWidth = 1024
    shadowHeight = 1024
    shadowInvResolution = Vector2(1.0 / shadowWidth, 1.0 / shadowHeight)
    shadowMap = LoadShadowMap(shadowWidth, shadowHeight)    
    
    # GBuffer and Render Textures
    
    gbuffer = LoadGBuffer(screenWidth, screenHeight)
    lighted = LoadRenderTexture(screenWidth, screenHeight)
    ssaoFront = LoadRenderTexture(screenWidth, screenHeight)
    ssaoBack = LoadRenderTexture(screenWidth, screenHeight)
    
    # UI
    
    drawBoneTransformsPtr = ffi.new('bool*'); drawBoneTransformsPtr[0] = False
    drawHumanML3DSkeletonPtr = ffi.new('bool*'); drawHumanML3DSkeletonPtr[0] = False
    
    # Go
    
    while not WindowShouldClose():
    
        # Animation

        animationTime = (animationTime + GetFrameTime()) % (len(localPositions) * bvhFrameTime)
        animationFrame = min(int(animationTime / bvhFrameTime), len(localPositions) - 1)
        vertices, normals = EvaluateSMPLRuntimeFrame(smplRuntimeAnimation, animationFrame)
        UpdateModelStaticMesh(characterModel, vertices, normals)

        # Shadow Light Tracks Character
        
        hipPosition = Vector3(*globalPositions[animationFrame][0])
        
        shadowLight.target = Vector3(hipPosition.x, groundPosition.y, hipPosition.z)
        shadowLight.position = Vector3Add(shadowLight.target, Vector3Scale(lightDir, -5.0))

        # Update Camera
        
        camera.update(
            Vector3(hipPosition.x, 0.75, hipPosition.z),
            GetMouseDelta().x if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(0) else 0.0,
            GetMouseDelta().y if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(0) else 0.0,
            GetMouseDelta().x if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(1) else 0.0,
            GetMouseDelta().y if IsKeyDown(KEY_LEFT_CONTROL) and IsMouseButtonDown(1) else 0.0,
            GetMouseWheelMove(),
            GetFrameTime())
        
        # Render
        
        rlDisableColorBlend()
        
        BeginDrawing()
        
        # Render Shadow Maps
        
        BeginShadowMap(shadowMap, shadowLight)  
        
        lightViewProj = MatrixMultiply(rlGetMatrixModelview(), rlGetMatrixProjection())
        lightClipNear = rlGetCullDistanceNear()
        lightClipFar = rlGetCullDistanceFar()

        lightClipNearPtr = ffi.new("float*"); lightClipNearPtr[0] = lightClipNear
        lightClipFarPtr = ffi.new("float*"); lightClipFarPtr[0] = lightClipFar
        
        SetShaderValue(shadowShader, shadowShaderLightClipNear, lightClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(shadowShader, shadowShaderLightClipFar, lightClipFarPtr, SHADER_UNIFORM_FLOAT)
        
        groundModel.materials[0].shader = shadowShader
        DrawModel(groundModel, groundPosition, 1.0, WHITE)
        
        characterModel.materials[0].shader = shadowShader
        DrawModel(characterModel, characterPosition, 1.0, WHITE)
        
        EndShadowMap()
        
        # Render GBuffer
        
        BeginGBuffer(gbuffer, camera.cam3d)
        
        camView = rlGetMatrixModelview()
        camProj = rlGetMatrixProjection()
        camInvProj = MatrixInvert(camProj)
        camInvViewProj = MatrixInvert(MatrixMultiply(camView, camProj))
        camClipNear = rlGetCullDistanceNear()
        camClipFar = rlGetCullDistanceFar()

        camClipNearPtr = ffi.new("float*"); camClipNearPtr[0] = camClipNear
        camClipFarPtr = ffi.new("float*"); camClipFarPtr[0] = camClipFar

        specularityPtr = ffi.new('float*'); specularityPtr[0] = 0.5
        glossinessPtr = ffi.new('float*'); glossinessPtr[0] = 10.0
        
        SetShaderValue(basicShader, basicShaderSpecularity, specularityPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(basicShader, basicShaderGlossiness, glossinessPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(basicShader, basicShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(basicShader, basicShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)

        groundModel.materials[0].shader = basicShader
        DrawModel(groundModel, groundPosition, 1.0, Color(190, 190, 190, 255))
        
        characterModel.materials[0].shader = basicShader
        DrawModel(characterModel, characterPosition, 1.0, characterTint)
        
        EndGBuffer(screenWidth, screenHeight)
        
        # Render SSAO and Shadows
        
        BeginTextureMode(ssaoFront)
        
        BeginShaderMode(ssaoShader)
        
        SetShaderValueTexture(ssaoShader, ssaoShaderGBufferNormal, gbuffer.normal)
        SetShaderValueTexture(ssaoShader, ssaoShaderGBufferDepth, gbuffer.depth)
        SetShaderValueMatrix(ssaoShader, ssaoShaderCamView, camView)
        SetShaderValueMatrix(ssaoShader, ssaoShaderCamProj, camProj)
        SetShaderValueMatrix(ssaoShader, ssaoShaderCamInvProj, camInvProj)
        SetShaderValueMatrix(ssaoShader, ssaoShaderCamInvViewProj, camInvViewProj)
        SetShaderValueMatrix(ssaoShader, ssaoShaderLightViewProj, lightViewProj)
        SetShaderValueShadowMap(ssaoShader, ssaoShaderShadowMap, shadowMap)
        SetShaderValue(ssaoShader, ssaoShaderShadowInvResolution, ffi.addressof(shadowInvResolution), SHADER_UNIFORM_VEC2)
        SetShaderValue(ssaoShader, ssaoShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(ssaoShader, ssaoShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(ssaoShader, ssaoShaderLightClipNear, lightClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(ssaoShader, ssaoShaderLightClipFar, lightClipFarPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(ssaoShader, ssaoShaderLightDir, ffi.addressof(lightDir), SHADER_UNIFORM_VEC3)
        
        ClearBackground(WHITE)
        
        DrawTextureRec(
            ssaoFront.texture,
            Rectangle(0, 0, ssaoFront.texture.width, -ssaoFront.texture.height),
            Vector2(0.0, 0.0),
            WHITE)

        EndShaderMode()

        EndTextureMode()
        
        # Blur Horizontal
        
        BeginTextureMode(ssaoBack)
        
        BeginShaderMode(blurShader)
        
        blurDirection = Vector2(1.0, 0.0)
        blurInvTextureResolution = Vector2(1.0 / ssaoFront.texture.width, 1.0 / ssaoFront.texture.height)
        
        SetShaderValueTexture(blurShader, blurShaderGBufferNormal, gbuffer.normal)
        SetShaderValueTexture(blurShader, blurShaderGBufferDepth, gbuffer.depth)
        SetShaderValueTexture(blurShader, blurShaderInputTexture, ssaoFront.texture)
        SetShaderValueMatrix(blurShader, blurShaderCamInvProj, camInvProj)
        SetShaderValue(blurShader, blurShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(blurShader, blurShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(blurShader, blurShaderInvTextureResolution, ffi.addressof(blurInvTextureResolution), SHADER_UNIFORM_VEC2)
        SetShaderValue(blurShader, blurShaderBlurDirection, ffi.addressof(blurDirection), SHADER_UNIFORM_VEC2)

        DrawTextureRec(
            ssaoBack.texture,
            Rectangle(0, 0, ssaoBack.texture.width, -ssaoBack.texture.height),
            Vector2(0, 0),
            WHITE)

        EndShaderMode()

        EndTextureMode()
      
        # Blur Vertical
        
        BeginTextureMode(ssaoFront)
        
        BeginShaderMode(blurShader)
        
        blurDirection = Vector2(0.0, 1.0)
        
        SetShaderValueTexture(blurShader, blurShaderInputTexture, ssaoBack.texture)
        SetShaderValue(blurShader, blurShaderBlurDirection, ffi.addressof(blurDirection), SHADER_UNIFORM_VEC2)

        DrawTextureRec(
            ssaoFront.texture,
            Rectangle(0, 0, ssaoFront.texture.width, -ssaoFront.texture.height),
            Vector2(0, 0),
            WHITE)

        EndShaderMode()

        EndTextureMode()
      
        # Light GBuffer
        
        BeginTextureMode(lighted)
        
        BeginShaderMode(lightingShader)
        
        sunColor = Vector3(253.0 / 255.0, 255.0 / 255.0, 232.0 / 255.0)
        sunStrengthPtr = ffi.new('float*'); sunStrengthPtr[0] = 0.25
        skyColor = Vector3(174.0 / 255.0, 183.0 / 255.0, 190.0 / 255.0)
        skyStrengthPtr = ffi.new('float*'); skyStrengthPtr[0] = 0.15
        groundStrengthPtr = ffi.new('float*'); groundStrengthPtr[0] = 0.1
        ambientStrengthPtr = ffi.new('float*'); ambientStrengthPtr[0] = 1.0
        exposurePtr = ffi.new('float*'); exposurePtr[0] = 0.9
        
        SetShaderValueTexture(lightingShader, lightingShaderGBufferColor, gbuffer.color)
        SetShaderValueTexture(lightingShader, lightingShaderGBufferNormal, gbuffer.normal)
        SetShaderValueTexture(lightingShader, lightingShaderGBufferDepth, gbuffer.depth)
        SetShaderValueTexture(lightingShader, lightingShaderSSAO, ssaoFront.texture)
        SetShaderValue(lightingShader, lightingShaderCamPos, ffi.addressof(camera.cam3d.position), SHADER_UNIFORM_VEC3)
        SetShaderValueMatrix(lightingShader, lightingShaderCamInvViewProj, camInvViewProj)
        SetShaderValue(lightingShader, lightingShaderLightDir, ffi.addressof(lightDir), SHADER_UNIFORM_VEC3)
        SetShaderValue(lightingShader, lightingShaderSunColor, ffi.addressof(sunColor), SHADER_UNIFORM_VEC3)
        SetShaderValue(lightingShader, lightingShaderSunStrength, sunStrengthPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderSkyColor, ffi.addressof(skyColor), SHADER_UNIFORM_VEC3)
        SetShaderValue(lightingShader, lightingShaderSkyStrength, skyStrengthPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderGroundStrength, groundStrengthPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderAmbientStrength, ambientStrengthPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderExposure, exposurePtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderCamClipNear, camClipNearPtr, SHADER_UNIFORM_FLOAT)
        SetShaderValue(lightingShader, lightingShaderCamClipFar, camClipFarPtr, SHADER_UNIFORM_FLOAT)
        
        ClearBackground(RAYWHITE)
        
        DrawTextureRec(
            gbuffer.color,
            Rectangle(0, 0, gbuffer.color.width, -gbuffer.color.height),
            Vector2(0, 0),
            WHITE)
        
        EndShaderMode()        
        
        # Debug Draw
        
        BeginMode3D(camera.cam3d)
        
        if drawBoneTransformsPtr[0]:
            DrawSkeleton(
                globalPositions[animationFrame], 
                globalRotations[animationFrame], 
                parents, GRAY)

        if drawHumanML3DSkeletonPtr[0]:
            DrawSkeleton(
                globalPositions[animationFrame],
                globalRotations[animationFrame],
                parents,
                BLUE)
  
        EndMode3D()

        EndTextureMode()
        
        # Render Final with FXAA
        
        BeginShaderMode(fxaaShader)

        fxaaInvTextureResolution = Vector2(1.0 / lighted.texture.width, 1.0 / lighted.texture.height)
        
        SetShaderValueTexture(fxaaShader, fxaaShaderInputTexture, lighted.texture)
        SetShaderValue(fxaaShader, fxaaShaderInvTextureResolution, ffi.addressof(fxaaInvTextureResolution), SHADER_UNIFORM_VEC2)
        
        DrawTextureRec(
            lighted.texture,
            Rectangle(0, 0, lighted.texture.width, -lighted.texture.height),
            Vector2(0, 0),
            WHITE)
        
        EndShaderMode()
  
        # UI
  
        rlEnableColorBlend()
  
        GuiGroupBox(Rectangle(20, 10, 190, 180), b"Camera")

        GuiLabel(Rectangle(30, 20, 150, 20), b"Ctrl + Left Click - Rotate")
        GuiLabel(Rectangle(30, 40, 150, 20), b"Ctrl + Right Click - Pan")
        GuiLabel(Rectangle(30, 60, 150, 20), b"Mouse Scroll - Zoom")
        GuiLabel(Rectangle(30, 80, 150, 20), b"Target: [% 5.3f % 5.3f % 5.3f]" % (camera.cam3d.target.x, camera.cam3d.target.y, camera.cam3d.target.z))
        GuiLabel(Rectangle(30, 100, 150, 20), b"Offset: [% 5.3f % 5.3f % 5.3f]" % (camera.offset.x, camera.offset.y, camera.offset.z))
        GuiLabel(Rectangle(30, 120, 150, 20), b"Azimuth: %5.3f" % camera.azimuth)
        GuiLabel(Rectangle(30, 140, 150, 20), b"Altitude: %5.3f" % camera.altitude)
        GuiLabel(Rectangle(30, 160, 150, 20), b"Distance: %5.3f" % camera.distance)
  
        GuiGroupBox(Rectangle(screenWidth - 260, 10, 240, 120), b"Rendering")

        GuiCheckBox(Rectangle(screenWidth - 250, 20, 20, 20), b"Draw Transforms", drawBoneTransformsPtr)
        GuiCheckBox(Rectangle(screenWidth - 250, 45, 20, 20), b"Draw HumanML3D", drawHumanML3DSkeletonPtr)
        GuiLabel(Rectangle(screenWidth - 250, 70, 220, 20), b"Mode: Runtime SMPL mesh")

  
        EndDrawing()

    UnloadRenderTexture(lighted)
    UnloadRenderTexture(ssaoBack)
    UnloadRenderTexture(ssaoFront)
    UnloadRenderTexture(lighted)
    UnloadGBuffer(gbuffer)

    UnloadShadowMap(shadowMap)
    
    UnloadModel(characterModel)
    UnloadModel(groundModel)
    
    UnloadShader(fxaaShader)    
    UnloadShader(blurShader)    
    UnloadShader(ssaoShader) 
    UnloadShader(lightingShader)    
    UnloadShader(basicShader)
    UnloadShader(shadowShader)
    
    CloseWindow()
