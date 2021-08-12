//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Rakolcza Peter
// Neptun : IMO7KC
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"
 
//---------------------------
template<class T> struct Dnum { // Dual numbers for automatic derivation
    //---------------------------
    float f; // function value
    T d;  // derivatives
    Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
    Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
    Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
    Dnum operator*(Dnum r) {
        return Dnum(f * r.f, f * r.d + d * r.f);
    }
    Dnum operator/(Dnum r) {
        return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
    }
};
 
// Elementary functions prepared for the chain rule as well
template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f) * g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }
template<class T> Dnum<T> Tan(Dnum<T>  g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f) * g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f) * g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
    return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}
 
typedef Dnum<vec2> Dnum2;
 
const int tessellationLevel = 100;
bool switchCam = false;
std::vector<vec2> weights;
float m = 0.01f;
 
float randomNum(float LO, float HI) {
    return  LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
}
 
//---------------------------
struct Camera { // 3D camera
    //---------------------------
    vec3 wEye, wLookat, wVup;   // extrinsic
    float fov, asp, fp, bp;        // intrinsic
public:
    Camera() {
        asp = (float)windowWidth / windowHeight;
        fov = 75.0f * (float)M_PI / 180.0f;
        fp = 0.01; bp = 100;
    }
    mat4 V() { // view matrix: translates the center to the origin
        vec3 w = normalize(wEye - wLookat);
        vec3 u = normalize(cross(wVup, w));
        vec3 v = cross(w, u);
        return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
            u.y, v.y, w.y, 0,
            u.z, v.z, w.z, 0,
            0, 0, 0, 1);
    }
 
    mat4 P() { // projection matrix
        return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
            0, 1 / tan(fov / 2), 0, 0,
            0, 0, -(fp + bp) / (bp - fp), -1,
            0, 0, -2 * fp * bp / (bp - fp), 0);
    }
};
 
struct OrtographicCamera : Camera {
    float w, h, n, f;
    OrtographicCamera() {
        w = 2;
        h = 2;
        n = 0.01f;
        f = 100;
    }
    mat4 P() { // projection matrix
        return mat4(
            2 / w, 0, 0, 0,
            0, 2 / h, 0, 0,
            0, 0, -2 / (f - n), -(f + n) / (f - n),
            0, 0, 0, 1
        );
    }
};
 
//---------------------------
struct Material {
    //---------------------------
    vec3 kd, ks, ka;
    float shininess;
};
 
float length(const vec4& v) { return sqrtf(dot(v, v)); }
 
//---------------------------
struct Light {
    //---------------------------
    vec3 La, Le;
    vec4 wLightPos; // homogeneous coordinates, can be at ideal point
    vec4 original;
 
 
    vec4 qmul(vec4 q1, vec4 q2) {	// quaternion multiplication
        vec4 q;
        q.x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
        q.y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x);
        q.z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w);
        q.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);
 
        return q;
    }
 
    void Animate(float tstart, float tend, vec4 pivot) {
        float dt = tend;
 
        vec4 q = vec4(cosf(dt / 4), sinf(dt / 4) * cosf(dt) / 2, sinf(dt / 4) * sinf(dt) / 2, sinf(dt / 4) * sqrtf(3 / 4));
        q = q / length(q);
        vec4 qinv = vec4(-q.x, -q.y, -q.z, q.w);
        qinv = qinv / length(qinv);
 
        wLightPos = qmul(q, original - pivot);
        wLightPos = qmul(wLightPos, qinv);
        wLightPos = wLightPos + pivot;
    }
};
 
class RandomColor : public Texture {
    //---------------------------
public:
    RandomColor(const int width, const int height) : Texture() {
        std::vector<vec4> image(width * height);
        vec4 color = vec4(randomNum(0, 1), randomNum(0, 1), randomNum(0, 1), 1);
        for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
            image[y * width + x] = color;
        }
        create(width, height, image, GL_NEAREST);
    }
};
 
//---------------------------
class CheckerBoardTexture : public Texture {
    //---------------------------
public:
    CheckerBoardTexture(const int width, const int height) : Texture() {
        std::vector<vec4> image(width * height);
        const vec4 yellow(1, 1, 0, 1), blue(0, 0, 1, 1);
        for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
            image[y * width + x] = (x & 1) ^ (y & 1) ? yellow : blue;
        }
        create(width, height, image, GL_NEAREST);
    }
};
 
//---------------------------
struct RenderState {
    //---------------------------
    mat4               MVP, M, Minv, V, P;
    Material* material;
    std::vector<Light> lights;
    Texture* texture;
    vec3               wEye;
};
 
//---------------------------
class Shader : public GPUProgram {
    //---------------------------
public:
    virtual void Bind(RenderState state) = 0;
 
    void setUniformMaterial(const Material& material, const std::string& name) {
        setUniform(material.kd, name + ".kd");
        setUniform(material.ks, name + ".ks");
        setUniform(material.ka, name + ".ka");
        setUniform(material.shininess, name + ".shininess");
    }
 
    void setUniformLight(const Light& light, const std::string& name) {
        setUniform(light.La, name + ".La");
        setUniform(light.Le, name + ".Le");
        setUniform(light.wLightPos, name + ".wLightPos");
    }
};
 
//---------------------------
class PhongShader : public Shader {
    //---------------------------
    const char* vertexSource = R"(
    #version 330
    precision highp float;
    
    struct Light {
    vec3 La, Le;
    vec4 wLightPos;
    };
    
    uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
    uniform Light[8] lights;    // light sources
    uniform int   nLights;
    uniform vec3  wEye;         // pos of eye
    
    layout(location = 0) in vec3  vtxPos;            // pos in modeling space
    layout(location = 1) in vec3  vtxNorm;           // normal in modeling space
    layout(location = 2) in vec2  vtxUV;
    
    out vec3 wNormal;            // normal in world space
    out vec3 wView;             // view in world space
    out vec3 wLight[8];            // light dir in world space
    out vec2 texcoord;
    out vec3 pos;
    
    void main() {
    gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
    // vectors for radiance computation
    vec4 wPos = vec4(vtxPos, 1) * M;
    for(int i = 0; i < nLights; i++) {
    wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
    }
    wView  = wEye * wPos.w - wPos.xyz;
    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
    texcoord = vtxUV;
    pos = vtxPos;
    }
    )";
 
    // fragment shader in GLSL
    const char* fragmentSource = R"(
    #version 330
    precision highp float;
    
    struct Light {
    vec3 La, Le;
    vec4 wLightPos;
    };
    
    struct Material {
    vec3 kd, ks, ka;
    float shininess;
    };
    
    uniform Material material;
    uniform Light[8] lights;    // light sources
    uniform int   nLights;
    uniform sampler2D diffuseTexture;
    
    in  vec3 wNormal;       // interpolated world sp normal
    in  vec3 wView;         // interpolated world sp view
    in  vec3 wLight[8];     // interpolated world sp illum dir
    in  vec2 texcoord;
    in  vec3 pos;
    
    out vec4 fragmentColor; // output goes to frame buffer
    
    void main() {
    vec3 N = normalize(wNormal);
    vec3 V = normalize(wView);
    if (dot(N, V) < 0) N = -N;    // prepare for one-sided surfaces like Mobius or Klein
    vec3 texColor = texture(diffuseTexture, texcoord).rgb;
 
    //float kdd = 0.01 / pos.y;
    float kdd = floor(pos.y * 8) / 8 + 1;
    vec3 ka = material.ka * texColor * kdd;
    vec3 kd = material.kd * texColor * kdd;
    
    vec3 radiance = vec3(0, 0, 0);
    for(int i = 0; i < nLights; i++) {
    vec3 L = normalize(wLight[i]);
    vec3 H = normalize(L + V);
    float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
 
    float d = length(lights[i].wLightPos);
 
    // kd and ka are modulated by the texture
    radiance += ka * lights[i].La +
    (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le / pow(d, 2);
 
    }
    fragmentColor = vec4(radiance, 1);
    }
    )";
public:
    PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }
 
    void Bind(RenderState state) {
        Use();         // make this program run
        setUniform(state.MVP, "MVP");
        setUniform(state.M, "M");
        setUniform(state.Minv, "Minv");
        setUniform(state.wEye, "wEye");
        setUniform(*state.texture, std::string("diffuseTexture"));
        setUniformMaterial(*state.material, "material");
 
        setUniform((int)state.lights.size(), "nLights");
        for (unsigned int i = 0; i < state.lights.size(); i++) {
            setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
        }
    }
};
 
class PhongShader2 : public Shader {
    //---------------------------
    const char* vertexSource = R"(
    #version 330
    precision highp float;
    
    struct Light {
    vec3 La, Le;
    vec4 wLightPos;
    };
    
    uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
    uniform Light[8] lights;    // light sources
    uniform int   nLights;
    uniform vec3  wEye;         // pos of eye
    
    layout(location = 0) in vec3  vtxPos;            // pos in modeling space
    layout(location = 1) in vec3  vtxNorm;           // normal in modeling space
    layout(location = 2) in vec2  vtxUV;
    
    out vec3 wNormal;            // normal in world space
    out vec3 wView;             // view in world space
    out vec3 wLight[8];            // light dir in world space
    out vec2 texcoord;
    out vec3 pos;
    
    void main() {
    gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
    // vectors for radiance computation
    vec4 wPos = vec4(vtxPos, 1) * M;
    for(int i = 0; i < nLights; i++) {
    wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
    }
    wView  = wEye * wPos.w - wPos.xyz;
    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
    texcoord = vtxUV;
    pos = vtxPos;
    }
    )";
 
    // fragment shader in GLSL
    const char* fragmentSource = R"(
    #version 330
    precision highp float;
    
    struct Light {
    vec3 La, Le;
    vec4 wLightPos;
    };
    
    struct Material {
    vec3 kd, ks, ka;
    float shininess;
    };
    
    uniform Material material;
    uniform Light[8] lights;    // light sources
    uniform int   nLights;
    uniform sampler2D diffuseTexture;
    
    in  vec3 wNormal;       // interpolated world sp normal
    in  vec3 wView;         // interpolated world sp view
    in  vec3 wLight[8];     // interpolated world sp illum dir
    in  vec2 texcoord;
    in  vec3 pos;
    
    out vec4 fragmentColor; // output goes to frame buffer
    
    void main() {
    vec3 N = normalize(wNormal);
    vec3 V = normalize(wView);
    if (dot(N, V) < 0) N = -N;    // prepare for one-sided surfaces like Mobius or Klein
    vec3 texColor = texture(diffuseTexture, texcoord).rgb;
    vec3 ka = material.ka * texColor;
    vec3 kd = material.kd * texColor;
    
    vec3 radiance = vec3(0, 0, 0);
    for(int i = 0; i < nLights; i++) {
    vec3 L = normalize(wLight[i]);
    vec3 H = normalize(L + V);
    float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
 
    float d = length(lights[i].wLightPos);
 
    // kd and ka are modulated by the texture
    radiance += ka * lights[i].La +
    (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le / pow(d, 2);
 
    }
    fragmentColor = vec4(radiance, 1);
    }
    )";
public:
    PhongShader2() { create(vertexSource, fragmentSource, "fragmentColor"); }
 
    void Bind(RenderState state) {
        Use();         // make this program run
        setUniform(state.MVP, "MVP");
        setUniform(state.M, "M");
        setUniform(state.Minv, "Minv");
        setUniform(state.wEye, "wEye");
        setUniform(*state.texture, std::string("diffuseTexture"));
        setUniformMaterial(*state.material, "material");
 
        setUniform((int)state.lights.size(), "nLights");
        for (unsigned int i = 0; i < state.lights.size(); i++) {
            setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
        }
    }
};
 
//---------------------------
class Geometry {
    //---------------------------
protected:
    unsigned int vao, vbo;        // vertex array object
public:
    Geometry() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
    }
    virtual void Draw() = 0;
    ~Geometry() {
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
    }
};
 
//---------------------------
class ParamSurface : public Geometry {
    //---------------------------
    struct VertexData {
        vec3 position, normal;
        vec2 texcoord;
    };
 
    unsigned int nVtxPerStrip, nStrips;
public:
    ParamSurface() { nVtxPerStrip = nStrips = 0; }
 
    virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;
 
 
    VertexData GenVertexData(float u, float v) {
        VertexData vtxData;
        vtxData.texcoord = vec2(u, v);
        Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
        Dnum2 X, Y, Z;
        eval(U, V, X, Y, Z);
        vtxData.position = vec3(X.f, Y.f, Z.f);
        vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
        vtxData.normal = cross(drdU, drdV);
        return vtxData;
    }
 
    void create(int N = tessellationLevel, int M = tessellationLevel) {
        nVtxPerStrip = (M + 1) * 2;
        nStrips = N;
        std::vector<VertexData> vtxData;    // vertices on the CPU
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= M; j++) {
                vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
                vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
            }
        }
        glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
        // Enable the vertex attribute arrays
        glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
        glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
        glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
        // attribute array, components/attribute, component type, normalize?, stride, offset
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
    }
 
    void Draw() {
        glBindVertexArray(vao);
        for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
    }
};
 
//---------------------------
class Sphere : public ParamSurface {
    //---------------------------
public:
    Sphere() { create(); }
    void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
        U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
        X = Cos(U) * Sin(V); Y = Sin(U) * Sin(V); Z = Cos(V);
    }
};
 
class Sheet : public ParamSurface {
public:
    Sheet() { create(); }
    void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) override {
        U = U * 2 - 1;
        V = V * 2 - 1;
        X = U * 2;
        Z = V * 2;
        Y = 0;
 
        for (int i = 0; i < weights.size(); i++) {
            Dnum2 Hole = Pow((Pow(Pow(X - weights[i].y, 2) + Pow(Z - weights[i].x, 2), 0.5f) + 4.0f * 0.005f), -1.0f) * -1.0f;
            Hole.f *= (i+1) * m;
            Y = Y + Hole;
        }
    }
};
 
//---------------------------
struct Object {
    //---------------------------
    Shader* shader;
    Material* material;
    Texture* texture;
    Geometry* geometry;
    vec3 scale, translation, rotationAxis, velocity;
    float rotationAngle;
    bool sheet = false;
 
public:
    Object(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) :
        scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
        shader = _shader;
        texture = _texture;
        material = _material;
        geometry = _geometry;
    }
 
    virtual void SetModelingTransform(mat4& M, mat4& Minv) {
        M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
        Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
    }
 
    void Draw(RenderState state) {
        mat4 M, Minv;
        SetModelingTransform(M, Minv);
        state.M = M;
        state.Minv = Minv;
        state.MVP = state.M * state.V * state.P;
        state.material = material;
        state.texture = texture;
        shader->Bind(state);
        geometry->Draw();
    }
 
    virtual void Animate(float tstart, float tend) {
        if (!sheet) {
            translation = translation + velocity * 0.0002f;
            
            Dnum2 Y = 0;
            for (int i = 0; i < weights.size(); i++) {
                Dnum2 X = Dnum2(translation.x);
                Dnum2 Z = Dnum2(translation.y);
                Dnum2 Hole = Pow((Pow(Pow(X - weights[i].y, 2) + Pow(Z - weights[i].x, 2), 0.5f) + 4.0f * 0.005f), -1.0f) * -1.0f;
                Hole.f *= (i + 1) * m;
                Y = Y + Hole;
            }
            translation.y = Y.f + 0.1f;
        }
 
        if (translation.x > 2)
            translation = vec3(translation.x - 4, translation.y, translation.z);
        else if (translation.z > 2)
            translation = vec3(translation.x, translation.y, translation.z - 4);
        else if (translation.x < -2)
            translation = vec3(translation.x + 4, translation.y, translation.z);
        else if (translation.z < -2)
            translation = vec3(translation.x, translation.y, translation.z + 4);
 
    }
};
 
//---------------------------
struct Scene {
    //---------------------------
    std::vector<Object*> objects;
    Camera camera2; // 3D camera
    OrtographicCamera camera; // 3D camera
    std::vector<Light> lights;
 
    Object* sphereObject1;
    Object* sheetObject1;
 
    // Materials
    Material* material0 = new Material;
    Material* material1 = new Material;
 
    Texture* texture;
 
    void Build() {
        // Shaders
        Shader* phongShader = new PhongShader();
        Shader* phongShader2 = new PhongShader2();
 
        // Materials
        material0->kd = vec3(0.6f, 0.4f, 0.2f);
        material0->ks = vec3(4, 4, 4);
        material0->ka = vec3(0.1f, 0.1f, 0.1f);
        material0->shininess = 100;
 
        material1->kd = vec3(0.8f, 0.6f, 0.4f);
        material1->ks = vec3(0.3f, 0.3f, 0.3f);
        material1->ka = vec3(0.2f, 0.2f, 0.2f);
        material1->shininess = 10;
 
        // Geometries
        Geometry* sphere = new Sphere();
        Geometry* sheet = new Sheet();
 
        texture = new CheckerBoardTexture(20, 25);
 
        // Create objects by setting up their vertex data on the GPU
        sheetObject1 = new Object(phongShader, material1, texture, sheet);
        sheetObject1->translation = vec3(0, 0, 0);
        sheetObject1->scale = vec3(1, 1, 1);
        sheetObject1->sheet = true;
        objects.push_back(sheetObject1);
 
        // Create objects by setting up their vertex data on the GPU
        sphereObject1 = new Object(phongShader2, material0, new RandomColor(100, 100), sphere);
        sphereObject1->translation = vec3(-1.8f, 0.1f, -1.8f);
        sphereObject1->scale = vec3(0.1f, 0.1f, 0.1f);
        objects.push_back(sphereObject1);
 
        // Camera
        camera.wEye = vec3(0, 1, 0);
        camera.wLookat = vec3(0, 0, 0);
        camera.wVup = vec3(1, 0, 0);
 
        // Camera
        camera2.wEye = vec3(0, 1, 3);
        camera2.wLookat = vec3(0, 0, 0);
        camera2.wVup = vec3(0, 1, 0);
 
        // Lights
        lights.resize(2);
        lights[0].wLightPos = vec4(0, 0, 3, 1);    // ideal point -> directional light source
        lights[0].original = vec4(0, 0, 3, 1);   // ideal point -> directional light source
        lights[0].La = vec3(1, 1, 1);
        lights[0].Le = vec3(5, 5, 5);
 
        lights[1].wLightPos = vec4(-1, -1, 6, 1);    // ideal point -> directional light source
        lights[1].original = vec4(-1, -1, 6, 1);    // ideal point -> directional light source
        lights[1].La = vec3(1, 1, 1);
        lights[1].Le = vec3(6, 6, 6);
    }
 
    void Render() {
        RenderState state;
        if (switchCam) {
            state.wEye = camera2.wEye;
            state.V = camera2.V();
            state.P = camera2.P();
        } else {
            state.wEye = camera.wEye;
            state.V = camera.V();
            state.P = camera.P();
        }
        state.lights = lights;
        for (Object* obj : objects) obj->Draw(state);
    }
 
    void Animate(float tstart, float tend) {
        for (Object* obj : objects) obj->Animate(tstart, tend);
        lights[0].Animate(tstart, tend, lights[1].original);
        lights[1].Animate(tstart, tend, lights[0].original);
 
        if (switchCam && objects.size() > 2) {
            camera2.wEye = vec3(objects[objects.size() - 2]->translation.x + 0.1f, objects[objects.size() - 2]->translation.y + 0.1f, objects[objects.size() - 2]->translation.z) + 0.1f;
            camera2.wLookat = objects[objects.size() - 2]->velocity;
        }
    }
};
 
Scene scene;
// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    scene.Build();
}
 
// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0.5f, 0.5f, 0.8f, 1.0f);                            // background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
    scene.Render();
    glutSwapBuffers();                                    // exchange the two buffers
}
 
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) { 
    if (key == ' ') switchCam = !switchCam;
}
 
// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }
 
// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        vec2 v = vec2(0, 600) - vec2(-pX, pY);
        
        scene.objects[scene.objects.size()-1]->velocity = vec3(v.y, 0, v.x);
 
        // Create objects by setting up their vertex data on the GPU
        Object* sphereObject1 = new Object(new PhongShader2(), scene.material0, new RandomColor(100, 100), new Sphere());
        sphereObject1->translation = vec3(-1.8f, 0.1f, -1.8f);
        sphereObject1->scale = vec3(0.1f, 0.1f, 0.1f);
        sphereObject1->rotationAxis = vec3(0, 1, 0);
        scene.objects.push_back(sphereObject1);
    } else if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
        float cX = 2.0f * pX / windowWidth - 1;
        float cY = 1.0f - 2.0f * pY / windowHeight;
        //printf("%lf %lf\n", cX, cY);
        vec2 temp = vec2(2 * cX, 2 * cY);
        weights.push_back(temp);
 
        //scene.objects.erase(scene.objects.begin());
 
        // Create objects by setting up their vertex data on the GPU
        Object* sheetObject1 = new Object(new PhongShader(), scene.material1, scene.texture, new Sheet());
        sheetObject1->translation = vec3(0, 0, 0);
        sheetObject1->scale = vec3(1, 1, 1);
        sheetObject1->sheet = true;
        scene.objects[0] = sheetObject1;
    }
 
    glutPostRedisplay();
}
 
// Move mouse with key pressed
void onMouseMotion(int pX, int pY) { }
 
// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    static float tend = 0;
    const float dt = 0.1f; // dt is infinitesimal
    float tstart = tend;
    tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
 
    for (float t = tstart; t < tend; t += dt) {
        float Dt = fmin(dt, tend - t);
        scene.Animate(t, t + Dt);
    }
    glutPostRedisplay();
}
