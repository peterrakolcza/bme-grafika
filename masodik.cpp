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
// Nev    : Rakolcza P�ter
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
 
const float epsilon = 0.0001f;
 
struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	bool rough, reflective, portal;
	vec3 F0;
};
 
struct RoughMaterial : Material {
	//---------------------------
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
		rough = true;
		reflective = false;
		portal = false;
	}
};
 
//---------------------------
struct SmoothMaterial : Material {
	//---------------------------
	SmoothMaterial(vec3 _F0, bool _portal) {
		F0 = _F0;
		rough = false;
		reflective = true;
		portal = _portal;
	}
};
 
struct Hit {
	float t;
	vec3 position, normal, planeCenter;
	Material* material;
	Hit() { t = -1; }
};
 
struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};
 
class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};
 
struct implicitSurface : public Intersectable {
	implicitSurface(Material* _material) {
		material = _material;
	}
 
	bool inSqrt3Sphere(vec3 p) {
		return p.x * p.x + p.y * p.y + p.z * p.z < 0.3 * 0.3;
	}
 
	Hit intersect(const Ray& ray) {
		float A = 0.1;
		float B = 0.7;
		float C = 0.08;
 
		Hit hit;
		hit.material = material;
 
		float a = (A * powf(ray.dir.x, 2)) + (B * powf(ray.dir.y, 2));
		float b = (2.0f * A * ray.start.x * ray.dir.x) + (2.0f * B * ray.start.y * ray.dir.y) - (C * ray.dir.z);
		float c = (A * powf(ray.start.x, 2)) + (B * powf(ray.start.y, 2)) - (C * ray.start.z);
		float discr = b * b - 4.0f * a * c;
 
		float sqrt_discr = sqrt(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
 
		vec3 p1 = ray.start + ray.dir * t1;
		vec3 p2 = ray.start + ray.dir * t2;
 
		if ((!inSqrt3Sphere(p1) && !inSqrt3Sphere(p2)) || (discr < 0)) {
			return hit;
		}
 
		if (!inSqrt3Sphere(p1) && inSqrt3Sphere(p2)) {
			hit.t = t2;
			hit.position = p2;
		}
 
		if (inSqrt3Sphere(p1) && !inSqrt3Sphere(p2)) {
			hit.t = t1;
			hit.position = p1;
		}
 
		if (inSqrt3Sphere(p1) && inSqrt3Sphere(p2)) {
			if (t1 < t2) {
				hit.t = t1;
				hit.position = p1;
			} else {
				hit.t = t2;
				hit.position = p2;
			}
		}
 
		vec3 u = normalize(vec3(1, 0, 2.0f * a * hit.position.x / c));
		vec3 v = normalize(vec3(0, 1, 2.0f * b * hit.position.y / c));
 
		hit.normal = normalize(cross(u, v));
		return hit;
	}
};
 
vec3 operator/(vec3 elso, vec3 masodik) {
	return vec3(elso.x / masodik.x, elso.y / masodik.y, elso.z / masodik.z);
}
 
struct Walls : public Intersectable {
	Material* material2;
 
	Walls(Material* _material, Material* _material2) {
		material = _material;
		material2 = _material2;
	}
 
	const int objFaces = 12;
 
	const float g = 0.618f, G = 1.618f;
	std::vector<vec3> v = {
			vec3(0, g, G), vec3(0, -g, G), vec3(0, -g, -G), vec3(0, g, -G),
			vec3(G, 0, g), vec3(-G, 0, g), vec3(-G, 0, -g), vec3(G, 0, -g),
			vec3(g, G, 0), vec3(-g, G, 0), vec3(-g, -G, 0), vec3(g, -G, 0),
			vec3(1, 1, 1), vec3(-1, 1, 1), vec3(-1, -1, 1), vec3(1, -1, 1),
			vec3(1, -1, -1), vec3(1, 1, -1), vec3(-1, 1, -1), vec3(-1, -1, -1)
	};
 
	std::vector<int> planes = {
		1, 2, 16,  1, 13, 9,  1, 14, 6,  2, 15, 11,  
		3, 4, 18,  3, 17, 12,  3, 20, 7,  19, 10, 9,  
		16, 12, 17,  5, 8, 18,  14, 10, 19,  6, 7, 20
	};
 
	std::vector<int> pl = {
		1, 2, 16, 5, 13,  1, 13, 9, 10, 14,
		1, 14, 6, 15, 2,  2, 15, 11, 12, 16,
		3, 4, 18, 8, 17,  3, 17, 12, 11, 20,
		3, 20, 7, 19, 4,  19, 10, 9, 18, 4,
		16, 12, 17, 8, 5,  5, 8, 18, 9, 13,
		14, 10, 19, 7, 6,  6, 7, 20, 11, 15
	};
 
	vec3 getObjPlane(int i, float scale, vec3& p, vec3& normal) {
		vec3 p1 = v[planes[3 * i] - 1], p2 = v[planes[3 * i + 1] - 1], p3 = v[planes[3 * i + 2] - 1];
		normal = cross(p2 - p1, p3 - p1);
		if (dot(p1, normal) < 0) normal = -normal;
		p = p1 * scale + vec3(0, 0, 0.03f);
 
		vec3 center = vec3(0, 0, 0);
		for (int j = 0; j < 5; j++)
			center = center + v[pl[i * 5 + j] - 1];
		return center * 0.2;
	}
 
 
	Hit intersect(const Ray& ray) {
		Hit hit;
 
		for (int i = 0; i < objFaces; i++) {
			vec3 p1, normal, center;
			center = getObjPlane(i, 1, p1, normal);
			float ti = fabs(dot(normal, ray.dir)) > epsilon ? dot(p1 - ray.start, normal) / dot(normal, ray.dir) : -1;
			if (ti <= epsilon || (ti > hit.t&& hit.t > 0)) continue;
			vec3 pintersect = ray.start + ray.dir * ti;
			bool outside = false;
			bool outside2 = false;
			for (int j = 0; j < objFaces; j++) {
				if (i == j) continue;
				vec3 p11, n;
				getObjPlane(j, 1.518 / 1.618, p11, n);
				if (dot(n, pintersect - p11) > 0) {
					outside = true;
				}
 
				getObjPlane(j, 1, p11, n);
				if (dot(n, pintersect - p11) > 0) {
					outside2 = true;
					break;
				}
			}
			
			hit.planeCenter = center;
			if (outside && !outside2) {
				hit.t = ti;
				hit.position = pintersect;
				hit.normal = normalize(normal);
				hit.material = material2;
			}
 
			if (!outside && !outside2) {
				hit.t = ti;
				hit.position = pintersect;
				hit.normal = normalize(normal);
				hit.material = material;
			}
 
 
		}
		return hit;
	}
};
 
 
struct Camera {
	//---------------------------
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(vup, w)) * f * tanf(fov / 2);
		up = normalize(cross(w, right)) * f * tanf(fov / 2);
	}
 
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
 
	void Animate(float dt) {
		eye = vec3((eye.x - lookat.x) * cos(dt) + (eye.z - lookat.z) * sin(dt) + lookat.x,
			eye.y,
			-(eye.x - lookat.x) * sin(dt) + (eye.z - lookat.z) * cos(dt) + lookat.z);
		set(eye, lookat, up, fov);
	}
};
 
struct Light {
	vec3 position;
	vec3 Le;
	Light(vec3 _position, vec3 _Le) {
		position = position;
		Le = _Le;
	}
};
 
 
float rnd() { return (float)rand() / RAND_MAX; }
 
void rotation(vec3& rotate, vec3 normal, float radian) {
	rotate = rotate * cosf(radian) + cross(rotate, normal) * sinf(radian) + normal * (dot(rotate, normal) * (1 - cosf(radian)));
}
 
float Fresnel(float n, float k) {
	return ((n - 1) * (n - 1) + k * k) / ((n + 1) * (n + 1) + k * k);
}
 
class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	std::vector<Material*> materials;
	Camera camera;
	vec3 La;
 
public:
	void build() {
		vec3 eye = vec3(0.3f, 0, 1.3f), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);
 
		La = vec3(0.529f, 0.808f, 0.922f);
		vec3 lighPos(0, 0, 0.65f), Le(1, 1, 1);
		lights.push_back(new Light(lighPos, Le));
 
		vec3 kd(0.3f, 0.2f, 0.1f), ks(10, 10, 10);
		materials.push_back(new RoughMaterial(kd, ks, 50));
		materials.push_back(new SmoothMaterial(vec3(Fresnel(0.17, 3.1), Fresnel(0.35, 2.7), Fresnel(1.5, 1.9)), false));
		materials.push_back(new SmoothMaterial(vec3(Fresnel(0, 1), Fresnel(0, 1), Fresnel(0, 1)), true));
 
		objects.push_back(new implicitSurface(materials[1]));
		objects.push_back(new Walls(materials[2], materials[0]));
	}
 
	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}
 
	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
 
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}
 
	vec3 trace(Ray ray, int depth = 0) {
		if (depth > 5) return La;
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance(0, 0, 0);
		if (hit.material->rough) {
			for (Light* light : lights) {
				outRadiance = hit.material->ka * La;
				vec3 lightdir = normalize(light->position - hit.position);
				float cosTheta = dot(hit.normal, lightdir);
				if (cosTheta > 0) {
					vec3 LeIn = light->Le / dot(light->position - hit.position, light->position - hit.position);
					outRadiance = outRadiance + LeIn * hit.material->kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + lightdir);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + LeIn * hit.material->ks * powf(cosDelta, hit.material->shininess);
				}
			}
		}
		if (hit.material->reflective) {
			vec3 reflectedDir;
			if (hit.material->portal) {
				vec3 temp = hit.position - hit.planeCenter;
				rotation(temp, hit.normal, 2 * M_PI / 5);
				hit.position = temp + hit.planeCenter;
 
				reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
				rotation(reflectedDir, hit.normal, 2 * M_PI / 5);
			}
			else {
				reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			}
 
			float cosa = 1 - dot(-ray.dir, hit.normal);
			vec3 one(1, 1, 1);
			vec3 F = hit.material->F0 + (one - hit.material->F0) * powf(1 - cosa, 5);
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
		}
		return outRadiance;
	}
 
	void Animate(float dt) { camera.Animate(dt); }
};
 
GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;
 
// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;
 
	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;
 
	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";
 
// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;
 
	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation
 
	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";
 
class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active
 
		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects
 
		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}
 
	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};
 
FullScreenTexturedQuad* fullScreenTexturedQuad;
 
// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
 
	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}
 
// Window has become invalid: Redraw
void onDisplay() {
	std::vector<vec4> image(windowWidth * windowHeight);
	long timeStart = glutGet(GLUT_ELAPSED_TIME);
	scene.render(image);
	long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));
 
	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
 
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}
 
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}
 
// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
 
}
 
// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}
 
// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}
 
// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	scene.Animate(0.01);
	glutPostRedisplay();
}
