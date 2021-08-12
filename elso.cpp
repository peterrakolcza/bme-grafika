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
 
#define FLT_MAX          3.402823466e+38F
#define FLT_MIN          1.175494351e-38F
 
// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char* const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers
 
	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec3 vp;	// Attrib Array 0
	layout(location = 1) in vec2 vertexUV;			// Attrib Array 1
 
	out vec2 texCoord;								// output attribute
 
	void main() {
		texCoord = vertexUV;
		gl_Position = vec4(vp.x/vp.z, vp.y/vp.z, 1, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";
 
// fragment shader in GLSL
const char* const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
 
	uniform sampler2D textureUnit;
 
	in vec2 texCoord;			// variable input: interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation
 
	void main() {
		fragmentColor = texture(textureUnit, texCoord);
	}
)";
 
GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU
unsigned int vbo[2];
vec2 uvs[20];
std::vector<std::vector<vec4>> textures;
Texture textureHomogen;
Texture texture;
int width = 8, height = 8;
 
float Lorentz(vec3 elso, vec3 masodik) {
	return (elso.x * masodik.x + elso.y * masodik.y - (elso.z * masodik.z));
}
 
float distance(vec3 elso, vec3 masodik) {
	float szorzat = (-1.0) * Lorentz(elso, masodik);
 
	if (szorzat < 1.0) szorzat = 1.0;
 
	return acoshf(szorzat);
}
 
/*
https://stackoverflow.com/questions/686353/random-float-number-generation?rq=1
*/
float randomNum(float LO, float HI) {
	return  LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
}
 
bool containsConnection(std::vector<int*>& connections, int test[2]) {
	for (int i = 0; i < connections.size(); i++) {
		if (connections[i][0] == test[0] && connections[i][1] == test[1]) return true;
	}
	return false;
}
 
struct Mouse {
	float x, y;
};
 
struct Edge {
	std::vector<vec3> edge;
	std::vector<int> pairs;
 
	void create(int size) {
 
		printf("\nRandom parok: \n");
 
		int debugSzamlalo = 0;
		std::vector<int*> connections;
		for (int i = 0; i < 62; i++) {
			auto tempNums = new int[2];
			auto tempNumsRev = new int[2];
			do {
				debugSzamlalo++;
				tempNums[0] = randomNum(0, size);
 
				tempNums[1] = randomNum(0, size);
				while (tempNums[0] == tempNums[1]) {
					tempNums[1] = randomNum(0, size);
				}
 
				tempNumsRev[0] = tempNums[1];
				tempNumsRev[1] = tempNums[0];
			} while (containsConnection(connections, tempNums) || containsConnection(connections, tempNumsRev));
 
			connections.push_back(tempNums);
			connections.push_back(tempNumsRev);
 
			pairs.push_back(tempNums[0]);
			pairs.push_back(tempNums[1]);
 
			printf("%d %d \n", tempNums[0], tempNums[1]);
		}
 
		printf("\nDebug szamlalo: %d\n", debugSzamlalo);
 
	}
 
	void updateEdges(std::vector<vec3>& node) {
		for (int i = 0; i < pairs.size(); i++) {
			edge.push_back(node[pairs[i]]);
		}
	}
 
	void draw() {
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		glBufferData(GL_ARRAY_BUFFER,
			sizeof(vec3) * edge.size(),
			&edge[0],
			GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0,
			3, GL_FLOAT, GL_FALSE,
			0, NULL);
 
		gpuProgram.setUniform(textureHomogen, "textureUnit");
 
		float MVPtransf[4][4] = { 1, 0, 0, 0,
							  0, 1, 0, 0,
							  0, 0, 1, 0,
							  0, 0, 0, 1 };
 
		int location = glGetUniformLocation(gpuProgram.getId(), "MVP");
		glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);
 
		glBindVertexArray(vao);  // Draw call
		glDrawArrays(GL_LINES, 0 /*startIdx*/, edge.size() /*# Elements*/);
		edge.clear();
	}
 
};
 
struct Node {
	std::vector<vec3> node;
	std::vector<vec3> netForce;
	std::vector<vec3> speed;
 
	void create() {
 
		for (int j = 0; j < 20; j++) {
			float fi = 2 * j * M_PI / 20;
			float x = cosf(fi) * 0.1;
			float y = sinf(fi) * 0.1;
			float z = sqrt(1 - pow(x, 2) - pow(y, 2));
			uvs[j] = vec2(x, y);
		}
 
		for (int i = 0; i < 50; i++) {
			netForce.push_back(vec3(0, 0, 0));
			speed.push_back(vec3(0, 0, 0));
 
			vec3 temp;
			temp.x = randomNum(-1.0f, 1.0f);
			temp.y = randomNum(-1.0f, 1.0f);
			temp.z = (float)sqrt(1 + temp.x * temp.x + temp.y * temp.y);
 
			node.push_back(temp);
		}
 
		glPointSize(1.0f);
 
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL); 
	}
 
	/*
	https://gist.github.com/linusthe3rd/803118
	https://www.youtube.com/watch?v=AOVVD1-Ars4
	*/
	void draw() {
 
		for (int i = 0; i < node.size(); i++) {
			vec3 circle[20];
			for (int j = 0; j < 20; j++) {
				float fi = 2 * j * M_PI / 20;
				float x = cosf(fi) * 0.1;
				float y = sinf(fi) * 0.1;
				float z = sqrt(1 - pow(x, 2) - pow(y, 2));
				circle[j] = node[i];
 
				vec3 diffVector(x / z, y / z, 1 / z);
				vec3 origo = vec3(0, 0, 1);
				float diff = distance(diffVector, origo);
 
				vec3 v = (diffVector - origo * coshf(diff)) / sinhf(diff);
 
				vec3 m1 = origo * coshf(diff / 4) + v * sinhf(diff / 4);
				vec3 m2 = origo * coshf(diff / 2) + v * sinhf(diff / 2);
 
 
				float diff1 = distance(m1, circle[j]);
				vec3 v1 = (-1) * circle[j] * coshf(diff1) / sinhf(diff1) + m1 / sinhf(diff1);
				circle[j] = circle[j] * coshf(2 * diff1) + v1 * sinhf(2 * diff1);
 
				diff1 = distance(m2, circle[j]);
				vec3 v2 = ((-1) * circle[j] * coshf(diff1) / sinhf(diff1)) + m2 * (1 / sinhf(diff1));
				circle[j] = circle[j] * coshf(2 * diff1) + v2 * sinhf(2 * diff1);
			}
 
			texture.create(width, height, textures[i]);
 
			glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
			glBufferData(GL_ARRAY_BUFFER,
				sizeof(vec3) * 20,
				&circle[0],
				GL_STATIC_DRAW);
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0,
				3, GL_FLOAT, GL_FALSE,
				0, NULL);
 
			float MVPtransf[4][4] = { 1, 0, 0, 0, 
								  0, 1, 0, 0,
								  0, 0, 1, 0,
								  0, 0, 0, 1 };
 
			int location = glGetUniformLocation(gpuProgram.getId(), "MVP");
			glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);
 
			gpuProgram.setUniform(texture, "textureUnit");
 
			glBindVertexArray(vao);  // Draw call
			glDrawArrays(GL_TRIANGLE_FAN, 0 /*startIdx*/, 20 /*# Elements*/);
		}
	}
};
 
Mouse mouse;
Node nodes;
Edge edges;
bool spacePressed = false;
bool mouseLeftPressed = false;
 
void moveGraph(float cX, float cY) {
	float x = cX - mouse.x;
	float y = cY - mouse.y;
	float w = sqrt(1 - x * x - y * y);
 
	if (mouseLeftPressed && (cX != mouse.x || cY != mouse.y)) {
		vec3 diffVector(x / w, y / w, 1 / w);
		vec3 origo = vec3(0, 0, 1);
		float diff = distance(diffVector, origo);
		if (diff < FLT_MAX) {
			vec3 v = (diffVector - origo * coshf(diff)) / sinhf(diff);
 
			vec3 m1 = origo * coshf(diff / 4) + v * sinhf(diff / 4);
			vec3 m2 = origo * coshf(diff / 2) + v * sinhf(diff / 2);
 
			for (int i = 0; i < nodes.node.size(); i++) {
				float diff1 = distance(m1, nodes.node[i]);
				vec3 v1 = (-1) * nodes.node[i] * coshf(diff1) / sinhf(diff1) + m1 / sinhf(diff1);
				nodes.node[i] = nodes.node[i] * coshf(2 * diff1) + v1 * sinhf(2 * diff1);
 
				diff1 = distance(m2, nodes.node[i]);
				vec3 v2 = ((-1) * nodes.node[i] * coshf(diff1) / sinhf(diff1)) + m2 * (1 / sinhf(diff1));
				nodes.node[i] = nodes.node[i] * coshf(2 * diff1) + v2 * sinhf(2 * diff1);
			}
		}
	}
}
 
bool intersect(vec3 p11, vec3 p22, vec3 q11, vec3 q22) {
	vec2 p1, p2, q1, q2;
 
	p1.x = p11.x; p1.y = p11.y;
	p2.x = p22.x; p2.y = p22.y;
	q1.x = q11.x; q1.y = q11.y;
	q2.x = q22.x; q2.y = q22.y;
	return (dot(cross(p2 - p1, q1 - p1), cross(p2 - p1, q2 - p1)) < 0 &&
		dot(cross(q2 - q1, p1 - q1), cross(q2 - q1, p2 - q1)) < 0);
}
 
bool contains(std::vector<vec2>& node, float x, float y) {
	for (int i = 0; i < node.size(); i++) {
		if (node[i].x == x && node[i].y == y) return true;
	}
	return false;
}
 
bool containsConnection(std::vector<int>& pairs, int test[2]) {
	for (int i = 1; i < pairs.size(); i += 2) {
		if (pairs[i - 1] == test[0] && pairs[i] == test[1]) return true;
	}
 
	return false;
}
 
float forceForRealConnections(float dStar, float distance) {
	return 50 * pow(distance - dStar, 3);
}
 
float forceForUnrealConnections(float distance) {
	return -1.0 / (3 * (distance));
}
 
void forceGraphToPos(float dStar) {
	for (int i = 0; i < nodes.node.size(); i++) {
		for (int j = 0; j < nodes.node.size(); j++) {
			float distanceBetweenTwoVec = distance(nodes.node[i], nodes.node[j]);
			if (distanceBetweenTwoVec != 0) {
				int testConnection[] = { i, j };
				float forceForNode;
 
				if (containsConnection(edges.pairs, testConnection))
					forceForNode = forceForRealConnections(dStar, distanceBetweenTwoVec);
				else
					forceForNode = forceForUnrealConnections(distanceBetweenTwoVec);
 
				vec3 newForce = (nodes.node[j] - nodes.node[i] * coshf(distanceBetweenTwoVec)) / sinhf(distanceBetweenTwoVec) * forceForNode;
				nodes.netForce[i] = nodes.netForce[i] + newForce;
			}
		}
		float gravitationForceMagnitude = distance(nodes.node[i], vec3(0, 0, 1));
		vec3 gravitation = ((-1) * nodes.node[i] * coshf(gravitationForceMagnitude) / sinhf(gravitationForceMagnitude) + vec3(0, 0, 1) / sinhf(gravitationForceMagnitude)) * (forceForRealConnections(0.0f, gravitationForceMagnitude));
		nodes.netForce[i] = nodes.netForce[i] + gravitation;
	}
 
	float m = 0.1;
	float p = 0.9;
	float t = 0.03;
 
	for (int i = 0; i < nodes.node.size(); i++) {
		nodes.netForce[i] = nodes.netForce[i] - nodes.speed[i] * p;
 
		nodes.speed[i] = nodes.speed[i] + nodes.netForce[i] / m * t;
		nodes.speed[i] = normalize(nodes.speed[i]);
		float deltaS = length(nodes.speed[i]);
 
		if(deltaS > FLT_MIN && abs(pow(nodes.node[i].x, 2) + pow(nodes.node[i].y, 2) - pow(nodes.node[i].z, 2)) < 2) 
			nodes.node[i] = nodes.node[i] * coshf(deltaS) + nodes.speed[i] * sinhf(deltaS);
 
		nodes.netForce[i] = vec3(0, 0, 0);
 
		nodes.speed[i].z = (float)sqrt(1 + pow(nodes.speed[i].x, 2) + pow(nodes.speed[i].y, 2));
	}
 
	edges.updateEdges(nodes.node);
 
	glutPostRedisplay();
}
 
void heuristic(float precision) {
	std::vector<vec2> best;
	int bestIntersects = 1000000000;
	for (int i = 0; i < nodes.node.size(); i++) {
		float bestPosX = nodes.node[i].x;
		float bestPosY = nodes.node[i].y;
		float bestPosZ = nodes.node[i].z;
		for (float j = -1.0; j < 1.0; j += precision) {
			for (float k = -1.0; k < 1.0; k += precision) {
				int intersects = 0;
 
				nodes.node[i].x = j;
				nodes.node[i].y = k;
				nodes.node[i].z = (float)sqrt(1 + pow(nodes.node[i].x, 2) + pow(nodes.node[i].y, 2));
				edges.updateEdges(nodes.node);
 
				for (int l = 1; l < edges.edge.size(); l += 2) {
					for (int m = l + 2; m < edges.edge.size(); m += 2) {
						if (intersect(edges.edge[l - 1], edges.edge[l], edges.edge[m - 1], edges.edge[m])) intersects++;
					}
				}
 
				if (bestIntersects > intersects && !contains(best, j, k)) {
					bestIntersects = intersects;
 
					bestPosX = j;
					bestPosY = k;
					bestPosZ = nodes.node[i].z;
				}
 
				edges.edge.clear();
			}
		}
 
		nodes.node[i].x = bestPosX;
		nodes.node[i].y = bestPosY;
		nodes.node[i].z = bestPosZ;
 
		vec2 temp;
		temp.x = bestPosX;
		temp.y = bestPosY;
		best.push_back(temp);
 
		edges.updateEdges(nodes.node);
	}
 
	glutPostRedisplay();
}
 
// Initialization, create an OpenGL context
void onInitialization() {
	//srand(NULL);
 
	glViewport(0, 0, windowWidth, windowHeight);
 
	glGenVertexArrays(1, &vao);	// get 1 vao id
	glBindVertexArray(vao);		// make it active
	glGenBuffers(2, vbo);	// Generate 2 buffer
 
	nodes.create();
 
	for (int i = 0; i < nodes.node.size(); i++) {
		std::vector<vec4> image(width * height);
		float color = randomNum(0, 1);
		float color2 = randomNum(0, 1);
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				if (x > width / 2 && y < height / 2) image[y * width + x] = vec4(color, color, color, 1);
				else if (x > width / 2 || y < height / 2) image[y * width + x] = vec4(randomNum(0, 1), randomNum(0, 1), randomNum(0, 1), randomNum(0, 1));
				else image[y * width + x] = vec4(color2, color2, color2, color2);
			}
		}
		textures.push_back(image);
	}
 
	std::vector<vec4> image(width * height);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			image[y * width + x] = vec4(0.6, 0.6, 0.6, 1);
		}
	}
	textureHomogen.create(width, height, image);
 
	edges.create(nodes.node.size());
 
	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}
 
// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer
 
	edges.updateEdges(nodes.node);
	edges.draw();
	nodes.draw();
 
	glutSwapBuffers(); // exchange buffers for double buffering
}
 
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();    
	if (key == ' ') spacePressed = !spacePressed;
}
 
// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}
 
// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
 
	moveGraph(cX, cY);
	glutPostRedisplay();
}
 
 
// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
 
	mouse.x = cX;
	mouse.y = cY;
 
	char* buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}
 
	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	}
 
	if (button == GLUT_LEFT_BUTTON) {
		if (state == GLUT_DOWN) mouseLeftPressed = true;
		else					mouseLeftPressed = false;
	}
}
 
bool temp = true;
// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
 
	
	if (temp && spacePressed) {
		heuristic(0.5);
		temp = false;
	}
	if (spacePressed) forceGraphToPos(10);
}
