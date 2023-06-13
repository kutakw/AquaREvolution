#ifndef WINDOW_H
#define WINDOW_H

#include <shader.h>
#include <cuda/helper_math.cuh>
#include <GLFW/glfw3.h>
#include <simulation/structs/aquarium.h>

class Window {
public:
	uint2 screen{ 1000, 1000 };
	GLFWwindow* window{ nullptr };
	Shader shader{ "shaders/texture.vs",  "shaders/texture.fs" };

	// x - bg, y = alga, z = fish
	uint3 VBO{};
	uint3 VAO{};
	uint3 EBO{};

private:
	static Window* instance_;
public:
	static Window& instance() {
		if (instance_ == nullptr) instance_ = new Window();
		return *instance_;
	}

	static void free() {
		if (instance_ != nullptr) {
			delete instance_;
			instance_ = nullptr;
		}
	}

	void renderLoop(Aquarium& aquarium);
private:
	Window();
	~Window();

	GLFWwindow* createGLWindow();
	void createBuffers();
	void createBackgroundBuffers();
	void createAlgaBuffers();
	void createFishBuffers();
	void cleanup();
	void processInput();
	void renderAquarium(Aquarium& aquarium);
};

#endif // !WINDOW_H
