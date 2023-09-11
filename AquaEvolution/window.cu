#include "window.cuh"
#include <glad/glad.h>
#include <glm/glm.hpp>


Window* Window::instance_ = nullptr;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void genVerticeAndIndices(float* vertices, size_t vertices_size,
	uint* indices, size_t indices_size,
	uint* VAO, uint* VBO, uint* EBO);

Window::Window() :
	window(createGLWindow())
{
	createBuffers();
}

Window::~Window() {
	cleanup();
	window = nullptr;
}

void dump_data(const Aquarium& aquarium, int iter) {
	std::string fish_path = "..\\Output\\fish_data";
	fish_path.append(std::to_string(iter));
	fish_path.append(".csv");
	std::ofstream fish_data(fish_path);
	if (!fish_data.good()) std::cout << "Couldnt dump fish data to file!\n";
	fish_data << *aquarium.fish;
	fish_data.close();
	std::cout << "Dumped to file " << fish_path << std::endl;
	std::string algae_path = "..\\Output\\algae_data";
	algae_path.append(std::to_string(iter));
	algae_path.append(".csv");
	std::ofstream algae_data(algae_path);
	if (!algae_data.good()) std::cout << "Couldnt dump fish data to file!\n";
	algae_data << *aquarium.algae;
	algae_data.close();
	std::cout << "Dumped to file " << algae_path << std::endl;

}

void Window::renderLoop(Aquarium& aquarium) {
	int dump_iter = 0;
	int t = 0;

	aquarium.generateLife();
	aquarium.generateMutations();

	while (!glfwWindowShouldClose(window))
	{
		// input
		processInput();

		for(uint32_t i = 0; i < DISPLAY_FREQ; i++)
			aquarium.simulateGeneration();

		aquarium.fish->update(aquarium.fish->host, aquarium.fish->device);
		aquarium.algae->update(aquarium.algae->host, aquarium.algae->device);
		if (t == DUMP_FREQ) {
			dump_data(aquarium, dump_iter);
			dump_iter++;
			t = 0;
		}
		t += 1;

		// render scene
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		renderAquarium(aquarium);

		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

}

void Window::windowless_simulation(Aquarium& aquarium)
{
	int dump_iter = 0;
	int t = 0;

	aquarium.generateLife();
	aquarium.generateMutations();

	while (true) {
		aquarium.simulateGeneration();

		if (t == DUMP_FREQ) {
			aquarium.fish->update(aquarium.fish->host, aquarium.fish->device);
			aquarium.algae->update(aquarium.algae->host, aquarium.algae->device);
			dump_data(aquarium, dump_iter);
			dump_iter++;
			t = 0;
		}
		t += 1;
	}
}

void Window::renderAquarium(Aquarium& aquarium) {
	shader.use();

	// render background
	glBindVertexArray(VAO.x);
	shader.setMat4("mvp", glm::mat4(1.0f));
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

	auto& a = aquarium.algae->host;
	for (uint64_t i = 0; i < a.positions.size(); ++i)
	{
		if (!a.alives[i]) continue;
		auto& pos = a.positions[i];

		shader.setMat4("mvp", Shader::getMVP(
			a.positions[i],
			a.directionVecs[i],
			.5f)
		);

		// render algae
		glBindVertexArray(VAO.y);
		glDrawElements(GL_TRIANGLES, 18, GL_UNSIGNED_INT, 0);
	}

	auto& f = aquarium.fish->host;
	for (uint64_t i = 0; i < f.positions.size(); ++i)
	{
		//if (!f.alives[i]) continue;
		auto& pos = f.positions[i];
		auto& vec = f.directionVecs[i];

		shader.setMat4("mvp", Shader::getMVP(
			pos,
			//float2{ 1.0f, 0.0f },
			vec,
			.5f)
		);

		// render fish
		glBindVertexArray(VAO.z);
		glDrawElements(GL_TRIANGLES, 9, GL_UNSIGNED_INT, 0);
	}
}


GLFWwindow* Window::createGLWindow()
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(screen.x, screen.y, "AquaEvolution", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		exit(-1);
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	//glfwSetKeyCallback(window, key_callback);
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		exit(-1);
	}

	return window;
}

void Window::createBuffers() {
	this->createBackgroundBuffers();
	this->createAlgaBuffers();
	this->createFishBuffers();
}

void Window::createBackgroundBuffers() {
	constexpr glm::vec3 SURFACE = { 0.0f, 0.9f, 0.9f };
	constexpr glm::vec3 DEEPWATER = { 0.0f, 0.0f, 0.0f };

	// set up vertex data (and buffer(s)) and configure vertex attributes
	float vertices[] =
	{	// coords			// colors 
		 1.0f,  1.0f, 1.0f, SURFACE.r,		SURFACE.g,		SURFACE.b ,		// top right
		 1.0f, -1.0f, 0.0f, DEEPWATER.r,	DEEPWATER.g,	DEEPWATER.b ,	// bottom right
		-1.0f, -1.0f, 0.0f, DEEPWATER.r,	DEEPWATER.g,	DEEPWATER.b ,	// bottom left
		-1.0f,  1.0f, 0.0f, SURFACE.r,		SURFACE.g,		SURFACE.b ,		// top left 
	};
	uint indices[] =
	{
		0, 1, 3,   // first triangle
		1, 2, 3    // second triangle
	};

	genVerticeAndIndices(vertices, sizeof(vertices),
		indices, sizeof(indices),
		&VAO.x, &VBO.x, &EBO.x);
}

void Window::createAlgaBuffers() {
	constexpr glm::vec3 ALGAECOLOR = { 0.0f, 1.0f, 0.0f };

	// set up vertex data (and buffer(s)) and configure vertex attributes
	float vertices[] =
	{	// coords			// colors 
		 0.0f,  1.0f, 0.0f, ALGAECOLOR.r,	ALGAECOLOR.g,	ALGAECOLOR.b,	// top 
		 0.7f,  0.7f, 0.0f, ALGAECOLOR.r,	ALGAECOLOR.g,	ALGAECOLOR.b,	// top right
		 1.0f,  0.0f, 0.0f, ALGAECOLOR.r,	ALGAECOLOR.g,	ALGAECOLOR.b,	// left
		 0.7f, -0.7f, 0.0f, ALGAECOLOR.r,	ALGAECOLOR.g,	ALGAECOLOR.b,	// bottom right 
		 0.0f, -1.0f, 0.0f, ALGAECOLOR.r,	ALGAECOLOR.g,	ALGAECOLOR.b,	// bottom
		-0.7f, -0.7f, 0.0f, ALGAECOLOR.r,	ALGAECOLOR.g,	ALGAECOLOR.b,	// bottom left 
		-1.0f,  0.0f, 0.0f, ALGAECOLOR.r,	ALGAECOLOR.g,	ALGAECOLOR.b,	// left 
		-0.7f,  0.7f, 0.0f, ALGAECOLOR.r,	ALGAECOLOR.g,	ALGAECOLOR.b,	// top left 
	};
	unsigned int indices[] =
	{
		4, 5, 6,
		4, 6, 7,
		4, 7, 0,
		4, 0, 1,
		4, 1, 2,
		4, 2, 3
	};

	genVerticeAndIndices(vertices, sizeof(vertices),
		indices, sizeof(indices),
		&VAO.y, &VBO.y, &EBO.y);
}

void Window::createFishBuffers() {
	constexpr glm::vec3 FISHCOLOR1 = { 0.94f, 0.54f, 0.09f };
	constexpr glm::vec3 FISHCOLOR2 = { 0.85f, 0.7f, 0.2f };

	// set up vertex data (and buffer(s)) and configure vertex attributes
	float vertices[] =
	{	// coords			// colors 
		 0.0f,  1.0f, 0.0f, FISHCOLOR1.r,	FISHCOLOR1.g,	FISHCOLOR1.b,	// top 
		 0.3f,  0.0f, 0.0f, FISHCOLOR2.r,	FISHCOLOR2.g,	FISHCOLOR2.b,	// top right
		-0.3f,  0.0f, 0.0f, FISHCOLOR2.r,	FISHCOLOR2.g,	FISHCOLOR2.b,	// top left
		 0.0f, -0.5f, 0.0f, FISHCOLOR1.r,	FISHCOLOR1.g,	FISHCOLOR1.b,	// bottom 
		-0.4f, -1.0f, 0.0f, FISHCOLOR1.r,	FISHCOLOR1.g,	FISHCOLOR1.b,	// tail left
		 0.4f, -1.0f, 0.0f, FISHCOLOR1.r,	FISHCOLOR1.g,	FISHCOLOR1.b,	// tail right 
	};
	unsigned int indices[] =
	{
		2, 1, 0,
		2, 3, 1,
		4, 5, 3
	};

	genVerticeAndIndices(
		vertices, sizeof(vertices),
		indices, sizeof(indices),
		&VAO.z, &VBO.z, &EBO.z);
}

void Window::cleanup() {
	// de-allocate all openGL buffers once they've outlived their purpose:
	glDeleteVertexArrays(1, &VAO.x);
	glDeleteBuffers(1, &VBO.x);
	glDeleteBuffers(1, &EBO.x);

	glDeleteVertexArrays(1, &VAO.y);
	glDeleteBuffers(1, &VBO.y);
	glDeleteBuffers(1, &EBO.y);

	glDeleteVertexArrays(1, &VAO.z);
	glDeleteBuffers(1, &VBO.z);
	glDeleteBuffers(1, &EBO.z);

	// glfw: terminate, clearing all previously allocated GLFW resources.
	glfwTerminate();
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
void Window::processInput()
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, true);
	}
}


void genVerticeAndIndices(float* vertices, size_t vertices_size,
	uint* indices, size_t indices_size,
	uint* VAO, uint* VBO, uint* EBO) {

	glGenVertexArrays(1, VAO);
	glGenBuffers(1, VBO);
	glGenBuffers(1, EBO);

	glBindVertexArray(*VAO);

	glBindBuffer(GL_ARRAY_BUFFER, *VBO);
	glBufferData(GL_ARRAY_BUFFER, vertices_size, vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_size, indices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
}


// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);

	// update parameters
	Window& instance = Window::instance();
	instance.screen.x = width;
	instance.screen.y = height;
}
