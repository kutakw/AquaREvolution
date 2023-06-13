#include <window.h>

int main(int argc, char* argv[])
{
	Window& instance = Window::instance();
	Aquarium aquarium;
	instance.renderLoop(aquarium);
	instance.free();

	return 0;
}