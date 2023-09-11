#include <window.cuh>
#include <time.h>

int main(int argc, char* argv[])
{
	srand(time(NULL));
	Window& instance = Window::instance();
	Aquarium aquarium;
	instance.renderLoop(aquarium);
	instance.free();

	return 0;
}