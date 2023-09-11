#include <window.cuh>
#include <time.h>
#include "config.h"

int main(int argc, char* argv[])
{
	srand(time(NULL));
	Aquarium aquarium;
	if (DISPLAY) 
	{
		Window& instance = Window::instance();
		instance.renderLoop(aquarium);
		instance.free();
	} 
	else 
	{
		Window::windowless_simulation(aquarium);
	}
	

	return 0;
}