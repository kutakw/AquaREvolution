#include <simulation/structs/algae.cuh>

thrust::tuple<thrust::zip_iterator<Algae::EntityIter>, thrust::zip_iterator<Algae::EntityIter>> Algae::Device::iter() {
	auto begin = thrust::make_zip_iterator(
		thrust::make_tuple(
		positions.begin(),
		directionVecs.begin(),
		alives.begin(),
		currentEnergy.begin()
	));
	auto end = thrust::make_zip_iterator(
		thrust::make_tuple(
		positions.end(),
		directionVecs.end(),
		alives.end(),
		currentEnergy.end()
	));

	return thrust::make_tuple(begin, end);
}

std::ostream& operator<<(std::ostream& stream, const Algae& algae) {
	stream.imbue(std::locale("pl_PL"));
	stream << "ALGAE;\n";
	stream << "POSITIONS_X;POSITIONS_Y;DIRECTIONS_X;DIRECTIONS_Y;\n";
	for (int i = 0; i < algae.host.positions.size(); i++) {
		stream 
			<< algae.host.positions[i].x << ";"
			<< algae.host.positions[i].y << ";"
			<< algae.host.directionVecs[i].x << ";"
			<< algae.host.directionVecs[i].y << ";"
			<< "\n";
	}
	return stream;
}
