#include <simulation/structs/fish.cuh>
#include <thrust/iterator/zip_iterator.h>

thrust::tuple<thrust::zip_iterator<Fish::EntityIter>, thrust::zip_iterator<Fish::EntityIter>> Fish::Device::iter() {
	auto begin = thrust::make_zip_iterator(
		thrust::make_tuple(
		positions.begin(),
		directionVecs.begin(),
		alives.begin(),
		currentEnergy.begin(),
		nextDecisions.begin(),
		eatenAlgaeId.begin(),
		energyParams.begin(),
		sightParams.begin(),
		velocity.begin()
	));
	auto end = thrust::make_zip_iterator(
		thrust::make_tuple(
		positions.end(),
		directionVecs.end(),
		alives.end(),
		currentEnergy.end(),
		nextDecisions.end(),
		eatenAlgaeId.end(),
		energyParams.end(),
		sightParams.end(),
		velocity.end()
	));

	return thrust::make_tuple(begin, end);
}

std::ostream& operator<<(std::ostream& stream, const Fish& fish) {
	stream.imbue(std::locale("pl_PL"));
	stream << "FISHES;\n";
	stream << "POSITIONS_X;POSITIONS_Y;DIRECTIONS_X;DIRECTIONS_Y;MAX_ENERGY;ENERGY_DECAY;SIGHT_DISTANCE;SIGHT_ANGLE;VELOCITY;\n";
	for (int i = 0; i < fish.host.positions.size(); i++) {
		stream 
			<< fish.host.positions[i].x << ";"
			<< fish.host.positions[i].y << ";"
			<< fish.host.directionVecs[i].x << ";"
			<< fish.host.directionVecs[i].y << ";"
			<< fish.host.energyParams[i].x << ";"
			<< fish.host.energyParams[i].y << ";"
			<< fish.host.sightParams[i].x << ";"
			<< fish.host.sightParams[i].y << ";"
			<< fish.host.velocity[i] << ";"
			<< "\n";
	}
	return stream;
}
