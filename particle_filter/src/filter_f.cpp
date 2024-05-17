#include "particle_filter.hpp"

static std::unique_ptr<IfaceFilter<float>> filter = nullptr;

extern "C" void create_particle_filter(size_t N){
    filter = std::make_unique<ParticleFilter<float>>(N);
}

extern "C" void initialize(float * __restrict__ S0){
    filter->initialize(*S0);
}

extern "C" void predict(float * __restrict__ x, float * __restrict__ sd){
    filter->predict(*x, *sd);
}

extern "C" void correct(float * __restrict__ S){
    filter->correct(*S);
}