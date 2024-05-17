#include "particle_filter.hpp"

static std::unique_ptr<IfaceFilter<double>> filter = nullptr;

extern "C" void create_particle_filter(size_t N){
    filter = std::make_unique<ParticleFilter<double>>(N);
}

extern "C" void initialize(double * __restrict__ S0){
    filter->initialize(*S0);
}

extern "C" void predict(double * __restrict__ x, double * __restrict__ sd){
    filter->predict(*x, *sd);
}

extern "C" void correct(double * __restrict__ S){
    filter->correct(*S);
}