#pragma once 

#include "system.hpp"
#include "iface_filter.hpp"

template <typename T>
class ParticleFilter : public IfaceFilter<T> {
public:
    ParticleFilter(size_t N):
        entt_(N) {}

    void initialize(T S0){
        entt_.prev_S() = S0;
        initialize_entity(entt_);
    }

    void predict(T & x, T & sd_x){
        predict_belief(entt_);
        estimate_belief(entt_);
        x = entt_.mean_x();
        sd_x = std::sqrt(entt_.mean_x_sq() - entt_.mean_x() * entt_.mean_x());
    }
    
    void correct(T & S){
        entt_.S() = S;
        correct_belief(entt_);
        update_belief(entt_);
    }

private:
    ParticleFilter() = delete;
    ParticleFilter(const ParticleFilter<T> & orig) = delete;
    void operator = (const ParticleFilter<T> & rhs) = delete;
    
    Entity<T> entt_;
};