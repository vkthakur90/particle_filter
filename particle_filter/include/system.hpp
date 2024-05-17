#pragma once 

#include <cstdlib>
#include <cmath>

#include "entity.hpp"

template <typename T>
static void initialize_entity(Entity<T> & entt){
    size_t N = entt.size();

    #pragma omp parallel for simd
    for(size_t idx = 0; idx < N; ++idx){
        entt.x(idx) = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    }
    
    #pragma omp parallel for simd
    for(size_t idx = 0; idx < N; ++idx){
        entt.x(idx) = 2 * entt.x(idx) - 1;
    }
    
    #pragma omp parallel for simd
    for(size_t idx = 0; idx < N; ++idx){
        entt.x(idx) *= 5;
    }

    #pragma omp parallel for simd
    for(size_t idx = 0; idx < N; ++idx){
        entt.prev_w(idx) = static_cast<T>(1);
    }    
}

template <typename T>
static void predict_belief(Entity<T> & entt){
    size_t N = entt.size();
    
    #pragma omp parallel for simd
    for(size_t k = 0; k < N; ++k){
        entt.pred_w(k) = entt.prev_w(k);
    }
}

template <typename T>
static void predict_belief_v2(Entity<T> & entt){
    size_t N = entt.size();
    
    for(size_t m = 0; m < N; ++m){
        entt.pred_w(m) = static_cast<T>(0);
    }
    
    for(size_t m = 0; m < N; ++m){
        #pragma omp parallel for simd         
        for(size_t k = 0; k < N; ++k){
            entt.reg(k) = entt.x(m);
        }
        
        #pragma omp parallel for simd 
        for(size_t k = 0; k < N; ++k){
            entt.reg(k) -= entt.x(k);
        }
        
        #pragma omp parallel for simd 
        for(size_t k = 0; k < N; ++k){
            entt.reg(k) = std::pow(entt.reg(k), 2);
        }
        
        #pragma omp parallel for simd 
        for(size_t k = 0; k < N; ++k){
            entt.reg(k) = std::exp(-entt.reg(k));
        }
        
        #pragma omp parallel for simd 
        for(size_t k = 0; k < N; ++k){
            entt.reg(k) *= entt.prev_w(m);
        }
        
        #pragma omp parallel for simd 
        for(size_t k = 0; k < N; ++k){
            entt.pred_w(k) += entt.reg(k);
        }
    }
    
    T & sum_pred_w = entt.sum_pred_w();
    
    sum_pred_w = static_cast<T>(0);
      
    #pragma omp parallel for simd reduction(+:sum_pred_w) 
    for(size_t m = 0; m < N; ++m){
        sum_pred_w += entt.pred_w(m);
    }
    
    #pragma omp parallel for simd 
    for(size_t m = 0; m < N; ++m){
        entt.pred_w(m) /= sum_pred_w + static_cast<T>(1.0e-6);
    }
    
    sum_pred_w = static_cast<T>(0);  
    
    #pragma omp parallel for simd reduction(+:sum_pred_w) 
    for(size_t m = 0; m < N; ++m){
        sum_pred_w += entt.pred_w(m);
    }
}

template <typename T>
static void estimate_belief(Entity<T> & entt){
    size_t N = entt.size();
    
    #pragma omp parallel for simd
    for(size_t m = 0; m < N; ++m){
        entt.x_pred_w(m) = entt.pred_w(m);
    }
    
    #pragma omp parallel for simd
    for(size_t m = 0; m < N; ++m){
        entt.x_pred_w(m) *= entt.x(m);
    }
    
    #pragma omp parallel for simd
    for(size_t m = 0; m < N; ++m){
        entt.x_sq_pred_w(m) = entt.x_pred_w(m);
    }
    
    #pragma omp parallel for simd
    for(size_t m = 0; m < N; ++m){
        entt.x_sq_pred_w(m) *= entt.x(m);
    }

    T & sum_pred_w = entt.sum_pred_w();
    
    sum_pred_w = static_cast<T>(0);
    
    
    #pragma omp parallel for simd reduction(+:sum_pred_w) 
    for(size_t m = 0; m < N; ++m){
        sum_pred_w += entt.pred_w(m);
    }
 
    T & mean_x = entt.mean_x();
    
    mean_x = static_cast<T>(0); 
    
    #pragma omp parallel for simd reduction(+:mean_x) 
    for(size_t m = 0; m < N; ++m){
        mean_x += entt.x_pred_w(m);
    }

    mean_x /= entt.sum_pred_w() + static_cast<T>(1.0e-6);

    T & mean_x_sq = entt.mean_x_sq();
    mean_x_sq = static_cast<T>(0); 
    
    #pragma omp parallel for simd reduction(+:mean_x_sq) 
    for(size_t m = 0; m < N; ++m){
        mean_x_sq += entt.x_sq_pred_w(m);
    }
    
    mean_x_sq /= entt.sum_pred_w() + static_cast<T>(1.0e-6);
}

template <typename T>
static void correct_belief(Entity<T> & entt){
    size_t N = entt.size();
    
    entt.y() = std::log(entt.S());
    entt.y() -= std::log(entt.prev_S());
    entt.y() *= static_cast<T>(100);
    
    #pragma omp parallel for simd
    for(size_t m = 0; m < N; ++m){
        entt.reg(m) = entt.y();
    }
    
    #pragma omp parallel for simd
    for(size_t m = 0; m < N; ++m){
        entt.reg(m) -= entt.x(m); 
    }
    
    #pragma omp parallel for simd
    for(size_t m = 0; m < N; ++m){
        entt.reg(m) = std::pow(entt.reg(m), 2.0);
    }
    
    #pragma omp parallel for simd
    for(size_t m = 0; m < N; ++m){
        entt.reg(m) = std::exp(-entt.reg(m));
    }
    
    #pragma omp parallel for simd
    for(size_t m = 0; m < N; ++m){
        entt.w(m) = entt.pred_w(m);
    }
    
    #pragma omp parallel for simd
    for(size_t m = 0; m < N; ++m){
        entt.w(m) *= entt.reg(m);
    }
    
    T & sum_w = entt.sum_w();
    
    sum_w = static_cast<T>(0);
      
    #pragma omp parallel for simd reduction(+:sum_w) 
    for(size_t m = 0; m < N; ++m){
        sum_w += entt.w(m);
    }
    
    #pragma omp parallel for simd 
    for(size_t m = 0; m < N; ++m){
        entt.w(m) /= sum_w + static_cast<T>(1.0e-6);
    }
    
    sum_w = static_cast<T>(0);  
    
    #pragma omp parallel for simd reduction(+:sum_w) 
    for(size_t m = 0; m < N; ++m){
        sum_w += entt.w(m);
    }
}

template <typename T>
static void update_belief(Entity<T> & entt){
    size_t N = entt.size();
    
    #pragma omp parallel for simd 
    for(size_t m = 0; m < N; ++m){
        entt.prev_w(m) = entt.w(m);
    } 

    entt.prev_S() = entt.S();    
}
