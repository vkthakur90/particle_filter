#pragma once 

#include "data_frame.hpp"

template <typename T>
class Entity {
public:
    Entity(size_t N):
        data_df_(N + 1, 7),
        N_(N) {}
    
    inline T & x(size_t idx) {
        return data_df_(idx % N_, 0);
    }
    
    inline T & prev_w(size_t idx) {
        return data_df_(idx % N_, 1);
    }
    
    inline T & pred_w(size_t idx) {
        return data_df_(idx % N_, 2);
    }
    
    inline T & w(size_t idx) {
        return data_df_(idx % N_, 3);
    }  

    inline T & x_pred_w(size_t idx) {
        return data_df_(idx % N_, 4);
    }

    inline T & x_sq_pred_w(size_t idx) {
        return data_df_(idx % N_, 5);
    }    
    
    inline T & reg(size_t idx) {
        return data_df_(idx % N_, 6);
    }
    
    inline T & y() {
        return data_df_(N_, 0);
    }
    
    inline T & prev_S() {
        return data_df_(N_, 1);
    }
    
    inline T & sum_pred_w() {
        return data_df_(N_, 2);
    }
    
    inline T & sum_w() {
        return data_df_(N_, 3);
    }
    
    inline T & mean_x() {
        return data_df_(N_, 4);
    }
    
    inline T & mean_x_sq() {
        return data_df_(N_, 5);
    }
    
    inline T & S() {
        return data_df_(N_, 6);
    }

    inline const size_t & size() const {
        return N_;
    }
    
private:
    Entity() = delete;
    Entity(const Entity<T> & orig) = delete;
    void operator = (const Entity<T> & rhs) = delete;

    data_frame<T> data_df_;
    size_t N_;    
};