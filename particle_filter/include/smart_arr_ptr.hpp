#pragma once 

#include <memory>

template <typename T>
class smart_arr_ptr{
public:
    smart_arr_ptr(size_t N){
        val_ = std::make_unique<T[]>(N);
        N_ = N;
    }    
    
    inline T & operator[](size_t idx){
        return val_[idx % N_];
    }
    
    inline const T & operator[](size_t idx) const{
        return val_[idx % N_];
    }
    
    inline const size_t & size() const {
        return N_;
    }
    
private:
    smart_arr_ptr() = delete;
    smart_arr_ptr(const smart_arr_ptr<T> & orig) = delete;
    void operator = (const smart_arr_ptr<T> & rhs) = delete;
    
    std::unique_ptr<T[]> val_{nullptr};
    size_t N_{0};
};