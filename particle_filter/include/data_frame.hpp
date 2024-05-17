#pragma once

#include "smart_arr_ptr.hpp"

template <typename T>
class data_frame{
public:
    data_frame(size_t nrows, size_t ncols):
        val_(nrows * ncols),
        nrows_(nrows),
        ncols_(ncols) {}

    inline T & operator()(size_t ridx, size_t cidx){
        return val_[(ridx % nrows_) + nrows_ * (cidx % ncols_)];
    }

    inline const T & operator()(size_t ridx, size_t cidx) const {
        return val_[(ridx % nrows_) + nrows_ * (cidx % ncols_)];
    }  

    inline const size_t & nrows() const {
        return nrows_;
    }    
    
    inline const size_t & ncols() const {
        return ncols_;
    }
    
private:
    data_frame() = delete;
    data_frame(const data_frame<T> & orig) = delete;
    void operator = (const data_frame<T> & rhs) = delete;

    smart_arr_ptr<T> val_;
    size_t nrows_;
    size_t ncols_;    
};