#pragma once 

template <typename T>
struct IfaceFilter{
    virtual void initialize(T S0) = 0;
    virtual void predict(T & x, T & sd_x) = 0;
    virtual void correct(T & S) = 0;    
};
