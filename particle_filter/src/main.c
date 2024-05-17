#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <dlfcn.h>

void * handle;
void (*create_particle_filter)(size_t N);
void (*initialize)(double * restrict S0);
void (*predict)(double * restrict x, double * restrict sd);
void (*correct)(double * restrict S);

void load_library(const char *);

int main(){
    load_library("filter_d.dll");
    
    double x;
    double sd_x;
    double start;
    double end;
    
    size_t N;
    
    
    double S = 100;
    
    scanf("%lu", &N);
    
    create_particle_filter(N);
    initialize(&S);
    
    for(size_t idx = 0; idx < 500; ++idx){
        S *= 1.001;
        start = omp_get_wtime();
        predict(&x, &sd_x);
        correct(&S);
        end = omp_get_wtime();
        
        printf("%f\t%f\t%f\t%f\n", S, x, sd_x, end - start);
    }
    
    dlclose(handle);
    
    return 0;
}

void load_library(const char * libfile){
    char * error;
    
    handle = dlopen (libfile, RTLD_LAZY);
    
    if (!handle) {
        fputs (dlerror(), stderr);
        exit(1);
    }
    
    create_particle_filter = dlsym(handle, "create_particle_filter");
    if ((error = dlerror()) != NULL)  {
        fputs(error, stderr);
        exit(1);
    }
    
    initialize = dlsym(handle, "initialize");
    if ((error = dlerror()) != NULL)  {
        fputs(error, stderr);
        exit(1);
    }
    
    predict = dlsym(handle, "predict");
    if ((error = dlerror()) != NULL)  {
        fputs(error, stderr);
        exit(1);
    }
    
    correct = dlsym(handle, "correct");
    if ((error = dlerror()) != NULL)  {
        fputs(error, stderr);
        exit(1);
    }    
}