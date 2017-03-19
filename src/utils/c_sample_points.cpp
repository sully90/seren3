#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

using namespace std;

void c_sample_sphere_surface(double * x, double * y, double * z, double r, int npoints)
{
    // Sample the surface of a sphere with npoints
    const double PI = std::atan(1.0)*4;
    double theta, phi;

    #pragma omp parallel for
    for (int i=0; i<npoints; i++)
    {
        theta = ((double)i/(double)npoints) * 2*PI;
        phi = ((double)i/(double)npoints) * PI;
        //cout << theta << " " << phi << endl;
        x[i] = r * sin(phi) * cos(theta);
        y[i] = r * sin(phi) * sin(theta);
        z[i] = r * cos(phi);
    }
}