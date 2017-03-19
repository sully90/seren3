/*
Cloud in cell kernel originally written by Alexander Eggemeier and used with his permission,
@author Alexander Eggemeier
@author David Sullivan

Version history
========================================================================================
Original kernel - Alex
OpenMP support - David
Python wrappers - David
*/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

using namespace std;

int VecId(int i, int j, int k, int N)
{
    return (k * N * N) + (j * N) + i;
}

void run(double * x, double * y, double * z, int NumPart, double L, int N, double * rho)
{
 cout << "Assign particles to mesh with size N=" << N << " ...";
 double spacing = L/N;  // fundamental mesh size = sidelength of box / number of grid cells per side
 //std::cout << spacing;
 int p;

 //double* rho = new double[N*N*N]; // works
 //rho = new double[N*N*N];

 //cout << NumPart << endl;

 #pragma omp parallel for
 for (p=0; p<NumPart; p++)
   {
     //cout << x[p] << y[p] << z[p] << endl;
     // P[p].Pos[i] = Position of particle ‘p’ in dimension ‘i'
     if ( (x[p] < 0) || (x[p] > L) || (y[p] < 0) || (y[p] > L) || (z[p] < 0) || (z[p] > L) ) {
        continue;
     }
     int i = floor(x[p]/spacing);
     int j = floor(y[p]/spacing);
     int k = floor(z[p]/spacing);

     double dx = x[p]/spacing-i;
     double dy = y[p]/spacing-j;
     double dz = z[p]/spacing-k;
     double tx = 1-dx;
     double ty = 1-dy;
     double tz = 1-dz;

     int ipp = (i+1)%N;
     int jpp = (j+1)%N;
     int kpp = (k+1)%N;

     rho[VecId(i,j,k,N)] += tx*ty*tz;
     rho[VecId(ipp,j,k,N)] += dx*ty*tz;
     rho[VecId(i,jpp,k,N)] += tx*dy*tz;
     rho[VecId(i,j,kpp,N)] += tx*ty*dz;
     rho[VecId(ipp,jpp,k,N)] += dx*dy*tz;
     rho[VecId(ipp,j,kpp,N)] += dx*ty*dz;
     rho[VecId(i,jpp,kpp,N)] += tx*dy*dz;
     rho[VecId(ipp,jpp,kpp,N)] += dx*dy*dz;
     //cout << rho[VecId(i,j,k,N)] << endl;
   }
   cout << "done!" << endl;
   return ;
}

void c_filt_particles(double bound_min, double bound_max, int NumPart, double * x, double * y, double * z, double * idx)
{
    cout << "Filtering particles" << endl;
    cout << "bound_min = " << bound_min << " bound_max = " << bound_max << endl;

    #pragma omp parallel for
    for (int p=0; p<NumPart; p++)
    {
        if ( (x[p] < bound_min) || (x[p] > bound_max) || (y[p] < bound_min) || (y[p] > bound_max) || (z[p] < bound_min) || (z[p] > bound_max) ) {
            idx[p] = 1;
        }
        else
        {
            idx[p] = 0;
        }
    }
    cout << "Done" << endl;
    return ;
}

void c_tag_refined_particles(int NumPart, int nx, double * cx, double * cy, double * cz, double * zx, double * zy, double * zz, double * refmap, double * idx)
{
    /**
    Tags particles which enter the refinement region with a 1
    refmap - flattened array of ref. region
    **/

    #pragma omp parallel for
    for (int i=0; i<NumPart; i++)
    {
        //cout << i << endl;
        int ci = floor(cx[i]);
        int cj = floor(cy[i]);
        int ck = floor(cz[i]);

        int pi = floor(zx[i]);
        int pj = floor(zy[i]);
        int pk = floor(zz[i]);

        if ( (ci < 0) || (ci > nx) || (cj < 0) || (cj > nx) || (ck < 0) || (ck > nx) ) {
            continue;
        }
        else if ( (pi < 0) || (pi > nx) || (pj < 0) || (pj > nx) || (pk < 0) || (pk > nx) )
        {
            continue;
        }

        //cout << VecId(ci, cj, ck, nx) << "\t" << VecId(pi, pj, pk, nx) << endl;
        if ( refmap[VecId(ci, cj, ck, nx)] == 0 && refmap[VecId(pi, pj, pk, nx)] == 1 ) {
            // Particle has moved into refinement region
            idx[i] = 1;
        }
        else
        {
            idx[i] = 0;
        }
        //cout << refmap[VecId(pi, pj, pk, nx)] << endl;
    }

    return ;
}
