#include <stdio.h>
#include <math.h>
#include <cstdlib>
#include <time.h>

// Function to write fields to files (space-separated matrix format)
void Write(int NX, int NY, int step, int N, double dx, double conc[], double phi1[], double phi2[], double phi3[], double phi4[])
{
    char filename[50];
    sprintf(filename, "Fields.%d.dat", step);
    FILE *OutFile = fopen(filename,"w");
    fprintf(OutFile, "x y c η_1 η_2 η_3 η_4 \n");

    for(int i=0;i<NX;i++)
    {
        for(int j=0;j<NY;j++)
        {
            int p = i+j*NX ;
            fprintf(OutFile, "%g %g ",(i-0.5)*dx,(j-0.5)*dx);
            fprintf(OutFile, "%g ",conc[p]);
            fprintf(OutFile, "%g ",phi1[p]);
            fprintf(OutFile, "%g ",phi2[p]);
            fprintf(OutFile, "%g ",phi3[p]);
            fprintf(OutFile, "%g ",phi4[p]);
            fprintf(OutFile, "\n");
        }
        fprintf(OutFile, "\n");
    }
    fclose(OutFile);
}

int main(int argc, char *argv[])
{
    clock_t begin=clock();

    // Input argument 1: 0 = No-flux BC, 1 = Periodic BC (default)
    const int Periodic = (argc>1)? atoi(argv[1]) : 1 ; // 0 for No-flux
    
    // ChiMaD Benchmark parameters
    const int N = 4 ;
    const double TIME = 100000. ;
    const double LENGTH = 200.0 ;
    const double C_ALPHA = 0.3 ;
    const double C_BETA = 0.7 ;
    const double RHO = sqrt(2.0) ;
    const double KAPPA_c = 3.0 ;
    const double KAPPA_p = 3.0 ;
    const double MOBILTY_c = 5.0 ;
    const double MOBILTY_p = 5.0 ;
    const double OMEGA = 1.0 ;
    const double ALPHA = 5.0 ;
    const double EPSILON_c = 0.05 ;
    const double C_ZERO = 0.5 ;
    const double EPSILON_p = 0.1 ;
    const double PSI = 1.5 ;
    
    // Numerical parameters
    const int Nx = 200 ; 
    const int Ny = 200 ; 
    const double dx = (Periodic)? LENGTH/Nx : LENGTH/(Nx-1) ;
    const double dt = 1.0e-3;
    const long int N_iter = TIME/dt ;
    int Iter_out_energy = 2 ; // Frequency log(1.5) base
    int Iter_out_fields = 1 ; // Frequency log(10) base
    
    // Fields arrays
    double c[Nx*Ny];
    double phi[N][Nx*Ny];
    double mu[Nx*Ny];
    double dpdt[N][Nx*Ny];
    const int di = 1 ;
    const int dj = Nx ;
    int p, pl, pr, pu, pd;
    int plu, pld, pru, prd; 
    
    // Field initialization
    double x,y;
    for (int i=0; i<Nx; i++)
    {
        for (int j=0; j<Ny; j++)
        {
            p = i*di + j*dj ;          
            x = (i-0.5)*dx;
            y = (j-0.5)*dx;

            c[p] =  C_ZERO + EPSILON_c*(cos(0.105*x)*cos(0.11*y) + (cos(0.13*x)*cos(0.087*y))*(cos(0.13*x)*cos(0.087*y)) + cos(0.025*x-0.15*y)*cos(0.07*x-0.02*y));
            
            for(int k=0;k<N;k++)
            {
                phi[k][p] = EPSILON_p*pow(cos((0.01*(k+1))*x-4.0)*cos((0.007+0.01*(k+1))*y)+cos((0.11+0.01*(k+1))*x)*cos((0.11+0.01*(k+1))*y)+PSI*pow(cos((0.046+0.001*(k+1))*x+(0.0405+0.001*(k+1))*y)*cos((0.031+0.001*(k+1))*x-(0.004+0.001*(k+1))*y),2),2) ;
            }
        }
    }
    
    // Boundary condition /x
    int right[Nx];
    int left[Nx];
    for (int i=0; i<Nx; i++)
    {
        // When in the bulk, right and left are just one di away
        right[i] = di ;
        left[i] = -di ;
    }
    if(Periodic)
    {
        right[Nx-1] -= Nx*di;
        left[0] += Nx*di; 
    }
    else // No-flux
    {
        right[Nx-1] -= di;
        left[0] += di; 
    }
    
    // Boundary condition /y
    int up[Ny];
    int down[Ny];
    for (int j=0; j<Ny; j++)
    {
        // When in the bulk, up and down are just one dj away
        up[j] = dj ;
        down[j] = -dj ;
    }
    if(Periodic)
    {
        up[Ny-1] -= Ny*dj ;
        down[0] += Ny*dj ;
    }
    else // No-flux
    {
        up[Ny-1] -= dj ;
        down[0] += dj ;
    }
    
    // Time evolution file opening  + header
    char EnergyFileName[24]="Energy.dat";
    FILE *TimeEvol = fopen(EnergyFileName,"w");
    fprintf(TimeEvol, "time E_{tot} \n");
    fclose(TimeEvol);

    // Initial field output
    Write(Nx, Ny, 0, N, dx, c, phi[0], phi[1], phi[2], phi[3]);

    ///////////////////////////////
    // Time Stepping
    ///////////////////////////////
	for (int iter=1; iter<=N_iter; iter++)
    {
        // Calculate mu and dp/dt
        for (int i=0; i<Nx; i++)
        {
            for (int j=0; j<Ny; j++)
            {
                p = i*di + j*dj ;            

                pl = p + left[i] ;            
                pr = p + right[i] ;            
                pu = p + up[j] ;            
                pd = p + down[j] ;

                double h = 0.0 ;
                for(int k=0;k<N;k++)
                {
                    h += phi[k][p]*phi[k][p]*phi[k][p]*(6.0*phi[k][p]*phi[k][p]-15.0*phi[k][p]+10.0) ;
                }
                
                mu[p] = 2.0*RHO*RHO*( (1-h)*(c[p]-C_ALPHA) + h*(c[p]-C_BETA) ) - KAPPA_c/dx/dx*(c[pr]+c[pl]+c[pu]+c[pd]-4.*c[p]) ;
                
                for(int k=0;k<N;k++)
                {
                    double sum = 0.0 ;
                    for(int kk=0;kk<N;kk++)
                    {
                        sum += phi[kk][p]*phi[kk][p] ;
                    }
                    sum -= phi[k][p]*phi[k][p] ;

                    dpdt[k][p] = 2.0*phi[k][p]*( 15.0*RHO*RHO*((c[p]-C_BETA)*(c[p]-C_BETA) - (c[p]-C_ALPHA)*(c[p]-C_ALPHA) )*phi[k][p]*(1.0-phi[k][p])*(1.0-phi[k][p]) + OMEGA*( (1.0-phi[k][p])*(1.0-2.0*phi[k][p]) + 2.0*ALPHA*sum ) ) ; // ∂f_c/∂η_i
                    dpdt[k][p] -= KAPPA_p/dx/dx*(phi[k][pr]+phi[k][pl]+phi[k][pu]+phi[k][pd]-4.*phi[k][p]) ; // κ*Lap(η_i)
                    dpdt[k][p] *= -MOBILTY_p ; // dη_i/dt
                }
            }
        }

        // Update fields
        for (int i=0; i<Nx ; i++)
        {
            for (int j=0; j<Ny; j++)
            {
                p = i*di + j*dj ;
                pl = p + left[i] ;
                pr = p + right[i] ;
                pu = p + up[j] ;
                pd = p + down[j] ;
                
                for(int k=0;k<N;k++)
                {
                    phi[k][p] = phi[k][p] + dt*dpdt[k][p] ;
                }

                c[p] = c[p] + MOBILTY_c*dt/dx/dx * (mu[pr] + mu[pl] + mu[pu] + mu[pd] - 4.*mu[p]);
                                
            }
        }
        
        // Output: time evolution of energy
        if(iter%Iter_out_energy==0)
        {
            // Energy calculation
            double Gibbs = 0.0 ;
            for (int i=0; i<Nx; i++)
            {
                for (int j=0; j<Ny; j++)
                {
                    p = i*di + j*dj ;            
                    pl = p + left[i] ;            
                    pr = p + right[i] ;            
                    pu = p + up[j] ;            
                    pd = p + down[j] ;

                    double GradC2dV = 0.25*( (c[pr]-c[pl])*(c[pr]-c[pl]) + (c[pu]-c[pd])*(c[pu]-c[pd]) ) ;
                    
                    double h = 0.0 ;
                    double g = 0.0 ;
                    double Gradp2dV = 0.0 ;
                    for(int k=0;k<N;k++)
                    {
                        h += phi[k][p]*phi[k][p]*phi[k][p]*(6.0*phi[k][p]*phi[k][p]-15.0*phi[k][p]+10.0) ;
                        
                        double sum = 0.0 ;
                        for(int kk=0;kk<N;kk++)
                        {
                            sum += phi[k][p]*phi[k][p]*phi[kk][p]*phi[kk][p] ;
                        }
                        sum -= phi[k][p]*phi[k][p]*phi[k][p]*phi[k][p] ;

                        g += phi[k][p]*phi[k][p]*(1.0-phi[k][p])*(1.0-phi[k][p]) + ALPHA*sum ;

                        Gradp2dV += 0.25*( (phi[k][pr]-phi[k][pl])*(phi[k][pr]-phi[k][pl]) + (phi[k][pu]-phi[k][pd])*(phi[k][pu]-phi[k][pd]) ) ;
                    }
                    
                    Gibbs += ( RHO*RHO*( (c[p]-C_ALPHA)*(c[p]-C_ALPHA)*(1.0-h) + (c[p]-C_BETA)*(c[p]-C_BETA)*h ) + OMEGA*g )*dx*dx  // f_chem * dx^2
                            + 0.5*KAPPA_c*GradC2dV //KAPPA_c/2*Grad(c)^2 * dx^2
                            + 0.5*KAPPA_p*Gradp2dV ; //KAPPA_η/2*Σ(Grad(η)^2) * dx^2
                    
                }
            }
            TimeEvol = fopen(EnergyFileName,"a");
            fprintf(TimeEvol, "%g %g \n", iter*dt, Gibbs);
            fclose(TimeEvol);
            
            // Field output
            if(iter*dt >= 1.0)
            {
                Write(Nx, Ny, int(iter*dt), N, dx, c, phi[0], phi[1], phi[2], phi[3]);
            }

            Iter_out_energy *= 1.5; // Output frequency log(1.5) basis
        }
        
        if(((iter-1)*dt-Iter_out_fields)*(iter*dt-Iter_out_fields) <= 0)
        {
            Write(Nx, Ny, int(Iter_out_fields), N, dx, c, phi[0], phi[1], phi[2], phi[3]);
            Iter_out_fields *= 10; // Output frequency log(10) basis
        }

	}

    // Final field output
    Write(Nx, Ny, N_iter, N, dx, c, phi[0], phi[1], phi[2], phi[3]);

    // Timing
    clock_t end=clock();
    double CompTime=(end-begin)/(1.0*CLOCKS_PER_SEC);
    printf("Done in %d seconds\n",(int)(CompTime));
    
    return EXIT_SUCCESS;
}
