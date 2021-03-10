# Copyright Asitha Senanayake 2020


import numpy as np
import matplotlib.pyplot as plt

###################################
#### Pile Geometry and Loading ####
###################################


def py_analysis_1_SI(soil_profile, L=10.0, D=1.0, t = 0.05, E=200e9, F = 0.0,
                     V_0=1000.0, M_0=0.0, V_n=0.0, M_n=0.0, n=50, iterations=10,
                     py_model='Matlock', print_output='Yes',
                     convergence_tracker='No', loc=2, **kwargs):
    '''Models a laterally loaded pile using the p-y method. The solution for
    lateral displacements is obtained by solving the 4th order ODE, EI*d4y/dz4
    -F*d2y/dz2 + ky = 0 using the finite difference method.

    Takes natural boundary conditions at the pile head and tip. For
    displacement controlled analysis check 'py_analysis_2'.

    Assumes that EI remains constant with respect to curvature i.e. pile
    material remains in the elastic region.

    Input:
    -----
    Su_profile  - A 2D array of depths (m) and corresponding undrained shear strength(Pa)
                  Eg: array([[z1,Su1],[z2,Su2],[z3,Su3]...])
                  Use small values for Su (eg: 0.001) instead of zeros to avoid divisions by zero but always start z at 0.0
                  Example of a valid data point at the mudline is [0.0, 0.001]

    L           - Length of pile         (m)
    D           - Outer diameter of pile (m)
    t           - Wall thickness of pile (m)
    E           - Elastic modulus of pile material (Pa)
    F           - Axial force at pile head (N), vertically downwards is postive.
    V_0, V_n    - Force at pile head/tip  (N),  shear causing clockwise rotation of pile is positive.
    M_0, M_n    - Moment at pile head/tip (N-m), moments causing tension on left side of pile is positive.
    n           - Number of elements (50 by default)
    iterations  - Number of iterations to repeat calculation in order obtain convergence of 'y'
                  (A better approach is to iterate until a predefined tolerance is achieved but this requires additional
                  coding so, I will implement this later.)
    py_model    - Select which p-y model to use, 'Matlock' or 'Jeanjean_2009', 'MM-1', or 'Jeajean_etal_2017'.

    Optional:
    convergence_tracker - Track how k_secant converges to actual p-y curve at a selected node
    loc                 - Node number at which k_secant to be tracked (2 to n+1)

    Output:
    ------
    y           - Lateral displacement at each node, length = n + 5, (n+1) real nodes and 4 imaginary nodes
    z           - Vector of node locations along pile
    '''

    # Extract optional keyword arguments
    epsilon_50, A, gapping, N_p_max = 0.02, 550, 'No', 12.0  # Default parameters if no **kwargs are defined
    custom_py, a, strain_f          = 'No', 0.0, 0.0
    ls, alpha                       = 'x', 0.0

    for arg in kwargs:
        if arg == 'epsilon_50':
            epsilon_50 = kwargs[arg]
        if arg == 'Gmax_Su_ratio':
            A = kwargs[arg]
        if arg == 'gapping':
            gapping = kwargs[arg]
        if arg == 'N_p_max':
            N_p_max = kwargs[arg]
        if arg == 'alpha':
            alpha = kwargs[arg]
        if arg == 'custom_py':
            custom_py = kwargs[arg]
        if arg == 'a':
            a = kwargs[arg]
        if arg == 'strain_f':
            strain_f = kwargs[arg]
        if arg == 'ls':
            ls = kwargs[arg]

    # Convert L and D to floating point numbers to avoid rounding errors
    L = float(L)
    D = float(D)

    # Pile geometry
    I  = np.pi*(D**4 - (D-2*t)**4)/64.0
    EI = E*I
    h  = L/n  # Element size
    N  = (n+1)+4      # (n+1) Real + 4 Imaginary nodes

    # Array for displacements at nodes, including imaginary nodes.
    y = np.ones(N)*(0.01*D)   # An initial value of 0.01D was arbitrarily chosen

    # Initialize and assemble array/list of p-y curves at each real node
    z = np.zeros(N)
    py_funs  = []
    k_secant = np.zeros(N)

    for i in [0, 1]:        # Top two imaginary nodes
        z[i] = (i-2)*h
        py_funs.append(0)
        k_secant[i] = 0.0

    # Extract soil profile data
    z_0, f_Su, f_sigma_v_eff = design_soil_profile_SI(soil_profile)

    for i in range(2, n+3):  # Real nodes
        z[i] = (i-2)*h

        Su, Su0, sigma_v_eff = f_Su(z[i]), f_Su(z_0+0.01), f_sigma_v_eff(z[i])

        if py_model == 'Matlock':
            py_funs.append(matlock_py_curves_SI(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=epsilon_50, print_curves='No'))
        elif py_model == 'Jeanjean_2009':
            py_funs.append(jeanjean_2009_py_curves_SI(z[i], D, Su, sigma_v_eff, z_0=z_0, Su_0=f_Su(z_0), A=A))
        elif py_model == 'MM-1':
            py_funs.append(MM_1_py_curves_SI(z[i], D, Su, Su0, sigma_v_eff, z_0=z_0, epsilon_50=epsilon_50, print_curves='No',
                                             gapping=gapping, alpha=alpha))
        elif py_model == 'Jeanjean_etal_2017':
            py_funs.append(jeanjean_2017_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, print_curves='No',
                                                  Su_0=f_Su(z_0+0.01), gapping=gapping, alpha=alpha,
                                                   custom_py=custom_py, a=a, strain_f=strain_f))
        else:
            print("P-y model not properly defined. Please select one of the following:")
            print("'Matlock', 'Jeanjean', 'Modified Matlock', 'Kodikara', 'MM-1', 'MM-2', 'Jeanjean et al 2017'")

        k_secant[i]     = py_funs[i](y[i])/y[i]

    for i in [n+3, n+4]:   # Bottom two imaginary nodes
        z[i] = (i-2)*h
        py_funs.append(0)
        k_secant[i] = 0.0

    # Track k_secant and current displacements
    if convergence_tracker == 'Yes':
        y1 = np.linspace(-2.*D,2.*D,500)
        plt.plot(y1, py_funs[loc](y1))
        plt.xlabel('y (m)'), plt.ylabel('p (N/m)'), plt.grid(True)

    for j in range(iterations):
        # if j == 0: print 'FD Solver started!'

        y = fd_solver_1(n,N,h,EI,F,V_0,V_n,M_0,M_n,k_secant)

        if convergence_tracker == 'Yes':
            plt.plot(y[loc], k_secant[loc]*y[loc], ls)

        for i in range(2, n+3):
            k_secant[i] = py_funs[i](y[i])/y[i]

    if print_output == 'Yes':
        print(f'y_0 = {y[2]:.3f}-m')

    return y[2:-2], z[2:-2]



def py_analysis_2_SI(soil_profile, L=10.0, D=1.0, t = 0.05, E=200e9, F = 0.0, y_0=0.0, M_0=0.0, V_n=0.0, M_n=0.0, n=50,
                  iterations=10, py_model='Matlock', print_output='Yes', convergence_tracker='No', loc=2, **kwargs):
    '''Models a laterally loaded pile using the p-y method. The solution for lateral displacements
    is obtained by solving the 4th order ODE, EI*d4y/dz4 -F*d2y/dz2 + ky = 0 using the finite difference method.

    Assumes that EI remains constant with respect to curvature i.e. pile material remains in the elastic region.

    Input:
    -----
    soil_profile - A 2D array of depths (m), corresponding undrained shear strength(Pa),
                   and corresponding vertical effective stress (Pa)
                   Eg: array([[z1,Su1,sigma_v_eff1],[z2,Su2,sigma_v_eff2],[z3,Su3,sigma_v_eff3]...])
                   Use small values for Su and sigma_v_eff (eg: 0.001) instead of zeros to avoid divisions
                   by zero but always start z at 0.0.
                   Example of a valid data point at the mudline is [0.0, 0.001, 0.001]

    L           - Length of pile         (m)
    D           - Outer diameter of pile (m)
    t           - Wall thickness of pile (m)
    E           - Elastic modulus of pile material (Pa)
    y_0         - Displacement at the pile head (m)
    F           - Axial force at pile head (m), vertically downwards is postive.
    M_0         - Moment at pile head (Nm), moments causing tension on left side of pile is positive.
    V_n         - Force at pile head/tip  (N),  shear causing clockwise rotation of pile is positive.
    M_n         - Moment at pile tip (N-m), moments causing tension on left side of pile is positive.
    n           - Number of elements (50 by default)
    iterations  - Number of iterations to repeat calculation in order obtain convergence of 'y'
                  (A better approach is to iterate until a predefined tolerance is achieved but this requires additional
                  coding so, I will implement this later.)
    py_model    - Select which p-y model to use, 'Matlock', 'Jeanjean_2009', 'MM-1', or 'Jeanjean_etal_2017'.
    print_output - Print results on screen, Choose 'Yes' or 'No' (default = 'No')

    Optional:
    convergence_tracker - Track how k_secant converges to actual p-y curve at a selected node
    loc                 - Node number at which k_secant to be tracked (2 to n+1)

    Optional keywords: **kwargs
    epsilon_50    - Define 'epsilon_50' if 'Matlock' p-y model is selected. (epsilon_50=0.02 by default)
    Gmax_Su_ratio - Define G_max/Su if Jeanjean or Kodikara p-y models are chosen. Default values are:
                     Jeanjean (2009) -> Gmax_Su_ratio = 550
                     Kodikara (2009) -> Gmax_Su_ratio = 250
    alpha         - Define coefficient of pile-soil interface adhesion if 'MM-1' is selected.

    Output:
    ------
    y           - Lateral displacement at each node, length = n + 5, (n+1) real nodes and 4 imaginary nodes
    l           - Vector of node locations along pile
    V_0         - Load at pile head (lb)
    '''

    from scipy import linalg

    # Extract optional keyword arguments
    epsilon_50, A, gapping, N_p_max = 0.02, 550, 'No', 12.0 # Default parameters if no **kwargs are defined
    custom_py, a, strain_f          = 'No', 0.0, 0.0
    ls, alpha                       = 'x', 0.0

    for arg in kwargs:
        if arg == 'epsilon_50':
            epsilon_50 = kwargs[arg]
        if arg == 'Gmax_Su_ratio':
            A = kwargs[arg]
        if arg == 'gapping':
            gapping=kwargs[arg]
        if arg == 'N_p_max':
            N_p_max=kwargs[arg]
        if arg == 'alpha':
            alpha=kwargs[arg]
        if arg == 'custom_py':
            custom_py=kwargs[arg]
        if arg == 'a':
            a = kwargs[arg]
        if arg == 'strain_f':
            strain_f=kwargs[arg]
        if arg == 'ls':
            ls=kwargs[arg]

    # Convert L and D to floating point numbers to avoid rounding errors
    L = float(L)
    D = float(D)

    # Pile geometry
    I  = np.pi*(D**4 - (D-2*t)**4)/64.0
    EI = E*I
    h  = L/n  # Element size
    N  = (n+1)+4      # (n+1) Real + 4 Imaginary nodes

    # Array for displacements at nodes, including imaginary nodes.
    y = np.ones(N)*(0.01*D)   # An initial value of 0.01D was arbitrarily chosen

    # Initialize and assemble array/list of p-y curves at each real node
    z = np.zeros(N)
    py_funs  = []
    k_secant = np.zeros(N)

    for i in [0,1]:        # Top two imaginary nodes
        z[i] = (i-2)*h
        py_funs.append(0)
        k_secant[i] = 0.0

    # Extract soil profile data
    z_0, f_Su, f_sigma_v_eff = design_soil_profile_SI(soil_profile)

    for i in range(2,n+3): # Real nodes
        z[i] = (i-2)*h

        Su, Su0, sigma_v_eff = f_Su(z[i]), f_Su(z_0+0.01), f_sigma_v_eff(z[i])

        if py_model == 'Matlock':
            py_funs.append(matlock_py_curves_SI(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=epsilon_50, print_curves='No'))
        elif py_model == 'Jeanjean_2009':
            py_funs.append(jeanjean_2009_py_curves(z[i], D,Su, sigma_v_eff, z_0=z_0, Su_0=f_Su(z_0), A=A))
        elif py_model == 'MM-1':
            py_funs.append(MM_1_py_curves_SI(z[i], D, Su, Su0, sigma_v_eff, z_0=z_0, epsilon_50=epsilon_50, print_curves='No',
                                                  gapping=gapping, alpha=alpha))
        elif py_model == 'Jeanjean_etal_2017':
            py_funs.append(jeanjean_2017_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, print_curves='No',
                                                  Su_0=f_Su(z_0+0.01), gapping=gapping, alpha=alpha,
                                                   custom_py=custom_py, a=a, strain_f=strain_f))
        else:
            print("P-y model not properly defined. Please select one of the following:")
            print("'Matlock', 'Jeanjean', 'Modified Matlock', 'Kodikara', 'MM-1', 'MM-2', 'Jeanjean et al 2017'")

        k_secant[i]     = py_funs[i](y[i])/y[i]

    for i in [n+3, n+4]:   # Bottom two imaginary nodes
        z[i] = (i-2)*h
        py_funs.append(0)
        k_secant[i] = 0.0

    # Track k_secant and current displacements
    if convergence_tracker == 'Yes':
        y1 = np.linspace(-2.*D,2.*D,500)
        plt.plot(y1, py_funs[loc](y1))
        plt.xlabel('y (m)'), plt.ylabel('p (N/m)'), plt.grid(True)

    for j in range(iterations):
        # if j == 0: print 'FD Solver started!'

        y,V_0 = fd_solver_2(n,N,h,EI,F,y_0,V_n,M_0,M_n,k_secant)

        if convergence_tracker == 'Yes':
            plt.plot(y[loc], k_secant[loc]*y[loc], ls)

        for i in range(2, n+3):
            k_secant[i] = py_funs[i](y[i])/y[i]

    if print_output == 'Yes':
        print('V_0 = %.2f-N' %V_0)

    return y[2:-2], z[2:-2], V_0



#################
#### Solvers ####
#################

def fd_solver_1(n,N,h,EI,F,V_0,V_n,M_0,M_n,k_secant):
    '''Solves the finite difference equations from 'py_analysis_1'. This function should be run iteratively for
    non-linear p-y curves by updating 'k_secant' using 'y'. A single iteration is sufficient if the p-y curves
    are linear.

    Input:
    -----
    n - Number of elements
    N - Total number of nodes
    h - Element size
    EI - Flexural rigidity of pile
    F  - Axial force at pile head
    V_0, V_n - Shear at pile head/tip
    M_0, M_n - Moment at pile head/tip
    k_secant - Secant stiffness from p-y curves

    Output:
    ------
    y_updated - Lateral displacement at each node
    '''

    from scipy import linalg

    # Initialize and assemble matrix
    X = np.zeros((N,N))

    # (n+1) finite difference equations for (n+1) real nodes
    for i in range(0,n+1):
        X[i,i]   =  1.0
        X[i,i+1] = -4.0 + F*h**2/EI
        X[i,i+2] =  6.0 - 2*F*h**2/EI + k_secant[i+2]*h**4/EI
        X[i,i+3] = -4.0 + F*h**2/EI
        X[i,i+4] =  1.0

    # Curvature at pile head
    X[n+1,1]   =  1.0
    X[n+1,2]   = -2.0
    X[n+1,3]   =  1.0

    # Shear at pile head
    X[n+2,0]   = -1.0
    X[n+2,1]   =  2.0 - F*h**2/EI
    X[n+2,2]   =  0.0
    X[n+2,3]   = -2.0 + F*h**2/EI
    X[n+2,4]   =  1.0

    # Curvature at pile tip
    X[n+3,-2]   =  1.0
    X[n+3,-3]   = -2.0
    X[n+3,-4]   =  1.0

    # Shear at pile tip
    X[n+4,-1]   =   1.0
    X[n+4,-2]   =  -2.0 + F*h**2/EI
    X[n+4,-3]   =   0.0
    X[n+4,-4]   =   2.0 - F*h**2/EI
    X[n+4,-5]   =  -1.0

    # X*y = q

    # Initialize vector q
    q = np.zeros(N)

    # Populate q with boundary conditions
    q[-1] = 2*V_n*h**3     # Shear at pile tip
    q[-2] = M_n*h**2       # Moment at pile tip
    q[-3] = 2*V_0*h**3     # Shear at pile head
    q[-4] = M_0*h**2       # Moment at pile head

    y = linalg.solve(EI*X,q)

    return y


def fd_solver_2(n,N,h,EI,F,y_0,V_n,M_0,M_n,k_secant):
    '''Solves the finite difference equations from 'py_analysis_2'. This function should be run iteratively for
    non-linear p-y curves by updating 'k_secant' using 'y'. A single iteration is sufficient if the p-y curves
    are linear.

    Specified boundary conditions: Displacement and moment at pile head. Shear and moment at pile tip.
                                   Axial force at pile head.

    Input:
    -----
    n        - Number of elements
    N        - Total number of nodes
    h        - Element size
    EI       - Flexural rigidity of pile
    F        - Axial force at pile head
    y_0      - Displacement at pile head
    V_n      - Shear at pile tip
    M_0, M_n - Moment at pile head/tip
    k_secant - Secant stiffness from p-y curves

    Output:
    ------
    y_updated - Lateral displacement at each node
    '''

    from scipy import linalg

    # Initialize and assemble matrix
    X = np.zeros((N,N))

    # (n+1) finite difference equations for (n+1) real nodes
    for i in range(0,n+1):
        X[i,i]   =  1.0
        X[i,i+1] = -4.0 + F*h**2/EI
        X[i,i+2] =  6.0 - 2*F*h**2/EI + k_secant[i+2]*h**4/EI
        X[i,i+3] = -4.0 + F*h**2/EI
        X[i,i+4] =  1.0

    # Curvature at pile head
    X[n+1,1]   =  1.0
    X[n+1,2]   = -2.0
    X[n+1,3]   =  1.0

    # Shear at pile head
    X[n+2,0]   = -1.0
    X[n+2,1]   =  2.0 - F*h**2/EI
    X[n+2,2]   =  0.0
    X[n+2,3]   = -2.0 + F*h**2/EI
    X[n+2,4]   =  1.0

    # Curvature at pile tip
    X[n+3,-2]  =  1.0
    X[n+3,-3]  = -2.0
    X[n+3,-4]  =  1.0

    # Shear at pile tip
    X[n+4,-1]  =   1.0
    X[n+4,-2]  =  -2.0 + F*h**2/EI
    X[n+4,-3]  =   0.0
    X[n+4,-4]  =   2.0 - F*h**2/EI
    X[n+4,-5]  =  -1.0

    # Repartition X since y_0 is specified.

    # print 'X \n', X

    X1 = np.zeros((N,N))
    X1[:,:] = X[:,:]

    X1[:,2]  = np.zeros(N)
    X1[-3,2] = -1.0/EI

    # X*y = q
    # Initialize vector q
    q = np.zeros(N)

    # Apply essential boundary condition i.e. y_0
    q[0:-4] = -X[0:-4,2]*EI*y_0

    # Populate q with natural boundary conditions
    q[-1] = 2*V_n*h**3 -X[-1,2]*EI*y_0      # Shear at pile tip
    q[-2] = M_n*h**2   -X[-2,2]*EI*y_0      # Moment at pile tip
    q[-3] =            -X[-3,2]*EI*y_0
    q[-4] = M_0*h**2   -X[-4,2]*EI*y_0      # Moment at pile head

    y1 = linalg.solve(EI*X1,q)

    V_0 = y1[2]/(2*h**3)

    y = np.zeros(N)
    y[:] = y1[:]
    y[2] = y_0

    return y, V_0


###############################
#### P-Y Curve Definitions ####
###############################

def matlock_py_curves_SI(z, D, Su, sigma_v_eff, z_0=0.0, epsilon_50=0.02, loading_type='static', print_curves='No',
                     return_Np='No',ls='-'):
    '''Returns an interp1d interpolation function which represents the Matlock (1970) p-y curve at the depth of interest.

    Important: Make sure to import the interp1 function by running 'from scipy.interpolate import interp1d' in the main program.

    Input:
    -----
    z            - Depth relative to pile head (m)
    D            - Pile diameter (m)
    Su           - Undrained shear strength (Pa)
    sigma_v_eff  - Effectve vertical stress (Pa)
    z_0         - Load eccentricity above the mudline or depth to mudline relative to the pile head (m)
    epsilon_50   - Strain at half the strength as defined by Matlock (1970).
                   Typically ranges from 0.005 (stiff clay) to 0.02 (soft clay).
    loading_type - Either 'static' or 'cyclic'

    Optional argument:
    return_Np    - Returns the Np values that in addtion to the p-y curve. This option was added to visualize
                   the gapping effect. It should only be used when this function is used by itself.
                   DO NOT set it to 'Yes' for p-y analysis as the program will crash!

    Output:
    ------
    Returns an interp1d interpolation function which represents the p-y curve at the depth of interest.
    'p' (N/m) and 'y' (m).
    '''

    from scipy.interpolate import interp1d


    # p-y curve properties
    J     = 0.5

    if (z-z_0)<0:
        # p-y curves for the virtual soil layer between the pile head and the mudline should have p=0
        N_p  = 0.0
        z_cr = 1.0 # Dummy value to keep program from crashing

    else:
        try:
            N_p   = 3.0 + sigma_v_eff/Su + J*(z-z_0)/D

            if N_p > 9.0: N_p = 9.0

            z_cr  = (6.0 - sigma_v_eff/Su)*D/J  # This condition is implemented to avoid zero division errors.

        except ZeroDivisionError:
            print("Division by zero! Su = 0.0 so z_cr cannot be calculated.")

    p_ult = Su*N_p*D
    y_50  = 2.5*epsilon_50*D

    # Normalized lateral displacement
    Y = np.concatenate((-np.logspace(3,-4,100),[0],np.logspace(-4,3,100)))

    # Normalized depths
    Z    = z/D
    Z_cr = z_cr/D

    # Normalized p-y curves
    P = 0.5*np.sign(Y)*abs(Y)**(1.0/3.0)  # sign(Y) and abs(Y) used since negative numbers cannot be raised to fractional powers
                                           # Expression equivalent to P = 0.5*Y**(1.0/3.0) for Y>=0
    for i in range(0,len(Y)):
        if P[i] > 1.0:    P[i] = 1.0
        elif P[i] < -1.0: P[i] = -1.0

    if loading_type == 'cyclic':

        for i in range(0,len(Y)):

            if Z<=Z_cr:
                if abs(Y[i]) <= 3:
                    P[i] = P[i]
                elif 3 <= Y[i] <= 15:
                    P[i] = 0.72*(Z/Z_cr - 1)/(15-3) * Y[i] + 0.72*(15-3*Z/Z_cr)/(15-3)
                elif Y[i] > 15:
                    P[i] = 0.72*Z/Z_cr
                elif -3 >= Y[i] >= -15:
                    P[i] = 0.72*(Z/Z_cr - 1)/(15-3) * Y[i] - 0.72*(15-3*Z/Z_cr)/(15-3)
                elif Y[i] < -15:
                    P[i] = -0.72*Z/Z_cr

            else:
                if abs(Y[i]) <= 3:
                    P[i] = P[i]
                elif Y[i]>=3:
                    P[i] = 0.72
                else:
                    P[i] = -0.72


    # Un-normallized p-y curves
    p = P*p_ult
    y = Y*y_50

    f = interp1d(y,p, kind='linear')   # Interpolation function for p-y curve

    # Secant stiffness
    # k = f(y1)/y1

    if print_curves == 'Yes':
        # Plot of p-y curve and check if 'k' is calculated correctly
        plt.plot(y,p,ls), plt.xlabel('y (m)'), plt.ylabel('p (N/m)')
        plt.grid(True)
        plt.xlim([-2*D,2*D])

    if return_Np == 'Yes':
        return f, N_p
    else:
        return f


def MM_1_py_curves_SI(z, D, Su, Su0, σ_v_eff, z_0=0.0, epsilon_50=0.02, gapping='No', alpha = 1.0,
                      loading_type='static', N_eq=0, print_curves='No',ls='-', return_Np='No', return_p_ult='No'):
    '''Returns an interp1d interpolation function which represents the MM-1 p-y curve at the depth of interest.

    Important: Make sure to import the interp1 function by running 'from scipy.interpolate import interp1d' in
    the main program.

    Input:
    -----
    z            - Depth relative to pile head (m)
    D            - Pile diameter (m)
    Su           - Undrained shear strength at depth 'z' (Pa)
    Su0          - Undrained shear strength at depth 'z0' (Pa)
                   Note: If setting Su0 based on the 'interp1d' function for Su from 'design_soil_profile' then,
                   it is safer to set Su0=f_Su(z0+0.01) rather than Su0=f_Su(z0) since f_Su(z0) could be zero
                   and lead to numerical instabilities in the code. 'py_analysis_2()' uses Su0=f_Su(z0+0.01) by
                   default.
    σ_v_eff  - Effectve vertical stress (Pa)
    z_0          - Load eccentricity above the mudline or depth to mudline relative to the pile head (m)
    epsilon_50   - Strain at half the strength as defined by Matlock (1970).
                   Typically ranges from 0.005 (stiff clay) to 0.02 (soft clay).
                   If the 'Auto' option is selected, then epsilon_50 is automatically calculated
                   based on the (S_u/(P_a + σ_v_eff)) ratio. See Senanayake (2016) Phd Thesis.
    gapping      - 'Yes' -> N_p = 2.0*sqrt(3**alpha) + gamma_eff*z/Su + (Su0 + Su)/Su * sqrt(2)*(z/D)
                   'No'  -> N_p = 4.0*sqrt(3**alpha) + (Su0 + Su)/Su * 2*sqrt(2)*(z/D)
    alpha        - Coefficient of pile-soil interface adhesion. 1.0 for a rough pile and 0.0 for a smooth pile.
                   The maximum value of the lateral bearing capacity factor will be calculated as follows:
                   N_p_max = 9 + 3*alpha
                   If the 'API' option is selected, then alpha is automatically calculated
                   based on the (S_u/σ_v_eff) ratio as per Equation 18 in API RP 2GEO (2011),
                   following Randolph & Murphy (1985).
    loading_type - Either 'static' (default) or 'cyclic'. If 'cyclic' is chose, then loading history of the
                   p-y curve has to be provided in terms of equivalent load cycles (N_eq).
    N_eq         - Equivalent load cycles to which this p-y curve (i.e. the soil represented by this curve)
                   has already been subjected. N_eq=0 will be equivalent to a monotonic/static loading
                   (i.e. loading_type=='static')



    Optional argument:
    return_Np    - Returns the Np values that in addtion to the p-y curve. This option was added to check how well visualize the
                   effect and depth of the gap and should only be used when this function is used by itself.
                   DO NOT set it to 'Yes' for p-y analysis as the program will crash!

    Output:
    ------
    Returns an interp1d interpolation function which represents the p-y curve at the depth of interest.
    'p' (N/m) and 'y' (m).
    '''

    from scipy.interpolate import interp1d

    # Calculate gradient of Su vs σ_v_eff profile
    psi = Su/σ_v_eff

    # Calculate alpha based on the API method
    if alpha == 'API':
        if psi<1.0:
            alpha = min(0.5*psi**(-0.5),1.0)
        elif psi>1.0:
            alpha = min(0.5*psi**(-0.25),1.0)
        elif z<z_0:
            alpha=0.0 # Assign default value to alpha above the mudline to avoid numerical errors.
        else:
            print('psi = %2.2f' %psi)
            raise Exception('Failed to calculate alpha based on API method!')

    # Calculate N_p_max based on pile-soil interface adhesion factor
    N_p_max = 9.0 + 3.0*alpha

    # p-y curve properties
    if gapping == 'No':
        N_p0 = 4.0 + alpha*np.pi
        N_p1 = 0.0
        J    = (Su0 + Su)/Su * 2*np.sqrt(2)
    else:
        N_p0 = 2.0 + alpha*np.pi/2
        N_p1 = σ_v_eff/Su
        J    = (Su0 + Su)/Su * np.sqrt(2)

    if (z-z_0)<0:
        # p-y curves for the virtual soil layer between the pile head and the mudline should have p=0
        N_p  = 0.0
        z_cr = 1.0  # Dummy value to keep program from crashing

    else:
        try:
            N_p   = N_p0 + N_p1 + J*(z-z_0)/D

            if N_p > N_p_max: N_p = N_p_max

            z_cr  = (6.0 - σ_v_eff/Su)*D/J  # This condition is implemented to avoid zero division errors.

        except ZeroDivisionError:
            print("Division by zero! Su = 0.0 so z_cr cannot be calculated.")

    if epsilon_50 == 'Auto' and Su!=0:
        # epsilon_50 = min(0.02, 0.004 + 0.0032*(σ_v_eff/Su))

        P_a = 101.325e3  # Pa, atmospheric pressure
        psi = Su/(P_a + σ_v_eff)  # Where (P_a + σ_v_eff) is the confining stress
        epsilon_50 = -0.0318*psi**0.109 + 0.0395  # This relationship was obtained by fitting y = a*x**b + c
                                                  # to Su and epsilon_50 data from Reese et al (1975)

        if epsilon_50 > 0.02:
            epsilon_50 = 0.02
        elif epsilon_50 < 0.004:
            epsilon_50 = 0.004

    elif epsilon_50 == 'Auto' and Su == 0:
        epsilon_50=0.02


    # Modify Su according to cyclic loading history
    if loading_type == 'cyclic':
        Su = Su*(0.3/(N_eq + 1) + 0.7)

    p_ult = Su*N_p*D
    y_50  = 2.5*epsilon_50*D

    # Normalized lateral displacement
    Y = np.concatenate((-np.logspace(3,-4,100),[0],np.logspace(-4,3,100)))

    # Normalized depths
    Z    = z/D
    Z_cr = z_cr/D

    # Normalized p-y curves
    P = 0.5*np.sign(Y)*abs(Y)**(1.0/3.0)  # sign(Y) and abs(Y) used since negative numbers cannot be raised to fractional powers
                                           # Expression equivalent to P = 0.5*Y**(1.0/3.0) for Y>=0
    for i in range(0,len(Y)):
        if P[i] > 1.0:    P[i] = 1.0
        elif P[i] < -1.0: P[i] = -1.0

    # Un-normallized p-y curves
    p = P*p_ult
    y = Y*y_50

    f = interp1d(y,p, kind='linear')   # Interpolation function for p-y curve

    if print_curves == 'Yes':
        # Plot of p-y curve and check if 'k' is calculated correctly
        plt.plot(y,p,ls)
        plt.xlabel('y (m)'), plt.ylabel('p (N/m)')
        plt.grid(True)

    if return_Np == 'Yes':
        return f, N_p
    elif return_p_ult == 'Yes':
        return f, p_ult
    else:
        return f


def jeanjean_2009_py_curves(z,D, Su, sigma_v_eff, Su_0=0.0, z_0=0.0, A=550, print_curves='No', return_Np='No', ls='-'):
    '''
    Returns an interp1d interpolation function which represents the Jeanjean (2009) p-y curve at the depth of interest.

    Important: Make sure to import the interp1 function by running 'from scipy.interpolate import interp1d' in the main program.

    Input:
    -----
    z            - Depth relative to pile head (m)
    D            - Pile diameter (m)
    Su           - Undrained shear strength (Pa)
    sigma_v_eff  - Effectve vertical stress (Pa)
    Su_0         - Undrained shear strength at the mudline (Pa)
    z_0          - Load eccentricity above the mudline or depth to mudline relative to the pile head (m)
    A            - G_max/Su (default = 550)

    Optional argument:
    return_Np    - Returns the Np values that in addtion to the p-y curve. This option was added to check how well visualize the
                   effect and depth of the gap and should only be used when this function is used by itself.
                   DO NOT set it to 'Yes' for p-y analysis as the program will crash!

    Output:
    ------
    Returns an interp1d interpolation function which represents the p-y curve at the depth of interest.
    'p' (N/m) and 'y' (m).
    '''

    from scipy.interpolate import interp1d

    G_max = A*Su

    #Normalized p-y curve
    Y = np.linspace(-3,3,1000)
    P = np.tanh(A/100.0*abs(Y)**(0.5))*np.sign(Y)

    if (z-z_0)<=0:
        #p-y curves for the virtual soil layer between the pile head and the mudline should have p=0
        N_p  = 0.0

    else:
        #P-y curves for the actual soil
        k = (Su - Su_0)/(z - z_0) #Secant gradient of the Su versus depth profile

        '''k is the gradient of the Su profile versus depth. This model is intended to be used with soil profiles
        with linear Su versus depth profiles and Jeanjean (2009) is not clearly on how to calculate k in the case
        of a non-linear Su versus depth profile. In such a case, a tangential gradient, a secant gradient,
        or gradient of fitted straight line could be used with varying results. Back-calculations based on the c/p
        ratio and the submergend unit weight of the clay test bed used by Jeanjean (2009), it seems that he assumed
        a lower bound linear Su profile.'''

        if k == 0:
            lamda = 6
        else:
            lamda = Su_0/(k*D)

        if lamda < 6:
            xi = 0.25 + 0.05*lamda
        else:
            xi = 0.55

        N_p = 12 - 4*np.exp(-xi*(z-z_0)/D)

    #Un-normalize p-y curves
    p_ult = N_p*Su*D

    p = P*p_ult
    y = Y*D

    f = interp1d(y,p, kind='linear')   #Interpolation function for p-y curve

    #Print curves
    if print_curves=='Yes':
        plt.plot(y,p,ls)
        plt.xlabel('y (m)'), plt.ylabel('p (N/m)')
        plt.grid(True)

    if return_Np == 'Yes':
        return f, N_p
    else:
        return f


def jeanjean_2017_py_curves(z, D, Su, sigma_v_eff, z_0=0.0, Su_0=0.0, gapping='No', alpha = 1.0, TE_DSS_ratio=0.9,
                            custom_py='No', a=0.0, strain_f=0.0, print_curves='No', ls='-', return_Np='No'):
    '''Returns an interp1d interpolation function which represents the  p-y curves by Jeanjean et al (2017) at the depth of interest.

    Important:
    -Make sure to import the interp1 function by running 'from scipy.interpolate import interp1d' in the main program.
    -Make sure that Su_0 > 0. If Su_0 = 0, then lamda = 0 => log(lamda) = inf! Therefore, N_p calculation will fail.

    Input:
    -----
    z            - Depth relative to pile head (m)
    D            - Pile diameter (m)
    Su           - Undrained shear strength at depth 'z' (Pa)
    Su_0         - Undrained shear strength at depth 'z0' (Pa)
                   Note: If setting Su0 based on the 'interp1d' function for Su from 'design_soil_profile' then,
                   it is safer to set Su0=f_Su(z0+0.01) rather than Su0=f_Su(z0) since f_Su(z0) could be zero
                   and lead to numerical instabilities in the code. 'py_analysis_2()' uses Su0=f_Su(z0+0.01) by default.
    sigma_v_eff  - Effectve vertical stress (Pa)
    z_0          - Load eccentricity above the mudline or depth to mudline relative to the pile head (m)
    epsilon_50   - Strain at half the strength as defined by Matlock (1970).
                   Typically ranges from 0.005 (stiff clay) to 0.02 (soft clay).
    gapping      - 'Yes' -> N_p = 2.0 + gamma*z/Su + (Su0 + Su)/Su * sqrt(2)*(z/D)
                   'No'  -> N_p = 4.0              + (Su0 + Su)/Su * 2*sqrt(2)*(z/D)
    alpha        - Coefficient of pile-soil interface adhesion. 1.0 for a rough pile and 0.0 for a smooth pile.
                   The maximum value of the lateral bearing capacity factor will be calculated as follows:
                   N_p_max = 9 + 3*alpha
                   Use option 'API' to calculate using the API method (Randolph & Murphy, 1985).
    TE_DSS_ratio - Ratio between the value of Su measured using DSS test and that measured by triaxial extension test.
                   (0.9 by default)
    custom_py    - Define customized p-y curves instead of the default curves, if DSS test data is available.
                   Options: 'Yes', 'No' (default). If yes, parameters 'a' and 'strain_f' have to be provided as input.
    a            - Fitting parameter obtained by optimizing the fit to DSS test data.
                   tau/Su = tanh a*(strain/strain_f)/tanh a
    strain_f     - Total shear strain at failure as determined from DSS test results.
                   (i.e. total strain when Su is mobilized)

    Optional argument:
    return_Np    - Returns the Np values that in addtion to the p-y curve. This option was added to check how well visualize the
                   effect and depth of the gap and should only be used when this function is used by itself.
                   DO NOT set it to 'Yes' for p-y analysis as the program will crash!

    Output:
    ------
    Returns an interp1d interpolation function which represents the p-y curve at the depth of interest.
    'p' (N/m) and 'y' (m).
    '''

    from scipy.interpolate import interp1d

    psi = Su/sigma_v_eff

    #Rate of increase of undrained shear strength with depth in linearly increasing
    #Su vs z profiles. Su = Su_0 + Su_1*depth

    Su_1 = (Su - Su_0)/(z - z_0)    #This works even for nonlinear and layers Su vs z profiles.
                                    #The "equivalent" or "approximate" Su_1 thus calculated is okay
                                    #moderately nonlinear Su vs z profiles (Figure 42, Jeanjean et al 2017)

    #Calculate alpha based on the API method
    if alpha=='API':
        if psi<1.0:
            alpha = min(0.5*psi**(-0.5),1.0)
        elif psi>1.0:
            alpha = min(0.5*psi**(-0.25),1.0)
        elif z<z_0:
            alpha=0.0 #Assign default value to alpha above the mudline to avoid numerical errors.
        else:
            print('psi = {psi:2.2f')
            raise Exception('Failed to calculate alpha based on API method!')

    N_1 = 12.0
    N_2 = 3.22

    #Normalized rate of undrained shear strength increase
    lamda = Su_0/(Su_1*D)

    d = max(16.8 - 2.3*np.log(lamda), 14.5)

    #Maximum value of N_p under plane strain/full flow conditions
    N_p_max = 9.0 + 3.0*alpha  #Refered to as 'N_pd' by Jeanjean et al (2017)

    if (z-z_0)<=0:
        #p-y curves for the virtual soil layer between the pile head and the mudline should have p=0
        N_p  = 0.0

    else:
        #P-y curves for the actual soil
        N_p0 = min(N_1 - (1-alpha) - (N_1 - N_2)*(1.0 - ((z-z_0)/d/D)**0.6)**1.35, N_p_max)

        if np.isnan(N_p0)==True:
            N_p0 = N_p_max

        if gapping=='Yes':
            N_p = min(N_p0 + sigma_v_eff/Su, N_p_max)

        else:
            N_p = min(2*N_p0, N_p_max)

    if gapping=='Yes' and N_p < N_p_max:
    #Reduce Su_DSS to Su_TE (i.e direct simple shear to triaxial extension) in shallow wedge region
        N_p_z0 = N_1 - (1-alpha) - (N_1 - N_2)
        C = 1 + (TE_DSS_ratio - 1)*(N_p_max - N_p)/(N_p_max - N_p_z0)

        N_p = C*N_p

    p_ult = Su*N_p*D

    #Normalized p-y curves
    if custom_py=='Yes' or custom_py=='yes':

        if a==0.0 or strain_f==0.0:
            print(f'a = {a:2.2f}, strain_f = {strain_f:2.2f}')
            raise Exception('Input values for a and strain_f required!')


        else:
            A = 1.33 + 0.45*np.log(a)
            Y_max = strain_f*(2.5-1.2*np.log(a))

            Y = np.concatenate((-np.logspace(1,-4,25),[0],np.logspace(-4,1,25)))

            P = np.sign(Y)*np.tanh(A*(abs(Y)/Y_max)**0.5) / np.tanh(A)

            for i in range(0,len(P)):
                if P[i] > 1.0:
                    P[i] = 1.0
                elif P[i] < -1.0:
                    P[i] = -1.0

    else:
        P = np.array([-1.0,-1.0,-0.975,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.05,
                       0,0.05,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.975,1.0,1.0])

        if Su <= 14.5: #i.e. Su < 14.5psi, 100kPa
            Y = np.array([-10.,-0.25,-0.15,-0.082,-0.05,-0.032,-0.022,-14.5e-3,-9e-3,-5.3e-3,-3e-3,-3e-4,
                       0,3e-4,3e-3,5.3e-3,9e-3,14.5e-3,0.022,0.032,0.05,0.082,0.15,0.25,10.])

        else: #i.e Su > 14.5psi, 100kPa
            Y = np.array([-10.,-0.25,-0.16,-0.114,-0.07,-0.045,-0.03,-0.02,-12.5e-3,-7e-3,-3.5e-3,-4e-4,
                       0,4e-4,3.5e-3,7e-3,12.5e-3,0.02,0.03,0.045,0.07,0.114,0.16,0.25,10.])

    #Un-normallized p-y curves
    p = P*p_ult
    y = Y*D

    f = interp1d(y,p, kind='linear')   #Interpolation function for p-y curve

    if print_curves=='Yes':
        #Plot of p-y curve and check if 'k' is calculated correctly
        plt.plot(y,p,ls)
        plt.xlabel('y (m)'), plt.ylabel('p (N/m)')
        plt.grid(True)

    if return_Np == 'Yes':
        return f, N_p
    else:
        return f


#######################
#### Soil Profile #####
#######################

def design_soil_profile_SI(soil_profile, plot_profile='No', y_axis_label='Depth below the pile head (m)'):
    '''Define the soil profile used by the p-y analyzer. Outputs 'interp1d' functions containing Su and sigma'_v
    profiles to be used by the p-y curve functions.

    Input:
    -----
    soil_profile - A 2D tuple in the following format: ([Depth (m), Su (kPa), gamma_sub (kN/m^3), py-model, model parameter])
                   The soil profile should be defined relative to the pile/tower head (i.e. point of lateral load application)
                   so that any load eccentricities can be taken into account. An example soil profile is shown below.
                   Eg: array([[z0,Su0,gamma_sub0,  'Matlock', 0.02],
                              [z1,Su1,gamma_sub1,  'Matlock', 0.01],
                              [z2,Su2,gamma_sub2, 'Jeanjean',  550],
                              ...])

                    *The current program cannot define layers with different p-y models. But it will added in the future.

    plot_profile - Plot Su vs depth profile. Choose 'Yes' to plot.

    Output:
    ------
    z0            - Depth of mudline relative to the pile head (m)
    f_Su          - 'interp1d' function containing undrained shear strength profile (Pa)
    f_sigma_v_eff - 'interp1d' function containing effective vertical stress profile (Pa)
    '''

    from scipy.interpolate import interp1d

    # Depth of mudline relative to pile head
    z0 = soil_profile[0,0].astype(float)

    # Extract data from soil_profile array and zero strength virtual soil layer
    # from the pile head down to the mudline
    depth     = np.concatenate([np.array([0,z0]), soil_profile[:,0].astype(float)])  # m
    Su        = np.concatenate([np.array([0, 0]), soil_profile[:,1].astype(float)])  # kPa
    gamma_sub = np.concatenate([np.array([0, 0]), soil_profile[:,2].astype(float)])  # kN/m^3

    if plot_profile == 'Yes':
        # Plot Su vs z profile for confirmation
        plt.plot(Su, depth, '-', label=r'$S_u$')
        plt.legend(loc='lower left')
        plt.xlabel('Undrained shear strength (kPa)'), plt.ylabel(y_axis_label), plt.grid(True)

        # Plot mudline/ground surface
        plt.plot([-0.5*max(Su),max(Su)], [z0,z0], '--', color='brown')
        plt.text(-0.5*max(Su), 0.95*z0, 'Mudline', color='brown')

        ax = plt.gca()
        ax.invert_yaxis(), ax.xaxis.tick_top(), ax.xaxis.set_label_position('top')

    # Calculate sigma_v_eff at each depth
    sigma_v_eff = np.zeros(len(depth))

    for i in range(1,len(depth)):
        sigma_v_eff[i] = sigma_v_eff[i-1] + gamma_sub[i-1]*(depth[i]-depth[i-1])

    # Define interpolation functions
    f_Su          = interp1d(depth, Su*1000,          kind='linear') # Pa
    f_sigma_v_eff = interp1d(depth, sigma_v_eff*1000, kind='linear') # Pa

    return z0, f_Su, f_sigma_v_eff

