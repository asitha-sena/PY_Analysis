#Copyright Asitha Senanayake 2016

from pylab import *

###################################
#### Pile Geometry and Loading ####
###################################

def py_analysis_1(Su_profile, L=10.0, D=0.375, t = 0.1, E=29e6, F = 0.0, V_0=0.0, V_n=0.0, M_0=0.0, M_n=0.0, n=50, 
                  iterations=15, convergence_tracker='No', loc=2, py_model='Matlock'):
    '''Models a laterally loaded pile using the p-y method. The solution for lateral displacements 
    is obtained by solving the 4th order ODE, EI*d4y/dz4 -F*d2y/dz2 + ky = 0 using the finite difference method.

    Takes natural boundary conditions at the pile head and tip. For displacement controlled analysis check
    'py_analysis_2'.
    
    Assumes that EI remains constant with respect to curvature i.e. pile material remains in the elastic region.

    Input:
    -----
    Su_profile  - A 2D array of depths (in) and corresponding undrained shear strength(psf)
                  Eg: array([[z1,Su1],[z2,Su2],[z3,Su3]...])
                  Use small values for Su (eg: 0.001) instead of zeros to avoid divisions by zero but always start z at 0.0
                  Example of a valid data point at the mudline is [0.0, 0.001]
                  
    L           - Length of pile         (inches)
    D           - Outer diameter of pile (inches)
    t           - Wall thickness of pile (inches)
    E           - Elastic modulus of pile material (psi)
    F           - Axial force at pile head (lb), vertically downwards is postive.
    V_0, V_n    - Force at pile head/tip  (lb),  shear causing clockwise rotation of pile is positive.
    M_0, M_n    - Moment at pile head/tip (lb-in), moments causing tension on left side of pile is positive.
    n           - Number of elements (50 by default)
    iterations  - Number of iterations to repeat calculation in order obtain convergence of 'y'
                  (A better approach is to iterate until a predefined tolerance is achieved but this requires additional
                  coding so, I will implement this later.)
    py_model    - Select which p-y model to use, 'Matlock' or 'Jeanjean'.
    
    Optional:
    convergence_tracker - Track how k_secant converges to actual p-y curve at a selected node
    loc                 - Node number at which k_secant to be tracked (2 to n+1)
    
    Output:
    ------
    y           - Lateral displacement at each node, length = n + 5, (n+1) real nodes and 4 imaginary nodes
    z           - Vector of node locations along pile
    '''
    
    from scipy import linalg
    
    #Pile geometry
    I  = pi*(D**4 - (D-2*t)**4)/64.0
    EI = E*I
    h  = L/n  #Element size
    N  = (n+1)+4      # (n+1) Real + 4 Imaginary nodes 
    
    #Array for displacement including imaginary nodes
    y = ones(N)*(0.01*D)   #An initial value of 0.01D was arbitrarily chosen
    
    #Initialize and assemble array/list of p-y curves at each real node
    z = zeros(N)
    py_funs  = []
    k_secant = zeros(N)
    
    for i in [0,1]:        #Top two imaginary nodes
        z[i] = (i-2)*h
        py_funs.append(0)
        k_secant[i] = 0.0
    
    for i in range(2,n+3): #Real nodes
        z[i] = (i-2)*h
        Su, sigma_v_eff = design_soil_profile(z[i], Su_profile)

        if py_model=='Matlock':
            py_funs.append(matlock_py_curves(z[i], D, Su, sigma_v_eff, epsilon_50=0.02, print_curves='No'))
        elif py_model=='Jeanjean':
            py_funs.append(jeanjean_py_curves(z[i], D, Su_profile))
        elif py_model=='Modified Matlock':
            py_funs.append(modified_matlock_py_curves(z[i], D, Su, sigma_v_eff, epsilon_50=0.02, print_curves='No'))
            
        k_secant[i]     = py_funs[i](y[i])/y[i]
    
    for i in [n+3, n+4]:   #Bottom two imaginary nodes
        z[i] = (i-2)*h
        py_funs.append(0)
        k_secant[i] = 0.0
    
    #Track k_secant and current displacements
    if convergence_tracker=='Yes':
        figure()
        y1 = linspace(-0.5*D,0.5*D,500) 
        plot(y1, py_funs[loc](y1))
        xlabel('y (in)'), ylabel('p (lb/in)'), grid(True)
        
    for j in range(iterations):
        
        y = fd_solver_1(n,N,h,EI,F,V_0,V_n,M_0,M_n,k_secant)
        
        if convergence_tracker=='Yes':
            plot(y[loc], k_secant[loc]*y[loc], 'x')
            
        for i in range(2, n+3):
            k_secant[i] = py_funs[i](y[i])/y[i]
            
            
    return y,z


def py_analysis_1_1(Su_profile, L=10.0, D=0.375, t = 0.1, E=29e6, F = 0.0, y_0=0.0, V_0=0.0, V_n=0.0, M_0=0.0, M_n=0.0, n=50, 
                  iterations=15, convergence_tracker='No', loc=2):
    '''Models a laterally loaded pile using the p-y method. The solution for lateral displacements 
    is obtained by solving the 4th order ODE, EI*d4y/dz4 -F*d2y/dz2 + ky = 0 using the finite difference method.
    
    Uses 'py_analysis_1' function iteratively to converge on a specified pile head displacement value and find the
    corresponding pile head load (V_0). An initial estimate of the pile head load has to be provided. 'py_analysis_2'
    implements a direct solution to this problem and is more robust and efficient.
    
    Assumes that EI remains constant with respect to curvature i.e. pile material remains in the elastic region.
    
    Input:
    -----
    Su_profile  - A 2D array of depths (in) and corresponding undrained shear strength(psf)
                  Eg: array([[z1,Su1],[z2,Su2],[z3,Su3]...])
                  Use small values for Su (eg: 0.001) instead of zeros to avoid divisions by zero but always start z at 0.0
                  Example of a valid data point at the mudline is [0.0, 0.001]
                  
    L           - Length of pile         (inches)
    D           - Outer diameter of pile (inches)
    t           - Wall thickness of pile (inches)
    E           - Elastic modulus of pile material (psi)
    F           - Axial force at pile head (lb), vertically downwards is postive.
    V_0, V_n    - Force at pile head/tip  (lb),  shear causing clockwise rotation of pile is positive.
    M_0, M_n    - Moment at pile head/tip (lb-in), moments causing tension on left side of pile is positive.
    n           - Number of elements (50 by default)
    iterations  - Number of iterations to repeat calculation in order obtain convergence of 'y'
                  (A better approach is to iterate until a predefined tolerance is achieved but this requires additional
                  coding so, I will implement this later.)
    
    Optional:
    convergence_tracker - Track how k_secant converges to actual p-y curve at a selected node
    loc                 - Node number at which k_secant to be tracked (2 to n+1)
    
    Output:
    ------
    y           - Lateral displacement at each node, length = n + 5, (n+1) real nodes and 4 imaginary nodes
    l           - Vector of node locations along pile
    X           - Finite difference matrix 
    '''
    
    from scipy import linalg
    
    #Pile geometry
    I  = pi*(D**4 - (D-2*t)**4)/64.0
    EI = E*I
    h  = L/n  #Element size
    N  = (n+1)+4      # (n+1) Real + 4 Imaginary nodes 
    
    #Array for displacement including imaginary nodes
    y = ones(N)*(0.01*D)   #An initial value of 0.01D was arbitrarily chosen
    
    #Initialize and assemble array/list of p-y curves at each real node
    z = zeros(N)
    py_funs  = []
    k_secant = zeros(N)
    
    for i in [0,1]:        #Top two imaginary nodes
        z[i] = (i-2)*h
        py_funs.append(0)
        k_secant[i] = 0.0
    
    for i in range(2,n+3): #Real nodes
        z[i] = (i-2)*h
        Su, sigma_v_eff = design_soil_profile(z[i], Su_profile)
        py_funs.append(matlock_py_curves(z[i], D, Su, sigma_v_eff, epsilon_50=0.02, print_curves='No'))
        k_secant[i]     = py_funs[i](y[i])/y[i]
    
    for i in [n+3, n+4]:   #Bottom two imaginary nodes
        z[i] = (i-2)*h
        py_funs.append(0)
        k_secant[i] = 0.0
    
    figure()
    #Track k_secant and current displacements
    if convergence_tracker=='Yes':
        y1 = linspace(-0.5*D,0.5*D,500) 
        plot(y1, py_funs[loc](y1))
        xlabel('y (in)'), ylabel('p (lb/in)'), grid(True)
        
    for j in range(iterations):
        
        y = fd_solver_1(n,N,h,EI,F,V_0,V_n,M_0,M_n,k_secant)
        
        if convergence_tracker=='Yes':
            plot(y[loc], k_secant[loc]*y[loc], 'x')
            
        for i in range(2, n+3):
            k_secant[i] = py_funs[i](y[i])/y[i]
    
    step = 0
    
    while abs(y_0 - y[2]) > 0.005:
        
        step += 1
        print 'Step %d, \t V_0 = %.2f-lb, \t y[2] = %.2f-in' %(step,V_0, y[2])
        
        V_0 += V_0*(y_0 - y[2])/y_0
        
        if step > 20: break
            
        for j in range(iterations):
        
            y = fd_solver_1(n,N,h,EI,F,V_0,V_n,M_0,M_n,k_secant)
            
            if convergence_tracker=='Yes':
                plot(y[loc], k_secant[loc]*y[loc], 'x')
                
            for i in range(2, n+3):
                k_secant[i] = py_funs[i](y[i])/y[i]
                
    print 'V_0 = %.2f-lb' %V_0
            
    return y,z



def py_analysis_2(soil_profile, L=10.0, D=2.0, t = 0.1, E=29e6, F = 0.0, y_0=0.0, M_0=0.0, V_n=0.0, M_n=0.0, n=50, 
                  iterations=10, py_model='Matlock', print_output='Yes', convergence_tracker='No', loc=2, **kwargs):
    '''Models a laterally loaded pile using the p-y method. The solution for lateral displacements 
    is obtained by solving the 4th order ODE, EI*d4y/dz4 -F*d2y/dz2 + ky = 0 using the finite difference method.
    
    Assumes that EI remains constant with respect to curvature i.e. pile material remains in the elastic region.
    
    Input:
    -----
    soil_profile - A 2D array of depths (in), corresponding undrained shear strength(psf), 
                   and corresponding vertical effective stress (psf)
                   Eg: array([[z1,Su1,sigma_v_eff1],[z2,Su2,sigma_v_eff2],[z3,Su3,sigma_v_eff3]...])
                   Use small values for Su and sigma_v_eff (eg: 0.001) instead of zeros to avoid divisions 
                   by zero but always start z at 0.0.
                   Example of a valid data point at the mudline is [0.0, 0.001, 0.001]
                  
    L           - Length of pile         (inches)
    D           - Outer diameter of pile (inches)
    t           - Wall thickness of pile (inches)
    E           - Elastic modulus of pile material (psi)
    y_0         - Displacement at the pile head (in)
    F           - Axial force at pile head (lb), vertically downwards is postive.
    M_0         - Moment at pile head (lb-in), moments causing tension on left side of pile is positive.
    V_n         - Force at pile head/tip  (lb),  shear causing clockwise rotation of pile is positive.
    M_n         - Moment at pile tip (lb-in), moments causing tension on left side of pile is positive.
    n           - Number of elements (50 by default)
    iterations  - Number of iterations to repeat calculation in order obtain convergence of 'y'
                  (A better approach is to iterate until a predefined tolerance is achieved but this requires additional
                  coding so, I will implement this later.)
    py_model    - Select which p-y model to use, 'Matlock', 'Jeanjean', 'MM-1', or 'Kodikara'.
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

    #Extract optional keyword arguments
    epsilon_50, A, gapping, N_p_max = 0.02, 550, 'No', 12.0 #Default parameters if no **kwargs are defined
    custom_py, a, strain_f          = 'No', 0.0, 0.0
    ls                              = 'x'
    
    for arg in kwargs:
        if arg=='epsilon_50':
            epsilon_50 = kwargs[arg]        
        if arg=='Gmax_Su_ratio':
            A = kwargs[arg]
        if arg=='gapping':
            gapping=kwargs[arg]
        if arg=='N_p_max':
            N_p_max=kwargs[arg]
        if arg=='alpha':
            alpha=kwargs[arg]
        if arg=='custom_py':
            custom_py=kwargs[arg]
        if arg=='a':
            a = kwargs[arg]
        if arg=='strain_f':
            strain_f=kwargs[arg]
        if arg=='ls':
            ls=kwargs[arg]
            
    #Convert L and D to floating point numbers to avoid rounding errors
    L = float(L)
    D = float(D)
    
    #Pile geometry
    I  = pi*(D**4 - (D-2*t)**4)/64.0
    EI = E*I
    h  = L/n  #Element size
    N  = (n+1)+4      # (n+1) Real + 4 Imaginary nodes 
    
    #Array for displacements at nodes, including imaginary nodes.
    y = ones(N)*(0.01*D)   #An initial value of 0.01D was arbitrarily chosen
    
    #Initialize and assemble array/list of p-y curves at each real node
    z = zeros(N)
    py_funs  = []
    k_secant = zeros(N)
    
    for i in [0,1]:        #Top two imaginary nodes
        z[i] = (i-2)*h
        py_funs.append(0)
        k_secant[i] = 0.0
    
    #Extract soil profile data
    z_0, f_Su, f_sigma_v_eff = design_soil_profile(soil_profile)
    
    for i in range(2,n+3): #Real nodes
        z[i] = (i-2)*h
        
        Su, sigma_v_eff = f_Su(z[i]), f_sigma_v_eff(z[i])
        
        if py_model=='Matlock':
            py_funs.append(matlock_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=epsilon_50, print_curves='No'))
        elif py_model=='Jeanjean':
            py_funs.append(jeanjean_py_curves(z[i], D,Su, sigma_v_eff, z_0=z_0, Su_0=f_Su(z_0), A=A))
        elif py_model=='Modified Matlock':
            py_funs.append(modified_matlock_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=epsilon_50, print_curves='No'))
        elif py_model=='Kodikara':
            py_funs.append(kodikara_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, R1=0.5, A=A, print_curves='No'))
        elif py_model=='Matlock No Gapping':
            py_funs.append(matlock_py_curves_no_gapping(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=epsilon_50, print_curves='No'))
        elif py_model=='MM-1':
            py_funs.append(MM_1_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=epsilon_50, print_curves='No',
                                                  Su0=f_Su(z_0+0.01), gapping=gapping, alpha=alpha))
        elif py_model=='MM-2':
            py_funs.append(MM_2_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=epsilon_50, print_curves='No',
                                                  Su0=f_Su(z_0+0.01), gapping=gapping, N_p_max=N_p_max))
        elif py_model=='Jeanjean_etal_2017':
            py_funs.append(jeanjean_2017_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, print_curves='No',
                                                  Su_0=f_Su(z_0+0.01), gapping=gapping, alpha=alpha, 
                                                   custom_py=custom_py, a=a, strain_f=strain_f))
        else:
            print "P-y model not properly defined. Please select one of the following:"
            print "'Matlock', 'Jeanjean', 'Modified Matlock', 'Kodikara', 'MM-1', 'MM-2', 'Jeanjean et al 2017'"

        k_secant[i]     = py_funs[i](y[i])/y[i]
    
    for i in [n+3, n+4]:   #Bottom two imaginary nodes
        z[i] = (i-2)*h
        py_funs.append(0)
        k_secant[i] = 0.0
    
    #Track k_secant and current displacements
    if convergence_tracker=='Yes':
        y1 = linspace(-2.*D,2.*D,500) 
        plot(y1, py_funs[loc](y1))
        xlabel('y (in)'), ylabel('p (lb/in)'), grid(True)
        
    for j in range(iterations):
        #if j==0: print 'FD Solver started!'
            
        y,V_0 = fd_solver_2(n,N,h,EI,F,y_0,V_n,M_0,M_n,k_secant)
        
        if convergence_tracker=='Yes':
            plot(y[loc], k_secant[loc]*y[loc], ls)
            
        for i in range(2, n+3):
            k_secant[i] = py_funs[i](y[i])/y[i]
    
    if print_output=='Yes':
        print 'V_0 = %.2f-lb' %V_0
    
    return y,z,V_0


def py_analysis_21(soil_profile, L=10.0, D=2.0, t = 0.1, E=29e6, F = 0.0, y_0=0.0, s_0=0.0, V_n=0.0, M_n=0.0, n=50, 
                  iterations=10, py_model='Matlock', print_output='Yes', convergence_tracker='No', loc=2, **kwargs):
    '''Models a laterally loaded pile using the p-y method. The solution for lateral displacements 
    is obtained by solving the 4th order ODE, EI*d4y/dz4 -F*d2y/dz2 + ky = 0 using the finite difference method.
    
    NOTE: This function is NOT the same as 'py_analysis_2_1'. This lets the pile head slope be defined instead of the
    pile head moment so that fixed head piles can be analyzed. It uses 'fd_solver_21'.

    Assumes that EI remains constant with respect to curvature i.e. pile material remains in the elastic region.
    
    Input:
    -----
    soil_profile - A 2D array of depths (in), corresponding undrained shear strength(psf), 
                   and corresponding vertical effective stress (psf)
                   Eg: array([[z1,Su1,sigma_v_eff1],[z2,Su2,sigma_v_eff2],[z3,Su3,sigma_v_eff3]...])
                   Use small values for Su and sigma_v_eff (eg: 0.001) instead of zeros to avoid divisions 
                   by zero but always start z at 0.0.
                   Example of a valid data point at the mudline is [0.0, 0.001, 0.001]
                  
    L           - Length of pile         (inches)
    D           - Outer diameter of pile (inches)
    t           - Wall thickness of pile (inches)
    E           - Elastic modulus of pile material (psi)
    y_0         - Displacement at the pile head (in)
    F           - Axial force at pile head (lb), vertically downwards is postive.
    s_0         - Slope at pile head (in/in), not sure which of the sign convention. Check deflection profile to be sure.
    V_n         - Force at pile head/tip  (lb),  shear causing clockwise rotation of pile is positive.
    M_n         - Moment at pile tip (lb-in), moments causing tension on left side of pile is positive.
    n           - Number of elements (50 by default)
    iterations  - Number of iterations to repeat calculation in order obtain convergence of 'y'
                  (A better approach is to iterate until a predefined tolerance is achieved but this requires additional
                  coding so, I will implement this later.)
    py_model    - Select which p-y model to use, 'Matlock', 'Jeanjean', 'MM-1', or 'Kodikara'.
    print_output - Print results on screen, Choose 'Yes' or 'No' (default = 'No')
    
    Optional:
    convergence_tracker - Track how k_secant converges to actual p-y curve at a selected node
    loc                 - Node number at which k_secant to be tracked (2 to n+1)

    Optional keywords: **kwargs
    epsilon_50  - Define 'epsilon_50' if 'Matlock' p-y model is selected. (epsilon_50=0.02 by default)
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

    #Extract optional keyword arguments
    epsilon_50, A, gapping, N_p_max = 0.02, 550, 'No', 12.0 #Default parameters if no **kwargs are defined
    for arg in kwargs:
        if arg=='epsilon_50':
            epsilon_50 = kwargs[arg]        
        if arg=='Gmax_Su_ratio':
            A = kwargs[arg]
        if arg=='gapping':
            gapping=kwargs[arg]
        if arg=='N_p_max':
            N_p_max=kwargs[arg]
        if arg=='alpha':
            alpha=kwargs[arg]
            
    #Convert L and D to floating point numbers to avoid rounding errors
    L = float(L)
    D = float(D)
    
    #Pile geometry
    I  = pi*(D**4 - (D-2*t)**4)/64.0
    EI = E*I
    h  = L/n  #Element size
    N  = (n+1)+4      # (n+1) Real + 4 Imaginary nodes 
    
    #Array for displacements at nodes, including imaginary nodes.
    y = ones(N)*(0.01*D)   #An initial value of 0.01D was arbitrarily chosen
    
    #Initialize and assemble array/list of p-y curves at each real node
    z = zeros(N)
    py_funs  = []
    k_secant = zeros(N)
    
    for i in [0,1]:        #Top two imaginary nodes
        z[i] = (i-2)*h
        py_funs.append(0)
        k_secant[i] = 0.0
    
    #Extract soil profile data
    z_0, f_Su, f_sigma_v_eff = design_soil_profile(soil_profile)
    
    for i in range(2,n+3): #Real nodes
        z[i] = (i-2)*h
        
        Su, sigma_v_eff = f_Su(z[i]), f_sigma_v_eff(z[i])
        
        if py_model=='Matlock':
            py_funs.append(matlock_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=epsilon_50, print_curves='No'))
        elif py_model=='Jeanjean':
            py_funs.append(jeanjean_py_curves(z[i], D,Su, sigma_v_eff, z_0=z_0, Su_0=f_Su(z_0), A=A))
        elif py_model=='Modified Matlock':
            py_funs.append(modified_matlock_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=epsilon_50, print_curves='No'))
        elif py_model=='Kodikara':
            py_funs.append(kodikara_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, R1=0.5, A=A, print_curves='No'))
        elif py_model=='Matlock No Gapping':
            py_funs.append(matlock_py_curves_no_gapping(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=epsilon_50, print_curves='No'))
        elif py_model=='MM-1':
            py_funs.append(MM_1_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=epsilon_50, print_curves='No',
                                                  Su0=f_Su(z_0+0.01), gapping=gapping, alpha=alpha))
        elif py_model=='MM-2':
            py_funs.append(MM_2_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=epsilon_50, print_curves='No',
                                                  Su0=f_Su(z_0+0.01), gapping=gapping, N_p_max=N_p_max))
        elif py_model=='Jeanjean_etal_2017':
            py_funs.append(jeanjean_2017_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, print_curves='No',
                                                  Su_0=f_Su(z_0+0.01), gapping=gapping, alpha=alpha))
        else:
            print "P-y model not properly defined. Please select one of the following:"
            print "'Matlock', 'Jeanjean', 'Modified Matlock', 'Kodikara', 'MM-1', 'MM-2','Jeanjean_etal_2017'"
        
        k_secant[i]     = py_funs[i](y[i])/y[i]
    
    for i in [n+3, n+4]:   #Bottom two imaginary nodes
        z[i] = (i-2)*h
        py_funs.append(0)
        k_secant[i] = 0.0
    
    #Track k_secant and current displacements
    if convergence_tracker=='Yes':
        y1 = linspace(-2.*D,2.*D,500) 
        plot(y1, py_funs[loc](y1))
        xlabel('y (in)'), ylabel('p (lb/in)'), grid(True)
        
    for j in range(iterations):
        #if j==0: print 'FD Solver started!'
            
        y,V_0 = fd_solver_21(n,N,h,EI,F,y_0,V_n,s_0,M_n,k_secant)
        
        if convergence_tracker=='Yes':
            plot(y[loc], k_secant[loc]*y[loc], 'x')
            
        for i in range(2, n+3):
            k_secant[i] = py_funs[i](y[i])/y[i]
    
    if print_output=='Yes':
        print 'V_0 = %.2f-lb' %V_0
    
    return y,z,V_0


def py_analysis_22(soil_profile, L=10.0, D=2.0, t = 0.1, E=29e6, F = 0.0, y_0=0.0, y_n=0.0, M_0=0.0, M_n=0.0, n=50, 
                  iterations=10, py_model='Matlock', print_output='Yes', convergence_tracker='No', loc=2, **kwargs):
    '''Models a laterally loaded pile using the p-y method. The solution for lateral displacements 
    is obtained by solving the 4th order ODE, EI*d4y/dz4 -F*d2y/dz2 + ky = 0 using the finite difference method.
    
    Assumes that EI remains constant with respect to curvature i.e. pile material remains in the elastic region.
    
    NOTE: This function allows the pile tip displacement (i.e y_n) to be defined instead of the pile tip force.
          It calls fd_solver_22() to solve the finite-difference equations.
    
    Input:
    -----
    soil_profile - A 2D array of depths (in), corresponding undrained shear strength(psf), 
                   and corresponding vertical effective stress (psf)
                   Eg: array([[z1,Su1,sigma_v_eff1],[z2,Su2,sigma_v_eff2],[z3,Su3,sigma_v_eff3]...])
                   Use small values for Su and sigma_v_eff (eg: 0.001) instead of zeros to avoid divisions 
                   by zero but always start z at 0.0.
                   Example of a valid data point at the mudline is [0.0, 0.001, 0.001]
                  
    L           - Length of pile         (inches)
    D           - Outer diameter of pile (inches)
    t           - Wall thickness of pile (inches)
    E           - Elastic modulus of pile material (psi)
    y_0         - Displacement at the pile head (in) (y_n =! 0 to avoid singular matrix)
    y_n         - Displacement at the pile tip (in). (y_n =! 0 to avoid singular matrix)
    F           - Axial force at pile head (lb), vertically downwards is postive.
    M_0         - Moment at pile head (lb-in), moments causing tension on left side of pile is positive.
    M_n         - Moment at pile tip (lb-in), moments causing tension on left side of pile is positive.
    n           - Number of elements (50 by default)
    iterations  - Number of iterations to repeat calculation in order obtain convergence of 'y'
                  (A better approach is to iterate until a predefined tolerance is achieved but this requires additional
                  coding so, I will implement this later.)
    py_model    - Select which p-y model to use, 'Matlock', 'Jeanjean', 'MM-1', or 'Kodikara'.
    print_output - Print results on screen, Choose 'Yes' or 'No' (default = 'No')
    
    Optional:
    convergence_tracker - Track how k_secant converges to actual p-y curve at a selected node
    loc                 - Node number at which k_secant to be tracked (2 to n+1)

    Optional keywords: **kwargs
    epsilon_50  - Define 'epsilon_50' if 'Matlock' p-y model is selected. (epsilon_50=0.02 by default)
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

    #Extract optional keyword arguments
    epsilon_50, A, gapping, N_p_max = 0.02, 550, 'No', 12.0 #Default parameters if no **kwargs are defined
    for arg in kwargs:
        if arg=='epsilon_50':
            epsilon_50 = kwargs[arg]        
        if arg=='Gmax_Su_ratio':
            A = kwargs[arg]
        if arg=='gapping':
            gapping=kwargs[arg]
        if arg=='N_p_max':
            N_p_max=kwargs[arg]
        if arg=='alpha':
            alpha=kwargs[arg]
            
    #Convert L and D to floating point numbers to avoid rounding errors
    L = float(L)
    D = float(D)
    
    #Pile geometry
    I  = pi*(D**4 - (D-2*t)**4)/64.0
    EI = E*I
    h  = L/n  #Element size
    N  = (n+1)+4      # (n+1) Real + 4 Imaginary nodes 
    
    #Array for displacements at nodes, including imaginary nodes.
    y = ones(N)*(0.01*D)   #An initial value of 0.01D was arbitrarily chosen
    
    #Initialize and assemble array/list of p-y curves at each real node
    z = zeros(N)
    py_funs  = []
    k_secant = zeros(N)
    
    for i in [0,1]:        #Top two imaginary nodes
        z[i] = (i-2)*h
        py_funs.append(0)
        k_secant[i] = 0.0
    
    #Extract soil profile data
    z_0, f_Su, f_sigma_v_eff = design_soil_profile(soil_profile)
    
    for i in range(2,n+3): #Real nodes
        z[i] = (i-2)*h
        
        Su, sigma_v_eff = f_Su(z[i]), f_sigma_v_eff(z[i])
        
        if py_model=='Matlock':
            py_funs.append(matlock_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=epsilon_50, print_curves='No'))
        elif py_model=='Jeanjean':
            py_funs.append(jeanjean_py_curves(z[i], D,Su, sigma_v_eff, z_0=z_0, Su_0=f_Su(z_0), A=A))
        elif py_model=='Modified Matlock':
            py_funs.append(modified_matlock_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=epsilon_50, print_curves='No'))
        elif py_model=='Kodikara':
            py_funs.append(kodikara_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, R1=0.5, A=A, print_curves='No'))
        elif py_model=='Matlock No Gapping':
            py_funs.append(matlock_py_curves_no_gapping(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=epsilon_50, print_curves='No'))
        elif py_model=='MM-1':
            py_funs.append(MM_1_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=epsilon_50, print_curves='No',
                                                  Su0=f_Su(z_0+0.01), gapping=gapping, alpha=alpha))
        elif py_model=='MM-2':
            py_funs.append(MM_2_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=epsilon_50, print_curves='No',
                                                  Su0=f_Su(z_0+0.01), gapping=gapping, N_p_max=N_p_max))
        elif py_model=='Jeanjean_etal_2017':
            py_funs.append(jeanjean_2017_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, print_curves='No',
                                                  Su_0=f_Su(z_0+0.01), gapping=gapping, alpha=alpha))
        else:
            print "P-y model not properly defined. Please select one of the following:"
            print "'Matlock', 'Jeanjean', 'Modified Matlock', 'Kodikara', 'MM-1', 'MM-2', 'Jeanjean_etal_2017'"
        
        k_secant[i]     = py_funs[i](y[i])/y[i]
    
    for i in [n+3, n+4]:   #Bottom two imaginary nodes
        z[i] = (i-2)*h
        py_funs.append(0)
        k_secant[i] = 0.0
    
    #Track k_secant and current displacements
    if convergence_tracker=='Yes':
        y1 = linspace(-2.*D,2.*D,500) 
        plot(y1, py_funs[loc](y1))
        xlabel('y (in)'), ylabel('p (lb/in)'), grid(True)
        
    for j in range(iterations):
        #if j==0: print 'FD Solver started!'
            
        y,V_0 = fd_solver_22(n,N,h,EI,F,y_0,y_n,M_0,M_n,k_secant)
        
        if convergence_tracker=='Yes':
            plot(y[loc], k_secant[loc]*y[loc], 'x')
            
        for i in range(2, n+3):
            k_secant[i] = py_funs[i](y[i])/y[i]
    
    if print_output=='Yes':
        print 'V_0 = %.2f-lb' %V_0
    
    return y,z,V_0


def py_analysis_2_1(soil_profile, L=10.0, D=2.0, t = 0.1, E=29e6, F = 0.0, y_0=0.0, M_0=0.0, V_n=0.0, M_n=0.0, n=50, 
                  iterations=10, py_model='Matlock', print_output='Yes', convergence_tracker='No', loc=2, **kwargs):
    '''Models a laterally loaded pile using the p-y method. The solution for lateral displacements 
    is obtained by solving the 4th order ODE, EI*d4y/dz4 -F*d2y/dz2 + ky = 0 using the finite difference method.
    
    Assumes that EI remains constant with respect to curvature i.e. pile material remains in the elastic region.
    
    Input:
    -----
    soil_profile - A 2D array of depths (in), corresponding undrained shear strength(psf), 
                   and corresponding vertical effective stress (psf)
                   Eg: array([[z1,Su1,sigma_v_eff1],[z2,Su2,sigma_v_eff2],[z3,Su3,sigma_v_eff3]...])
                   Use small values for Su and sigma_v_eff (eg: 0.001) instead of zeros to avoid divisions 
                   by zero but always start z at 0.0.
                   Example of a valid data point at the mudline is [0.0, 0.001, 0.001]
                  
    L           - Length of pile         (inches)
    D           - Outer diameter of pile (inches)
    t           - Wall thickness of pile (inches)
    E           - Elastic modulus of pile material (psi)
    y_0         - Displacement at the pile head (in)
    F           - Axial force at pile head (lb), vertically downwards is postive.
    M_0         - Moment at pile head (lb-in), moments causing tension on left side of pile is positive.
    V_n         - Force at pile head/tip  (lb),  shear causing clockwise rotation of pile is positive.
    M_n         - Moment at pile tip (lb-in), moments causing tension on left side of pile is positive.
    n           - Number of elements (50 by default)
    iterations  - Number of iterations to repeat calculation in order obtain convergence of 'y'
                  (A better approach is to iterate until a predefined tolerance is achieved but this requires additional
                  coding so, I will implement this later.)
    py_model    - Select which p-y model to use, 'Matlock', 'Jeanjean', 'Modified Matlock', or 'Kodikara'.
    print_output - Print results on screen, Choose 'Yes' or 'No' (default = 'No')
    **kwargs    - Specify parameters for different p-y models
    
    Optional:
    convergence_tracker - Track how k_secant converges to actual p-y curve at a selected node
    loc                 - Node number at which k_secant to be tracked (2 to n+1)
    
    Output:
    ------
    y           - Lateral displacement at each node, length = n + 5, (n+1) real nodes and 4 imaginary nodes
    l           - Vector of node locations along pile
    X           - Finite difference matrix 
    '''
    
    from scipy import linalg
    
    #Convert L and D to floating point numbers to avoid rounding errors
    L = float(L)
    D = float(D)
    
    #Pile geometry
    I  = pi*(D**4 - (D-2*t)**4)/64.0
    EI = E*I
    h  = L/n  #Element size
    N  = (n+1)+4      # (n+1) Real + 4 Imaginary nodes 
    
    #Array for displacements at nodes, including imaginary nodes.
    y = ones(N)*(0.01*D)   #An initial value of 0.01D was arbitrarily chosen
    
    #Initialize and assemble array/list of p-y curves at each real node
    z = zeros(N)
    py_funs  = []
    k_secant = zeros(N)
    
    for i in [0,1]:        #Top two imaginary nodes
        z[i] = (i-2)*h
        py_funs.append(0)
        k_secant[i] = 0.0
    
    #Extract soil profile data
    z_0, f_Su, f_sigma_v_eff = design_soil_profile(soil_profile)

    #Extract p-y model parameters from **kwargs
    eps_50 = 0.02
    A = 550
    
    if kwargs is not None:
        for key, value in kwargs.iteritems():
            if key=='epsilon_50': eps_50 = value
            elif key=='A': A=value
            elif key=='R1': R1=0.5
    
    
    for i in range(2,n+3): #Real nodes
        z[i] = (i-2)*h
        
        Su, sigma_v_eff = f_Su(z[i]), f_sigma_v_eff(z[i])
        
        if py_model=='Matlock':
            py_funs.append(matlock_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=eps_50, print_curves='No'))
        elif py_model=='Jeanjean':
            py_funs.append(jeanjean_py_curves(z[i], D,Su, sigma_v_eff, z_0=z_0, Su_0=f_Su(z_0), A=A))
        elif py_model=='Modified Matlock':
            py_funs.append(modified_matlock_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=eps_50, print_curves='No'))
        elif py_model=='Kodikara':
            py_funs.append(kodikara_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, R1=R1, A=A, print_curves='No'))
        else:
            print "P-y model not properly defined. Please select one of the following:"
            print "'Matlock', 'Jeanjean', 'Modified Matlock', 'Kodikara'"
        
        k_secant[i]     = py_funs[i](y[i])/y[i]
    
    for i in [n+3, n+4]:   #Bottom two imaginary nodes
        z[i] = (i-2)*h
        py_funs.append(0)
        k_secant[i] = 0.0
    
    #Track k_secant and current displacements
    if convergence_tracker=='Yes':
        y1 = linspace(-2.*D,2.*D,500) 
        plot(y1, py_funs[loc](y1))
        xlabel('y (in)'), ylabel('p (lb/in)'), grid(True)
        
    for j in range(iterations):
        #if j==0: print 'FD Solver started!'
            
        y,V_0 = fd_solver_2(n,N,h,EI,F,y_0,V_n,M_0,M_n,k_secant)
        
        if convergence_tracker=='Yes':
            plot(y[loc], k_secant[loc]*y[loc], 'x')
            
        for i in range(2, n+3):
            k_secant[i] = py_funs[i](y[i])/y[i]
    
    if print_output=='Yes':
        print 'V_0 = %.2f-lb' %V_0
    
    return y,z,V_0


def py_analysis_3(Su_profile, L=10.0, D=0.375, t = 0.1, E=29e6, F = 0.0, V_0=0.0, V_n=0.0, M_0=0.0, M_n=0.0, n=50, 
                  iterations=15, convergence_tracker='No', loc=2):
    '''Models a laterally loaded pile using the p-y method. The solution for lateral displacements 
    is obtained by solving the 4th order ODE, EI*d4y/dz4 -F*d2y/dz2 + ky + D*dt/dz = 0 
    using the finite difference method. The extra term "D*dt/dz" accounts for the side shear
    resistance on the pile. 
    
    Assumptions:
    - EI remains constant with respect to curvature i.e. pile material remains in the elastic region.
    - Side shear force per unit length 't' is fully mobilized and equal to D*Su (i.e. alpha = 1.0).
    - The assumption that alpha = 1.0 is valid only for NC clay. 
    
    Input:
    -----
    Su_profile  - A 2D array of depths (in) and corresponding undrained shear strength(psf)
                  Eg: array([[z1,Su1],[z2,Su2],[z3,Su3]...])
                  Use small values for Su (eg: 0.001) instead of zeros to avoid divisions by zero but always start z at 0.0
                  Example of a valid data point at the mudline is [0.0, 0.001]
                  
    L           - Length of pile         (inches)
    D           - Outer diameter of pile (inches)
    t           - Wall thickness of pile (inches)
    E           - Elastic modulus of pile material (psi)
    F           - Axial force at pile head (lb), vertically downwards is postive.
    V_0, V_n    - Force at pile head/tip  (lb),  shear causing clockwise rotation of pile is positive.
    M_0, M_n    - Moment at pile head/tip (lb-in), moments causing tension on left side of pile is positive.
    n           - Number of elements (50 by default)
    iterations  - Number of iterations to repeat calculation in order obtain convergence of 'y'
                  (A better approach is to iterate until a predefined tolerance is achieved but this requires additional
                  coding so, I will implement this later.)
    
    Optional:
    convergence_tracker - Track how k_secant converges to actual p-y curve at a selected node
    loc                 - Node number at which k_secant to be tracked (2 to n+1)
    
    Output:
    ------
    y           - Lateral displacement at each node, length = n + 5, (n+1) real nodes and 4 imaginary nodes
    l           - Vector of node locations along pile
    X           - Finite difference matrix 
    '''
    
    from scipy import linalg
    
    #Pile geometry
    I  = pi*(D**4 - (D-2*t)**4)/64.0
    EI = E*I
    h  = L/n  #Element size
    N  = (n+1)+4      # (n+1) Real + 4 Imaginary nodes 
    
    #Array for displacement including imaginary nodes
    y = ones(N)*(0.01*D)   #An initial value of 0.01D was arbitrarily chosen
    
    #Initialize and assemble array/list of p-y curves at each real node
    z = zeros(N)
    Su = zeros(N)
    
    py_funs  = []
    k_secant = zeros(N)
    
    tz_funs  = []
    c_secant = zeros(N) #secant stiffness of t-z curves
    
    for i in [0,1]:        #Top two imaginary nodes
        z[i] = (i-2)*h
        py_funs.append(0)
        tz_funs.append(0)
    
    for i in range(2,n+3): #Real nodes
        z[i] = (i-2)*h
        Su[i], sigma_v_eff = design_soil_profile(z[i], Su_profile)
        
        py_funs.append(matlock_py_curves(z[i], D, Su[i], sigma_v_eff, epsilon_50=0.02, print_curves='No'))
        k_secant[i]     = py_funs[i](y[i])/y[i]
        
        tz_funs.append(tz_curve(Su[i], D, z_0=0.1))
                       
    for i in [n+3, n+4]:   #Bottom two imaginary nodes
        z[i] = (i-2)*h
        py_funs.append(0)
        tz_funs.append(0)
        
    
    #Track k_secant and current displacements
    if convergence_tracker=='Yes':
        figure()
        y1 = linspace(-0.5*D,0.5*D,500) 
        plot(y1, py_funs[loc](y1))
        xlabel('y (in)'), ylabel('p (lb/in)'), grid(True)
        
    for j in range(iterations):
        
        y = fd_solver_3(n,N,h,EI,F,V_0,V_n,M_0,M_n,k_secant,D,c_secant)
        
        if convergence_tracker=='Yes':
            plot(y[loc], k_secant[loc]*y[loc], 'x')
            
        for i in range(2, n+3):
            k_secant[i] = py_funs[i](y[i])/y[i]
            
            delta_z = 0.5*D*(y[i+1] - y[i-1])/(2*h)
            c_secant[i] = tz_funs[i](delta_z)/delta_z            
            
    return y,z



def py_fe_1(soil_profile, L=10.0, D=2.0, t=0.1, E=29e6, F=0.0, y_0=0.0, V_n=0.0, M_0=0.0, M_n=0.0, n=100, 
                  iterations=10, py_model='Matlock', print_output='Yes', convergence_tracker='No', loc=2, **kwargs):
    '''Models a laterally loaded pile using the p-y method. The solution for lateral displacements 
    is obtained by solving the 4th order ODE, EI*d4y/dz4 + ky = 0 using the finite element method.

    ***Axial forces are neglected in the current form of this code. "fe_solver_1" will have to be upgraded
    with frame elements in order to account for axial loads.
    
    Assumes that EI remains constant with respect to curvature i.e. pile material remains in the elastic region.

    Uses 'fe_solver_1()'.
    
    Input:
    -----
    soil_profile - A 2D array of depths (in), corresponding undrained shear strength(psf), 
                   and corresponding vertical effective stress (psf)
                   Eg: array([[z1,Su1,sigma_v_eff1],[z2,Su2,sigma_v_eff2],[z3,Su3,sigma_v_eff3]...])
                   Use small values for Su and sigma_v_eff (eg: 0.001) instead of zeros to avoid divisions 
                   by zero but always start z at 0.0.
                   Example of a valid data point at the mudline is [0.0, 0.001, 0.001]
                  
    L           - Length of pile         (inches)
    D           - Outer diameter of pile (inches)
    t           - Wall thickness of pile (inches)
    E           - Elastic modulus of pile material (psi)
    y_0         - Displacement at pile head (inches)
    F           - Axial force at pile head (lb), vertically downwards is postive.
    V_n         - Force at pile head/tip  (lb),  shear causing clockwise rotation of pile is positive.
    M_0, M_n    - Moment at pile head/tip (lb-in), moments causing tension on left side of pile is positive.
    n           - Number of elements (100 by default)
    iterations  - Number of iterations to repeat calculation in order obtain convergence of 'y'
                  (A better approach is to iterate until a predefined tolerance is achieved but this requires additional
                  coding so, I will implement this later.)
    py_model    - Select which p-y model to use, 'Matlock', 'Jeanjean', 'Modified Matlock', or 'Kodikara'.
                  The following parameters in the p-y models can be changed using **kwargs:
                  'epsilon_50' in the Matlock and Modified Matlock models.
                  'Gmax_Su_ratio' in the Jeanjean and Kodikara models.
                  'R1' in the Kodikara model (R1 is the ration between undrained tensile and shear strengths in a clay)
    print_output - Print results on screen, Choose 'Yes' or 'No' (default = 'No')
    
    Optional:
    convergence_tracker - Track how k_secant converges to actual p-y curve at a selected node
    loc                 - Node number at which k_secant to be tracked (0 to n+1)

    Optional keywords: **kwargs
    epsilon_50  - Define 'epsilon_50' if 'Matlock' p-y model is selected. (epsilon_50=0.02 by default)
    Gmax_Su_ratio - Define G_max/Su if Jeanjean or Kodikara p-y models are chosen. Default values are:
                     Jeanjean (2009) -> Gmax_Su_ratio = 550
                     Kodikara (2009) -> Gmax_Su_ratio = 250
    
    Output:
    ------
    y           - Lateral displacement at each node, length of array = (n+1)
    l           - Vector of node locations along pile
    V_0         - Lateral load at pile head (lb)
    '''
    
    from scipy import linalg

    #Extract optional keyword arguments
    epsilon_50, A = 0.02, 550 #Default parameters if no **kwargs are defined
    for arg in kwargs:
        if arg=='epsilon_50':
            epsilon_50 = kwargs[arg]        
        if arg=='Gmax_Su_ratio':
            A = kwargs[arg]
    
    #Convert L and D to floating point numbers to avoid rounding errors
    L = float(L)
    D = float(D)
    
    #Pile geometry
    I  = pi*(D**4 - (D-2*t)**4)/64.0 * ones(n)
    E  = E*ones(n)
    h  = L/n   #Element size
    N  = (n+1) #Number of nodes
    
    #Array for displacement including imaginary nodes
    y = ones(N)*(0.01*D)   #An initial value of 0.01D was arbitrarily chosen
    
    #Initialize and assemble array/list of p-y curves at each real node
    z = zeros(N)
    py_funs  = []
    k_secant = zeros(N)
    
    #Extract soil profile data
    z_0, f_Su, f_sigma_v_eff = design_soil_profile(soil_profile)
    
    for i in range(0,N):
        z[i] = i*h
        
        Su, sigma_v_eff = f_Su(z[i]), f_sigma_v_eff(z[i])
        
        if py_model=='Matlock':
            py_funs.append(matlock_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=epsilon_50, print_curves='No'))
        elif py_model=='Jeanjean':
            py_funs.append(jeanjean_py_curves(z[i], D,Su, sigma_v_eff, z_0=z_0, Su_0=f_Su(z_0), A=A))
        elif py_model=='Modified Matlock':
            py_funs.append(modified_matlock_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=epsilon_50, print_curves='No'))
        elif py_model=='Kodikara':
            py_funs.append(kodikara_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, R1=0.5, A=A, print_curves='No'))
        else:
            print "P-y model not properly defined. Please select one of the following:"
            print "'Matlock', 'Jeanjean', 'Modified Matlock', 'Kodikara'"
        
        k_secant[i]     = py_funs[i](y[i])/y[i]
    
    #Track k_secant and current displacements
    if convergence_tracker=='Yes':
        y1 = linspace(-2.*D,2.*D,500) 
        plot(y1, py_funs[loc](y1))
        xlabel('y (in)'), ylabel('p (lb/in)'), grid(True)
        
    for j in range(iterations):
        #if j==0: print 'FD Solver started!'
            
        y,slope,V_0 = fe_solver_2(n,h,E,I,y_0,V_n,M_0,M_n,k_secant)
        
        if convergence_tracker=='Yes':
            plot(y[loc], k_secant[loc]*y[loc], 'x')
            
        for i in range(0,N):
            k_secant[i] = py_funs[i](y[i])/y[i]
    
    if print_output=='Yes':
        print 'V_0 = %.2f-lb' %V_0
    
    return y,z,V_0


def py_fe_2(soil_profile, L=10.0, D=2.0, t=0.1, E=29e6, F=0.0, y_0=0.0, V_n=0.0, M_0=0.0, M_n=0.0, n=100, 
                  iterations=10, py_model='Matlock', print_output='Yes', convergence_tracker='No', loc=2):
    '''Models a laterally loaded pile using the p-y method. The solution for lateral displacements 
    is obtained by solving the 4th order ODE, EI*d4y/dz4 + ky = 0 using the finite element method.

    ***Axial forces are neglected in the current form of this code. "fe_solver_2" will have to be upgraded
    with frame elements in order to account for axial loads.
    
    Assumes that EI remains constant with respect to curvature i.e. pile material remains in the elastic region.

    Uses 'fe_solver_2()'.
    
    Input:
    -----
    soil_profile - A 2D array of depths (in), corresponding undrained shear strength(psf), 
                   and corresponding vertical effective stress (psf)
                   Eg: array([[z1,Su1,sigma_v_eff1],[z2,Su2,sigma_v_eff2],[z3,Su3,sigma_v_eff3]...])
                   Use small values for Su and sigma_v_eff (eg: 0.001) instead of zeros to avoid divisions 
                   by zero but always start z at 0.0.
                   Example of a valid data point at the mudline is [0.0, 0.001, 0.001]
                  
    L           - Length of pile         (inches)
    D           - Outer diameter of pile (inches)
    t           - Wall thickness of pile (inches)
    E           - Elastic modulus of pile material (psi)
    y_0         - Displacement at pile head (inches)
    F           - Axial force at pile head (lb), vertically downwards is postive.
    V_n         - Force at pile head/tip  (lb),  shear causing clockwise rotation of pile is positive.
    M_0, M_n    - Moment at pile head/tip (lb-in), moments causing tension on left side of pile is positive.
    n           - Number of elements (100 by default)
    iterations  - Number of iterations to repeat calculation in order obtain convergence of 'y'
                  (A better approach is to iterate until a predefined tolerance is achieved but this requires additional
                  coding so, I will implement this later.)
    py_model    - Select which p-y model to use, 'Matlock', 'Jeanjean', 'Modified Matlock', or 'Kodikara'.
    print_output - Print results on screen, Choose 'Yes' or 'No' (default = 'No')
    
    Optional:
    convergence_tracker - Track how k_secant converges to actual p-y curve at a selected node
    loc                 - Node number at which k_secant to be tracked (0 to n+1)
    
    Output:
    ------
    y           - Lateral displacement at each node, length of array = (n+1)
    l           - Vector of node locations along pile
    V_0         - Lateral load at pile head (lb)
    '''
    
    from scipy import linalg
    
    #Convert L and D to floating point numbers to avoid rounding errors
    L = float(L)
    D = float(D)
    
    #Pile geometry
    I  = pi*(D**4 - (D-2*t)**4)/64.0 * ones(n)
    E  = E*ones(n)
    h  = L/n   #Element size
    N  = (n+1) #Number of nodes
    
    #Array for displacement
    y = ones(N)*(0.01*D)   #An initial value of 0.01D was arbitrarily chosen
    
    #Initialize and assemble array/list of p-y curves at each real node
    z = zeros(N)
    py_funs  = []
    k_secant = zeros(N)
    
    #Extract soil profile data
    z_0, f_Su, f_sigma_v_eff = design_soil_profile(soil_profile)
    
    for i in range(0,N):
        z[i] = i*h
        
        Su, sigma_v_eff = f_Su(z[i]), f_sigma_v_eff(z[i])
        
        if py_model=='Matlock':
            py_funs.append(matlock_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=0.02, print_curves='No'))
        elif py_model=='Jeanjean':
            py_funs.append(jeanjean_py_curves(z[i], D,Su, sigma_v_eff, z_0=z_0, Su_0=f_Su(z_0), A=550))
        elif py_model=='Modified Matlock':
            py_funs.append(modified_matlock_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=0.02, print_curves='No'))
        elif py_model=='Kodikara':
            py_funs.append(kodikara_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, R1=0.5, A=50, print_curves='No'))
        else:
            print "P-y model not properly defined. Please select one of the following:"
            print "'Matlock', 'Jeanjean', 'Modified Matlock', 'Kodikara'"
        
        k_secant[i]     = py_funs[i](y[i])/y[i]
    
    #Track k_secant and current displacements
    if convergence_tracker=='Yes':
        y1 = linspace(-2.*D,2.*D,500) 
        plot(y1, py_funs[loc](y1))
        xlabel('y (in)'), ylabel('p (lb/in)'), grid(True)
        
    for j in range(iterations):
        #if j==0: print 'FD Solver started!'
            
        y,slope,V_0 = fe_solver_1(n,h,E,I,y_0,V_n,M_0,M_n,k_secant)
        
        if convergence_tracker=='Yes':
            plot(y[loc], k_secant[loc]*y[loc], 'x')
            
        for i in range(0,N):
            k_secant[i] = py_funs[i](y[i])/y[i]
    
    if print_output=='Yes':
        print 'V_0 = %.2f-lb' %V_0
    
    return y,z,V_0


def py_fe_dyn_1(soil_profile, L=52.0, D=4.0, t=0.125, E=29e6, rho=0.283, n=100, iterations=1, py_model='Matlock', 
              y_1=0.01, print_output='Yes', convergence_tracker='No', loc=2, modes=[1,2]):
    '''Models a laterally loaded pile using the p-y method. The solution for lateral displacements 
    is obtained by solving the 4th order ODE, EI*d4y/dz4 + ky = 0 using the finite element method and calculates
    the natural frequencies of that pile when embedded in the specified soil profile.

    This function can only model the pile. Use py_fe_dyn_2() to model a wind turbine tower together with the pile
    foundation. It can also accommodate tapering cross sections and provides more control over the selection of the
    secant stiffness of the p-y curves.
    
    Assumes that EI remains constant with respect to curvature i.e. pile material remains in the elastic region.
    
    Input:
    -----
    soil_profile - A 2D array of depths (in), corresponding undrained shear strength(psf), 
                   and corresponding vertical effective stress (psf)
                   Eg: array([[z1,Su1,sigma_v_eff1],[z2,Su2,sigma_v_eff2],[z3,Su3,sigma_v_eff3]...])
                   Use small values for Su and sigma_v_eff (eg: 0.001) instead of zeros to avoid divisions 
                   by zero but always start z at 0.0.
                   Example of a valid data point at the mudline is [0.0, 0.001, 0.001]          
    L           - Length of pile         (inches)
    D           - Outer diameter of pile (inches)
    t           - Wall thickness of pile (inches)
    E           - Elastic modulus of pile material (psi)
    rho         - Mass density of pile material (pci)
    n           - Number of elements (100 by default)
    y_1         - Displacement at first discretization point in pile diameters (Default=0.01D)
    iterations  - Number of iterations to repeat calculation in order obtain convergence of 'y'. Only a single
                  iteration is required for the dynamic analysis, in its current form.
    py_model    - Select which p-y model to use, 'Matlock', 'Jeanjean', 'Modified Matlock', or 'Kodikara'.
    print_output - Print results on screen, Choose 'Yes' or 'No' (default = 'No')
    modes - List of the modes whose results that should be printed
    
    Optional:
    convergence_tracker - Track how k_secant converges to actual p-y curve at a selected node
    loc                 - Node number at which k_secant to be tracked (0 to n+1)
    
    Output:
    ------
    w         - Vector with all eigen values i.e square of natural frequencies. 
                eg: Mode 1 frequency = sqrt(w[-1]) (unit = rad/s)
    u         - Matrix with all eigen vectors i.e.vibration modes. 
                eg: Mode 1 shape = u[:,-1]
                Includes both displacements and rotations at each node.
    
    Prints natural frequencies and periods for specified modes.
    '''
    
    from scipy import linalg
    
    #Convert L and D to floating point numbers to avoid rounding errors
    L = float(L)
    D = float(D)
    
    #Pile geometry
    I  = pi*(D**4 - (D-2*t)**4)/64.0 * ones(n)
    A  = pi*(D**2 - (D-2*t)**2)/4.0  * ones(n)
    E  = E*ones(n)
    rho = rho*ones(n)
    h  = L/n   #Element size
    N  = (n+1) #Number of nodes
    
    #Array for displacement
    y = ones(N)*(y_1*D)   #An initial value of 0.01D was arbitrarily chosen. 
                           #This initial estimate governs the k_secant values and in turn the 
                           #dynamic behavior of the whole structure. Proper estimation of this 
                           #value is critical!
    
    #Initialize and assemble array/list of p-y curves at each real node
    z = zeros(N)
    py_funs  = []
    k_secant = zeros(N)
    
    #Extract soil profile data
    z_0, f_Su, f_sigma_v_eff = design_soil_profile(soil_profile)
    
    for i in range(0,N):
        z[i] = i*h
        
        Su, sigma_v_eff = f_Su(z[i]), f_sigma_v_eff(z[i])
        
        if py_model=='Matlock':
            py_funs.append(matlock_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=0.02, print_curves='No'))
        elif py_model=='Jeanjean':
            py_funs.append(jeanjean_py_curves(z[i], D,Su, sigma_v_eff, z_0=z_0, Su_0=f_Su(z_0), A=550))
        elif py_model=='Modified Matlock':
            py_funs.append(modified_matlock_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=0.02, print_curves='No'))
        elif py_model=='Kodikara':
            py_funs.append(kodikara_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, R1=0.5, A=50, print_curves='No'))
        else:
            print "P-y model not properly defined. Please select one of the following:"
            print "'Matlock', 'Jeanjean', 'Modified Matlock', 'Kodikara'"
        
        k_secant[i]     = py_funs[i](y[i])/y[i]
    
    #Track k_secant and current displacements
    if convergence_tracker=='Yes':
        y1 = linspace(-2.*D,2.*D,500) 
        plot(y1, py_funs[loc](y1))
        xlabel('y (in)'), ylabel('p (lb/in)'), grid(True)
        
    w, vr = fe_solver_dyn(n,h,E,I,A,rho,k_secant)
    
    u = zeros(N)
    
    if print_output=='Yes':
        for i in modes:
            omega = w[-i]**0.5

            print 'Mode %2d:  Frequency = %8.3f rad/sec = %8.3f Hz, Period = %8.4f s' %(i, omega, omega/(2*pi), 2*pi/omega)
            for j in range(0,len(vr[:,-i])/2):
                u[j] = vr[2*j,-i]

            plot(u/u[-1],z), grid(True)

    #figure()
    #plot(k_secant,z)
    
    return w, vr


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
    
    #Initialize and assemble matrix
    X = zeros((N,N))
    
    #(n+1) finite difference equations for (n+1) real nodes
    for i in range(0,n+1):
        X[i,i]   =  1.0
        X[i,i+1] = -4.0 + F*h**2/EI
        X[i,i+2] =  6.0 - 2*F*h**2/EI + k_secant[i+2]*h**4/EI
        X[i,i+3] = -4.0 + F*h**2/EI
        X[i,i+4] =  1.0
    
    #Curvature at pile head
    X[n+1,1]   =  1.0
    X[n+1,2]   = -2.0
    X[n+1,3]   =  1.0
    
    #Shear at pile head
    X[n+2,0]   = -1.0
    X[n+2,1]   =  2.0 - F*h**2/EI
    X[n+2,2]   =  0.0
    X[n+2,3]   = -2.0 + F*h**2/EI
    X[n+2,4]   =  1.0
    
    #Curvature at pile tip
    X[n+3,-2]   =  1.0
    X[n+3,-3]   = -2.0
    X[n+3,-4]   =  1.0
    
    #Shear at pile tip
    X[n+4,-1]   =   1.0
    X[n+4,-2]   =  -2.0 + F*h**2/EI
    X[n+4,-3]   =   0.0
    X[n+4,-4]   =   2.0 - F*h**2/EI
    X[n+4,-5]   =  -1.0
    
    #X*y = q
    
    #Initialize vector q
    q = zeros(N)  
    
    #Populate q with boundary conditions
    q[-1] = 2*V_n*h**3     #Shear at pile tip
    q[-2] = M_n*h**2       #Moment at pile tip
    q[-3] = 2*V_0*h**3     #Shear at pile head
    q[-4] = M_0*h**2       #Moment at pile head
    
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
    
    #Initialize and assemble matrix
    X = zeros((N,N))
    
    #(n+1) finite difference equations for (n+1) real nodes
    for i in range(0,n+1):
        X[i,i]   =  1.0
        X[i,i+1] = -4.0 + F*h**2/EI
        X[i,i+2] =  6.0 - 2*F*h**2/EI + k_secant[i+2]*h**4/EI
        X[i,i+3] = -4.0 + F*h**2/EI
        X[i,i+4] =  1.0
    
    #Curvature at pile head
    X[n+1,1]   =  1.0
    X[n+1,2]   = -2.0
    X[n+1,3]   =  1.0
    
    #Shear at pile head
    X[n+2,0]   = -1.0
    X[n+2,1]   =  2.0 - F*h**2/EI
    X[n+2,2]   =  0.0
    X[n+2,3]   = -2.0 + F*h**2/EI
    X[n+2,4]   =  1.0
    
    #Curvature at pile tip
    X[n+3,-2]  =  1.0
    X[n+3,-3]  = -2.0
    X[n+3,-4]  =  1.0
    
    #Shear at pile tip
    X[n+4,-1]  =   1.0
    X[n+4,-2]  =  -2.0 + F*h**2/EI
    X[n+4,-3]  =   0.0
    X[n+4,-4]  =   2.0 - F*h**2/EI
    X[n+4,-5]  =  -1.0
    
    #Repartition X since y_0 is specified. 
    
    #print 'X \n', X
    
    X1 = zeros((N,N))
    X1[:,:] = X[:,:]
    
    X1[:,2]  = zeros(N)
    X1[-3,2] = -1.0/EI
       
    #X*y = q
    #Initialize vector q
    q = zeros(N)
    
    #Apply essential boundary condition i.e. y_0
    q[0:-4] = -X[0:-4,2]*EI*y_0
    
    #Populate q with natural boundary conditions
    q[-1] = 2*V_n*h**3 -X[-1,2]*EI*y_0     #Shear at pile tip
    q[-2] = M_n*h**2   -X[-2,2]*EI*y_0       #Moment at pile tip
    q[-3] =            -X[-3,2]*EI*y_0
    q[-4] = M_0*h**2   -X[-4,2]*EI*y_0      #Moment at pile head
    
    #print '\n X1 \n', X1
    
    #print '\n X[:,2] \n', X[:,2]
    #print '\n q \n', q
    
    y1 = linalg.solve(EI*X1,q)
    
    V_0 = y1[2]/(2*h**3)
    
    y = zeros(N)
    y[:] = y1[:]
    y[2] = y_0
    
    return y, V_0


def fd_solver_21(n,N,h,EI,F,y_0,V_n,s_0,M_n,k_secant):
    '''Solves the finite difference equations from 'py_analysis_2'. This function should be run iteratively for
    non-linear p-y curves by updating 'k_secant' using 'y'. A single iteration is sufficient if the p-y curves
    are linear.
    
    Specified boundary conditions: Displacement and slope at pile head. Shear and moment at pile tip. 
                                   Axial force at pile head.
    
    Input:
    -----
    n        - Number of elements
    N        - Total number of nodes
    h        - Element size
    EI       - Flexural rigidity of pile
    F        - Axial force at pile head
    y_0      - Displacement at pile head
    s_0      - Slope at the pile head
    V_n      - Shear at pile tip
    M_n      - Moment at pile head/tip
    k_secant - Secant stiffness from p-y curves
    
    Output:
    ------
    y_updated - Lateral displacement at each node
    '''
    
    #Initialize and assemble matrix
    X = zeros((N,N))
    
    #(n+1) finite difference equations for (n+1) real nodes
    for i in range(0,n+1):
        X[i,i]   =  1.0
        X[i,i+1] = -4.0 + F*h**2/EI
        X[i,i+2] =  6.0 - 2*F*h**2/EI + k_secant[i+2]*h**4/EI
        X[i,i+3] = -4.0 + F*h**2/EI
        X[i,i+4] =  1.0
    
    #Slope at pile head
    X[n+1,1]   = -1.0
    X[n+1,3]   =  1.0
    
    #Shear at pile head
    X[n+2,0]   = -1.0
    X[n+2,1]   =  2.0 - F*h**2/EI
    X[n+2,2]   =  0.0
    X[n+2,3]   = -2.0 + F*h**2/EI
    X[n+2,4]   =  1.0
    
    #Curvature at pile tip
    X[n+3,-2]  =  1.0
    X[n+3,-3]  = -2.0
    X[n+3,-4]  =  1.0
    
    #Shear at pile tip
    X[n+4,-1]  =   1.0
    X[n+4,-2]  =  -2.0 + F*h**2/EI
    X[n+4,-3]  =   0.0
    X[n+4,-4]  =   2.0 - F*h**2/EI
    X[n+4,-5]  =  -1.0
    
    #Repartition X since y_0 is specified. 
    
    #print 'X \n', X
    
    X1 = zeros((N,N))
    X1[:,:] = X[:,:]
    
    X1[:,2]  = zeros(N)
    X1[-3,2] = -1.0/EI
       
    #X*y = q
    #Initialize vector q
    q = zeros(N)
    
    #Apply essential boundary condition i.e. y_0
    q[0:-4] = -X[0:-4,2]*EI*y_0
    
    #Populate q with natural boundary conditions
    q[-1] = 2*V_n*h**3 -X[-1,2]*EI*y_0     #Shear at pile tip
    q[-2] = M_n*h**2   -X[-2,2]*EI*y_0     #Moment at pile tip
    q[-3] =            -X[-3,2]*EI*y_0
    q[-4] = s_0*2*h*EI   #-X[-4,2]*EI*y_0     #Moment at pile head
    
    #print '\n X1 \n', X1
    
    #print '\n X[:,2] \n', X[:,2]
    #print '\n q \n', q
    
    y1 = linalg.solve(EI*X1,q)
    
    V_0 = y1[2]/(2*h**3)
    
    y = zeros(N)
    y[:] = y1[:]
    y[2] = y_0
    
    return y, V_0


def fd_solver_22(n,N,h,EI,F,y_0,y_n,M_0,M_n,k_secant):
    '''Solves the finite difference equations from 'py_analysis_22'. This function should be run iteratively for
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
    
    #Initialize and assemble matrix
    X = zeros((N,N))
    
    #(n+1) finite difference equations for (n+1) real nodes
    for i in range(0,n+1):
        X[i,i]   =  1.0
        X[i,i+1] = -4.0 + F*h**2/EI
        X[i,i+2] =  6.0 - 2*F*h**2/EI + k_secant[i+2]*h**4/EI
        X[i,i+3] = -4.0 + F*h**2/EI
        X[i,i+4] =  1.0
    
    #Curvature at pile head
    X[n+1,1]   =  1.0
    X[n+1,2]   = -2.0
    X[n+1,3]   =  1.0
    
    #Shear at pile head
    X[n+2,0]   = -1.0
    X[n+2,1]   =  2.0 - F*h**2/EI
    X[n+2,2]   =  0.0
    X[n+2,3]   = -2.0 + F*h**2/EI
    X[n+2,4]   =  1.0
    
    #Curvature at pile tip
    X[n+3,-2]  =  1.0
    X[n+3,-3]  = -2.0
    X[n+3,-4]  =  1.0
    
    #Shear at pile tip
    X[n+4,-1]  =   1.0
    X[n+4,-2]  =  -2.0 + F*h**2/EI
    X[n+4,-3]  =   0.0
    X[n+4,-4]  =   2.0 - F*h**2/EI
    X[n+4,-5]  =  -1.0
    
    #Repartition X since y_0 is specified. 
    
    #print 'X \n', X
    
    X1 = zeros((N,N))
    X1[:,:] = X[:,:]
    
    X1[:,2]  = zeros(N)
    X1[:,-3] = zeros(N)
    X1[-3,2]  = -1.0/EI
    X1[-1,-3] = -1.0/EI
       
    #X*y = q
    #Initialize vector q
    q = zeros(N)
    
    #Apply essential boundary conditions i.e. y_0, y_n
    q[0:-4] = -X[0:-4,2]*EI*y_0 - X[0:-4,-3]*EI*y_n
    
    #Populate q with natural boundary conditions
    q[-1] =            -X[-1,2]*EI*y_0 -X[-1,-3]*EI*y_n    #Shear at pile tip
    q[-2] = M_n*h**2   -X[-2,2]*EI*y_0 -X[-2,-3]*EI*y_n    #Moment at pile tip
    q[-3] =            -X[-3,2]*EI*y_0 -X[-3,-3]*EI*y_n
    q[-4] = M_0*h**2   -X[-4,2]*EI*y_0 -X[-4,-3]*EI*y_n     #Moment at pile head
    
    #print '\n X1 \n', X1
    
    #print '\n X[:,2] \n', X[:,2]
    #print '\n q \n', q
    
    y1 = linalg.solve(EI*X1,q)
    
    V_0 = y1[2]/(2*h**3)
    V_n = y1[-3]/(2*h**3)
    
    y = zeros(N)
    y[:]  = y1[:]
    y[2]  = y_0
    y[-3] = y_n
    
    return y, V_0

def fd_solver_3(n,N,h,EI,F,V_0,V_n,M_0,M_n,k_secant,D,c_secant):
    '''Solves the finite difference equations from 'py_analysis_3'. This function should be run iteratively for
    non-linear p-y curves by updating 'k_secant' using 'y'. A single iteration is sufficient if the p-y curves
    are linear. 
    
    Specified boundary conditions: Shear and moment at pile head and pile tip. Axial force at pile head.
    
    Input:
    -----
    n - Number of elements
    N - Total number of nodes
    h - Element size
    EI - Flexural rigidity of pile (lb-in2)
    F  - Axial force at pile head (lb)
    V_0, V_n - Shear at pile head/tip (lb)
    M_0, M_n - Moment at pile head/tip (lb-in)
    k_secant - Secant stiffness from p-y curves (lb/in/in)
    D - Pile diameter (in)
    tau - 1-D array of shear strength (Su) at pile nodes
    
    Output:
    ------
    y_updated - Lateral displacement at each node
    '''
    
    #Initialize and assemble matrix
    X = zeros((N,N))
    
    #(n+1) finite difference equations for (n+1) real nodes
    for i in range(0,n+1):
        X[i,i]   =  1.0 - c_secant[i+2]*(D*h)**2/(8*EI)
        X[i,i+1] = -4.0 +   F*h**2/EI
        X[i,i+2] =  6.0 - 2*F*h**2/EI + k_secant[i+2]*h**4/EI + 2*c_secant[i+2]*(D*h)**2/(8*EI)
        X[i,i+3] = -4.0 +   F*h**2/EI 
        X[i,i+4] =  1.0 - c_secant[i+2]*(D*h)**2/(8*EI)
    
    #Curvature at pile head
    X[n+1,1]   =  1.0
    X[n+1,2]   = -2.0
    X[n+1,3]   =  1.0
    
    #Shear at pile head
    X[n+2,0]   = -1.0
    X[n+2,1]   =  2.0 - F*h**2/EI + c_secant[2]*(D*h)**2/(2*EI)
    X[n+2,2]   =  0.0
    X[n+2,3]   = -2.0 + F*h**2/EI - c_secant[2]*(D*h)**2/(2*EI)
    X[n+2,4]   =  1.0
    
    #Curvature at pile tip
    X[n+3,-2]  =  1.0
    X[n+3,-3]  = -2.0
    X[n+3,-4]  =  1.0
    
    #Shear at pile tip
    X[n+4,-1]  =   1.0
    X[n+4,-2]  =  -2.0 + F*h**2/EI - c_secant[-3]*(D*h)**2/(2*EI)
    X[n+4,-3]  =   0.0
    X[n+4,-4]  =   2.0 - F*h**2/EI + c_secant[-3]*(D*h)**2/(2*EI)
    X[n+4,-5]  =  -1.0
    
    #X*y = q
    
    #Initialize vector q
    q = zeros(N)  
    
    #Populate q with boundary conditions
    q[-1] = 2*V_n*h**3     #Shear at pile tip
    q[-2] = M_n*h**2       #Moment at pile tip
    q[-3] = 2*V_0*h**3     #Shear at pile head
    q[-4] = M_0*h**2       #Moment at pile head
    
    y = linalg.solve(EI*X,q)
    
    return y


def fe_solver_1(n,l,E,I,y_0,V_n,M_0,M_n,k_secant):
    '''Solves the finite element model from 'py_fe_1' (using the penalty method for boundary conditionos.
    This function should be run iteratively for non-linear p-y curves by updating 'k_secant' using 'y'.
    A single iteration is sufficient if the p-y curves are linear.
    
    Specified boundary conditions: Displacement and moment at pile head. Shear and moment at pile tip. 
                                   Axial force at pile head.

    Note:
    The penalty method is used to apply the specified displacement at the pile head and then the resulting
    stiffness matrix is solved. In contrast, 'fe_solver_2' uses the direct elimination method to solve a
    reduced matrix.
    

    Input:
    -----
    n        - Number of elements
    l        - Length of element
    E        - Elastic modulus of material (lb/in^2)
    I        - Second moment of area (in^4)
    y_0      - Displacement at pile head (in)
    V_n      - Shear at pile tip (lb)
    M_0, M_n - Moment at pile head/tip (lb-in)
    k_secant - Vector of secant stiffness from p-y curves for each element (lb/in^2)
    
    Output:
    ------
    u         - Displacement vector
    slope     - Slope vector
    R         - Reactions at fixed dofs
    '''
    
    from scipy import linalg
    
    #FE Model
    N       = n+1               #Number of nodes
    l       = l*ones(n)    #Element lengths (m)

    #External loads
    f = 0*ones(2*N)   #Nodal actions (point loads on even dofs and moments on odd dofs)
    p = 0*ones(n)         #Distributed load
    k = k_secant               #Winkler spring stiffness, F/L^2
    
    #f[0]       = V_0
    f[1]       = M_0
    f[2*N - 2] = V_n
    f[2*N - 1] = M_n

    #Initiate global stiffness matrix and force vectors
    K = zeros((2*N, 2*N))
    F = zeros(2*N)

    #Assemble global stiffness matrix and force vectors
    for i in range(0,n):
        #Elemental stiffness matrix
        E_e = E[i]
        l_e = l[i]
        I_e = I[i]
        p_e = p[i]

        #Element stiffness
        K_elem = E_e*I_e/(l_e**3) * array([[   12,      6*l_e,   -12,      6*l_e],
                                           [6*l_e, 4*(l_e**2),-6*l_e, 2*(l_e**2)],
                                           [  -12,     -6*l_e,    12,     -6*l_e],
                                           [6*l_e, 2*(l_e**2),-6*l_e, 4*(l_e**2)]])

        #Winkler support stiffness
        K_s = k[i]*l_e/420. * array([[    156,      22*l_e,     54,    -13*l_e],
                                     [ 22*l_e,  4*(l_e**2), 13*l_e,-3*(l_e**2)],
                                     [     54,      13*l_e,    156,    -22*l_e],
                                     [-13*l_e, -3*(l_e**2),-22*l_e, 4*(l_e**2)]])

        #Element force and moment vector
        F_elem = p_e*l_e * array([[  1./2.],
                                  [l_e/12.],
                                  [  1./2.],
                                  [-l_e/12.]])

        #Global stiffness matrix and force vector
        for j in range(0,4):
            K[2*i+j,2*i]   += K_elem[j,0] + K_s[j,0]
            K[2*i+j,2*i+1] += K_elem[j,1] + K_s[j,1]
            K[2*i+j,2*i+2] += K_elem[j,2] + K_s[j,2]
            K[2*i+j,2*i+3] += K_elem[j,3] + K_s[j,3]

            F[2*i+j] += F_elem[j]
            
        #Add nodal loads and moments to the global force vector
        F[2*i]   += f[2*i]
        F[2*i+1] += f[2*i+1]
    
    #Add loads and moments acting on the last node (missed by previous loop) to the global force vector
    F[2*N - 2] += f[2*N - 2]
    F[2*N - 1] += f[2*N - 1]

    #Apply boundary conditions (using the penalty method) and solve equations
    boundary_dof     = array([0])
    boundary_dof_val = array([y_0])

    C = 1e6*E[0]*I[0] #Assumed spring stiffness at support for penalty method

    for i in range(len(boundary_dof)):
        loc         = boundary_dof[i]
        K[loc,loc] += C
        F[loc]     += C*boundary_dof_val[i]

    q     = linalg.solve(K,F)
    
    #Reactions
    R = zeros(len(boundary_dof))
    for i in range(len(boundary_dof)):
        loc = boundary_dof[i]
        R[i] = -C*(q[loc]-boundary_dof_val[i])

    u     = zeros(int(len(q)/2))
    slope = zeros(int(len(q)/2))

    counter = 0
    for i in range(0,len(q),2):
        u[counter]     = q[i]
        slope[counter] = q[i+1]
        counter += 1
    
    #print 'Pile Head: Deflection = %e \t Slope =%e ' %(u[0], slope[0])
    #print 'Pile Tip: Deflection = %e \t Slope =%e ' %(u[N-1], slope[N-1])
    #print 'Reactions: V=%f' %(R)
    
    V_0 = R[0]
    
    return u,slope, V_0


def fe_solver_2(n,l,E,I,y_0,V_n,M_0,M_n,k_secant):
    '''Solves the finite element model from 'py_fe_2' using the elimination method for boundary condidition.
    This function should be run iteratively for non-linear p-y curves by updating 'k_secant' using 'y'.
    A single iteration is sufficient if the p-y curves are linear.
    
    Specified boundary conditions: Displacement and moment at pile head. Shear and moment at pile tip. 
                                   Axial force at pile head.

    Note:
    The reduced stiffness matrix is derived based on specified displacement at the pile head and then solved.
    In contrast, 'fe_solver_1' used the penalty method.
    

    Input:
    -----
    n        - Number of elements
    l        - Length of element
    E        - Elastic modulus of material (lb/in^2)
    I        - Second moment of area (in^4)
    y_0      - Displacement at pile head (in)
    V_n      - Shear at pile tip (lb)
    M_0, M_n - Moment at pile head/tip (lb-in)
    k_secant - Vector of secant stiffness from p-y curves for each element (lb/in^2)
    
    Output:
    ------
    u         - Displacement vector
    slope     - Slope vector
    R         - Reactions at fixed dofs
    '''
    
    from scipy import linalg
    
    #FE Model
    N       = n+1               #Number of nodes
    l       = l*ones(n)    #Element lengths (m)

    #External loads
    f = 0*ones(2*N)   #Nodal actions (point loads on even dofs and moments on odd dofs)
    p = 0*ones(n)         #Distributed load
    k = k_secant               #Winkler spring stiffness, F/L^2
    
    #f[0]       = V_0
    f[1]       = M_0
    f[2*N - 2] = V_n
    f[2*N - 1] = M_n

    #Initiate global stiffness matrix and force vectors
    K = zeros((2*N, 2*N))
    F = zeros(2*N)

    #Assemble global stiffness matrix and force vectors
    for i in range(0,n):
        #Elemental stiffness matrix
        E_e = E[i]
        l_e = l[i]
        I_e = I[i]
        p_e = p[i]

        #Element stiffness
        K_elem = E_e*I_e/(l_e**3) * array([[   12,      6*l_e,   -12,      6*l_e],
                                           [6*l_e, 4*(l_e**2),-6*l_e, 2*(l_e**2)],
                                           [  -12,     -6*l_e,    12,     -6*l_e],
                                           [6*l_e, 2*(l_e**2),-6*l_e, 4*(l_e**2)]])

        #Winkler support stiffness
        K_s = k[i]*l_e/420. * array([[    156,      22*l_e,     54,    -13*l_e],
                                     [ 22*l_e,  4*(l_e**2), 13*l_e,-3*(l_e**2)],
                                     [     54,      13*l_e,    156,    -22*l_e],
                                     [-13*l_e, -3*(l_e**2),-22*l_e, 4*(l_e**2)]])

        #Element force and moment vector
        F_elem = p_e*l_e * array([[  1./2.],
                                  [l_e/12.],
                                  [  1./2.],
                                  [-l_e/12.]])

        #Global stiffness matrix and force vector
        for j in range(0,4):
            K[2*i+j,2*i]   += K_elem[j,0] + K_s[j,0]
            K[2*i+j,2*i+1] += K_elem[j,1] + K_s[j,1]
            K[2*i+j,2*i+2] += K_elem[j,2] + K_s[j,2]
            K[2*i+j,2*i+3] += K_elem[j,3] + K_s[j,3]

            F[2*i+j] += F_elem[j]
            
        #Add nodal loads and moments to the global force vector
        F[2*i]   += f[2*i]
        F[2*i+1] += f[2*i+1]
    
    #Add loads and moments acting on the last node (missed by previous loop) to the global force vector
    F[2*N - 2] += f[2*N - 2]
    F[2*N - 1] += f[2*N - 1]

    #Apply boundary conditions (using the elimination method) and solve equations
    K1 = zeros((2*N-1,2*N-1))
    F1 = zeros(2*N-1)
    
    K1[:,:] = K[1:,1:]
    F1[:]   = F[1:] -K[1:,0]*y_0

    q1     = linalg.solve(K1,F1)

    u     = zeros(N)
    slope = zeros(N)
    
    u[0] = y_0

    for i in range(0,N):
        slope[i] = q1[2*i]
        
    for i in range(1,N):
        u[i] = q1[2*i-1]
        
    V_0 = K[0,0]*y_0 + K[0,1:].dot(q1) - F[0]
    
    #print 'Pile Head: Deflection = %e \t Slope =%e ' %(u[0], slope[0])
    #print 'Pile Tip: Deflection = %e \t Slope =%e ' %(u[N-1], slope[N-1])
    #print 'Reactions: V=%f' %(V_0)
    
    #plot(u,'-o')
    
    return u,slope, V_0


def fe_solver_dyn_1(n,l,E,I,A,rho,k_secant):
    '''Solves the finite element model from 'py_fe_dyn' (using the penalty method for boundary conditions).
    The calculated natural frequencies are based on the k_secant values estimated from the input p-y curves. 
    Therefore, the k_secant values should be carefully selected. The values of k_secant calculated based on 
    p-y curves at each node is dependent on the initial assumed displacement.
    
    Specified boundary conditions: Displacement and moment at pile head. Shear and moment at pile tip. 
                                   Axial force at pile head.

    'fe_solver_dyn_2()' can be used to define the fixity at the base (i.e. node 0) with the keyword argument
    'base_fixity'. This is the only difference between 'fe_solver_dyn_2()' and 'fe_solver_dyn_1()'
    

    Input:
    -----
    n        - Number of elements
    l        - Length of element
    E        - Elastic modulus of material (lb/in^2)
    I        - Second moment of area (in^4)
    A        - Cross sectional area (in^2)
    rho      - Mass density (lb-mass/in^3)
    k_secant - Vector of secant stiffness from p-y curves for each element (lb/in^2)
    
    Output:
    ------
    vr        - Matrix with all eigen vectors i.e.vibration modes. 
                eg: Mode 1 shape = vr[:,-1]
    w         - Vector with all eigen values i.e square of natural frequencies. 
                eg: Mode 1 frequency = sqrt(w[-1]) (unit = rad/s)
    '''
    
    from scipy import linalg
    
    #FE Model
    N       = n+1          #Number of nodes
    l       = l*ones(n)    #Element lengths (m)

    #External loads
    f = 0*ones(2*N)   #Nodal actions (point loads on even dofs and moments on odd dofs)
    p = 0*ones(n)         #Distributed load
    k = k_secant               #Winkler spring stiffness, F/L^2

    #Initiate global stiffness matrix and force vectors
    K = zeros((2*N, 2*N))
    M = zeros((2*N, 2*N))
    F = zeros(2*N)

    #Assemble global stiffness matrix and force vectors
    for i in range(0,n):
        #Elemental stiffness matrix
        E_e = E[i]
        l_e = l[i]
        A_e = A[i]
        I_e = I[i]
        p_e = p[i]

        #Element stiffness
        K_elem = E_e*I_e/(l_e**3) * array([[   12,      6*l_e,   -12,      6*l_e],
                                           [6*l_e, 4*(l_e**2),-6*l_e, 2*(l_e**2)],
                                           [  -12,     -6*l_e,    12,     -6*l_e],
                                           [6*l_e, 2*(l_e**2),-6*l_e, 4*(l_e**2)]])

        #Winkler support stiffness
        K_s = k[i]*l_e/420. * array([[    156,      22*l_e,     54,    -13*l_e],
                                     [ 22*l_e,  4*(l_e**2), 13*l_e,-3*(l_e**2)],
                                     [     54,      13*l_e,    156,    -22*l_e],
                                     [-13*l_e, -3*(l_e**2),-22*l_e, 4*(l_e**2)]])

        #Element force and moment vector
        F_elem = p_e*l_e * array([[  1./2.],
                                  [l_e/12.],
                                  [  1./2.],
                                  [-l_e/12.]])

        #Element mass matrix
        M_elem = rho[i]*A_e*l_e/420. * array([[    156,      22*l_e,     54,    -13*l_e],
                                              [ 22*l_e,  4*(l_e**2), 13*l_e,-3*(l_e**2)],
                                              [     54,      13*l_e,    156,    -22*l_e],
                                              [-13*l_e, -3*(l_e**2),-22*l_e, 4*(l_e**2)]])

        for j1 in range(0,4):
            for j2 in range(0,4):
                K[2*i+j1,2*i+j2]   += K_elem[j1,j2] + K_s[j1,j2]
                M[2*i+j1,2*i+j2]   += M_elem[j1,j2]

            F[2*i+j1] += F_elem[j1]
            
        #Add nodal loads and moments to the global force vector
        F[2*i]   += f[2*i]
        F[2*i+1] += f[2*i+1]
    
    #Add loads and moments acting on the last node (missed by previous loop) to the global force vector
    F[2*N - 2] += f[2*N - 2]
    F[2*N - 1] += f[2*N - 1]

    #Apply boundary conditions (using the penalty method) and solve equations.
    #No natural boundary conditions are specified for the dynamic analysis
    boundary_dof     = array([])
    boundary_dof_val = array([])

    C = 1e6*E[0]*I[0] #Assumed spring stiffness at support for penalty method

    for i in range(len(boundary_dof)):
        loc         = boundary_dof[i]
        K[loc,loc] += C
        F[loc]     += C*boundary_dof_val[i]

    w, vr = linalg.eig(K,M)
        
    return w, vr


def fe_solver_dyn_2(n,l,E,I,A,rho,k_secant, base_fixity='Free'):
    '''Solves the finite element model from 'py_fe_dyn_1()', 'py_fe_dyn_2()', 'dyn_lat_tower_1()', and
    'dyn_lat_tower_2()'. The boundary conditions are defined using the penalty method.
    The calculated natural frequencies are based on the k_secant values estimated from
    the input p-y curves. Therefore, the k_secant values should be carefully selected. The values of k_secant
    calculated based on p-y curves at each node is dependent on the initial assumed displacement.
    
    Specified boundary conditions: Displacement and moment at pile head. Shear and moment at pile tip. 
                                   Axial force at pile head.

    The fixity at the base (i.e. node 0) can be controlled by the keyword argument 'base_fixity'. This is the
    only difference between 'fe_solver_dyn_2()' and 'fe_solver_dyn_1()'.
    

    Input:
    -----
    n        - Number of elements
    l        - 1-d array of element lengths (in)
    E        - 1-d array of elastic modulus of material in each element (lb/in^2)
    I        - 1-d array of second moment of area of each element (in^4)
    A        - 1-d array of cross sectional area of each element (in^2)
    rho      - 1-d array of mass density of each element (lb-mass/in^3)
    k_secant - Vector of secant stiffness from p-y curves for each element (lb/in^2)
    base_fixity - Boundary condition at the base (i.e. node 0). 
                  Options: 'Free' (default), 'Fixed', 'Pinned'
    
    Output:
    ------
    w         - Vector with all eigen values i.e square of natural frequencies. 
                eg: Mode 1 frequency = sqrt(w[-1]) (unit = rad/s)
    vr        - Matrix with all eigen vectors i.e.vibration modes. 
                eg: Mode 1 shape = vr[:,-1]
    '''
    
    from scipy import linalg
    
    #FE Model
    N       = n+1          #Number of nodes

    #External loads
    f = 0*ones(2*N)   #Nodal actions (point loads on even dofs and moments on odd dofs)
    p = 0*ones(n)     #Distributed load
    k = k_secant      #Winkler spring stiffness, FL^-2

    #Initiate global stiffness matrix, mass matrix, and force vectors
    K = zeros((2*N, 2*N))
    M = zeros((2*N, 2*N))
    F = zeros(2*N)

    #Assemble global stiffness matrix and force vectors
    for i in range(0,n):
        #Elemental stiffness matrix
        E_e = E[i]
        l_e = l[i]
        A_e = A[i]
        I_e = I[i]
        p_e = p[i]

        #Element stiffness
        K_elem = E_e*I_e/(l_e**3) * array([[   12,      6*l_e,   -12,      6*l_e],
                                           [6*l_e, 4*(l_e**2),-6*l_e, 2*(l_e**2)],
                                           [  -12,     -6*l_e,    12,     -6*l_e],
                                           [6*l_e, 2*(l_e**2),-6*l_e, 4*(l_e**2)]])

        #Winkler support stiffness
        K_s = k[i]*l_e/420. * array([[    156,      22*l_e,     54,    -13*l_e],
                                     [ 22*l_e,  4*(l_e**2), 13*l_e,-3*(l_e**2)],
                                     [     54,      13*l_e,    156,    -22*l_e],
                                     [-13*l_e, -3*(l_e**2),-22*l_e, 4*(l_e**2)]])

        #Element force and moment vector
        F_elem = p_e*l_e * array([[  1./2.],
                                  [l_e/12.],
                                  [  1./2.],
                                  [-l_e/12.]])

        #Element mass matrix
        M_elem = rho[i]*A_e*l_e/420. * array([[    156,      22*l_e,     54,    -13*l_e],
                                              [ 22*l_e,  4*(l_e**2), 13*l_e,-3*(l_e**2)],
                                              [     54,      13*l_e,    156,    -22*l_e],
                                              [-13*l_e, -3*(l_e**2),-22*l_e, 4*(l_e**2)]])

        for j1 in range(0,4):
            for j2 in range(0,4):
                K[2*i+j1,2*i+j2]   += K_elem[j1,j2] + K_s[j1,j2]
                M[2*i+j1,2*i+j2]   += M_elem[j1,j2]

            F[2*i+j1] += F_elem[j1]
            
        #Add nodal loads and moments to the global force vector
        F[2*i]   += f[2*i]
        F[2*i+1] += f[2*i+1]
    
    #Add loads and moments acting on the last node (missed by previous loop) to the global force vector
    F[2*N - 2] += f[2*N - 2]
    F[2*N - 1] += f[2*N - 1]

    #Apply boundary conditions (using the penalty method) and solve equations.
    #No natural boundary conditions are specified for the dynamic analysis
    
    if base_fixity =='Fixed':
        boundary_dof     = array([0,1])
        boundary_dof_val = array([0.,0.])
    elif base_fixity =='Pinned':
        boundary_dof     = array([0])
        boundary_dof_val = array([0.])
    else: #base_fixity=='Free'
        boundary_dof     = array([])
        boundary_dof_val = array([])

    C = 1e6*E[0]*I[0] #Assumed spring stiffness at support for penalty method

    for i in range(len(boundary_dof)):
        loc         = boundary_dof[i]
        K[loc,loc] += C
        F[loc]     += C*boundary_dof_val[i]
    
    #Calculate eigen values based on K and M matrices
    w, vr = linalg.eig(K,M)
        
    return w, vr


###############################
#### P-Y Curve Definitions ####
###############################

def matlock_py_curves(z, D, Su, sigma_v_eff, z_0=0.0, epsilon_50=0.02, loading_type='static', print_curves='No',
                     return_Np='No',ls='-'):
    '''Returns an interp1d interpolation function which represents the Matlock (1970) p-y curve at the depth of interest.
    
    Important: Make sure to import the interp1 function by running 'from scipy.interpolate import interp1d' in the main program.

    Input:
    -----
    z            - Depth relative to pile head (in)
    D            - Pile diameter (in)
    Su           - Undrained shear strength (psf)
    sigma_v_eff  - Effectve vertical stress (psf)
    z_0         - Load eccentricity above the mudline or depth to mudline relative to the pile head (in)
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
    'p' (lb/in) and 'y' (in).
    '''
    
    from scipy.interpolate import interp1d
    
    #Convert to psi
    Su = Su/144.
    sigma_v_eff = sigma_v_eff/144.
    
    #p-y curve properties
    J     = 0.5
    
    if (z-z_0)<=0:
        #p-y curves for the virtual soil layer between the pile head and the mudline should have p=0
        N_p  = 0.0
        z_cr = 1.0 #Dummy value to keep program from crashing
    
    else:
        try:
            N_p   = 3.0 + sigma_v_eff/Su + J*(z-z_0)/D 

            if N_p > 9.0: N_p = 9.0

            z_cr  = (6.0 - sigma_v_eff/Su)*D/J  #This condition is implemented to avoid zero division errors.

        except ZeroDivisionError:
            print "Division by zero! Su = 0.0 so z_cr cannot be calculated."
    
    p_ult = Su*N_p*D
    y_50  = 2.5*epsilon_50*D
    
    #Normalized lateral displacement
    Y = concatenate((-logspace(3,-4,100),[0],logspace(-4,3,100)))
    
    #Normalized depths
    Z    = z/D
    Z_cr = z_cr/D
    
    #Normalized p-y curves
    P = 0.5*sign(Y)*abs(Y)**(1.0/3.0)  #sign(Y) and abs(Y) used since negative numbers cannot be raised to fractional powers
                                           #Expression equivalent to P = 0.5*Y**(1.0/3.0) for Y>=0
    for i in range(0,len(Y)): 
        if P[i] > 1.0:    P[i] = 1.0
        elif P[i] < -1.0: P[i] = -1.0
    
    if loading_type=='cyclic':
        
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
            
    
    #Un-normallized p-y curves
    p = P*p_ult
    y = Y*y_50
    
    f = interp1d(y,p, kind='linear')   #Interpolation function for p-y curve
    
    #Secant stiffness
    #k = f(y1)/y1
    
    if print_curves=='Yes':
        #Plot of p-y curve and check if 'k' is calculated correctly
        plot(y,p,ls), xlabel('y (in)'), ylabel('p (lb/in)')
        grid(True)
        #plot(y1, k*y1, 'x')
    
    if return_Np == 'Yes':
        return f, N_p
    else:
        return f

def matlock_py_curves_no_gapping(z, D, Su, sigma_v_eff, z_0=0.0, epsilon_50=0.02, loading_type='static', print_curves='No',ls='-'):
    '''Returns an interp1d interpolation function which represents the Matlock (1970) p-y curve at the depth of interest.
    
    Important: Make sure to import the interp1 function by running 'from scipy.interpolate import interp1d' in the main program.

    Input:
    -----
    z            - Depth relative to pile head (in)
    D            - Pile diameter (in)
    Su           - Undrained shear strength (psf)
    sigma_v_eff  - Effectve vertical stress (psf)
    z_0         - Load eccentricity above the mudline or depth to mudline relative to the pile head (in)
    epsilon_50   - Strain at half the strength as defined by Matlock (1970).
                   Typically ranges from 0.005 (stiff clay) to 0.02 (soft clay).
    loading_type - Either 'static' or 'cyclic'
    
    Output:
    ------
    Returns an interp1d interpolation function which represents the p-y curve at the depth of interest. 
    'p' (lb/in) and 'y' (in).
    '''
    
    from scipy.interpolate import interp1d
    
    #Convert to psi
    Su = Su/144.
    sigma_v_eff = sigma_v_eff/144.
    
    #p-y curve properties
    J     = 1.0  #0.5x2 to account for the active soil wedge
    
    if (z-z_0)<=0:
        #p-y curves for the virtual soil layer between the pile head and the mudline should have p=0
        N_p  = 0.0
        z_cr = 1.0 #Dummy value to keep program from crashing
    
    else:
        try:
            N_p   = 4.0 + J*(z-z_0)/D 

            if N_p > 9.0: N_p = 9.0

            z_cr  = (6.0 - sigma_v_eff/Su)*D/J  #This condition is implemented to avoid zero division errors.

        except ZeroDivisionError:
            print "Division by zero! Su = 0.0 so z_cr cannot be calculated."
    
    p_ult = Su*N_p*D
    y_50  = 2.5*epsilon_50*D
    
    #Normalized lateral displacement
    Y = concatenate((-logspace(3,-4,100),[0],logspace(-4,3,100)))
    
    #Normalized depths
    Z    = z/D
    Z_cr = z_cr/D
    
    #Normalized p-y curves
    P = 0.5*sign(Y)*abs(Y)**(1.0/3.0)  #sign(Y) and abs(Y) used since negative numbers cannot be raised to fractional powers
                                           #Expression equivalent to P = 0.5*Y**(1.0/3.0) for Y>=0
    for i in range(0,len(Y)): 
        if P[i] > 1.0:    P[i] = 1.0
        elif P[i] < -1.0: P[i] = -1.0
    
    if loading_type=='cyclic':
        
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
            
    
    #Un-normallized p-y curves
    p = P*p_ult
    y = Y*y_50
    
    f = interp1d(y,p, kind='linear')   #Interpolation function for p-y curve
    
    #Secant stiffness
    #k = f(y1)/y1
    
    if print_curves=='Yes':
        #Plot of p-y curve and check if 'k' is calculated correctly
        plot(y,p,ls), xlabel('y (in)'), ylabel('p (lb/in)')
        grid(True)
        #plot(y1, k*y1, 'x')
    
    return f

def MM_1_py_curves(z, D, Su, sigma_v_eff, z_0=0.0, Su0=0.0, epsilon_50=0.02, 
                           gapping='No', alpha = 1.0, print_curves='No',ls='-', return_Np='No'):
    '''Returns an interp1d interpolation function which represents the Matlock (1970) p-y curve at the depth of interest.
    
    Important: Make sure to import the interp1 function by running 'from scipy.interpolate import interp1d' in the main program.

    Input:
    -----
    z            - Depth relative to pile head (in)
    D            - Pile diameter (in)
    Su           - Undrained shear strength at depth 'z' (psf)
    Su0          - Undrained shear strength at depth 'z0' (psf)
                   Note: If setting Su0 based on the 'interp1d' function for Su from 'design_soil_profile' then, 
                   it is safer to set Su0=f_Su(z0+0.01) rather than Su0=f_Su(z0) since f_Su(z0) could be zero 
                   and lead to numerical instabilities in the code. 'py_analysis_2()' uses Su0=f_Su(z0+0.01) by default.
    sigma_v_eff  - Effectve vertical stress (psf)
    z_0          - Load eccentricity above the mudline or depth to mudline relative to the pile head (in)
    epsilon_50   - Strain at half the strength as defined by Matlock (1970).
                   Typically ranges from 0.005 (stiff clay) to 0.02 (soft clay).
    gapping      - 'Yes' -> N_p = 2.0 + gamma*z/Su + (Su0 + Su)/Su * sqrt(2)*(z/D)
                   'No'  -> N_p = 4.0              + (Su0 + Su)/Su * 2*sqrt(2)*(z/D)
    alpha        - Coefficient of pile-soil interface adhesion. 1.0 for a rough pile and 0.0 for a smooth pile.
                   The maximum value of the lateral bearing capacity factor will be calculated as follows:
                   N_p_max = 9 + 3*alpha
                   Use option 'API' to calculate using the API method (Randolph & Murphy, 1985).

    Optional argument:
    return_Np    - Returns the Np values that in addtion to the p-y curve. This option was added to check how well visualize the
                   effect and depth of the gap and should only be used when this function is used by itself.
                   DO NOT set it to 'Yes' for p-y analysis as the program will crash!
    
    Output:
    ------
    Returns an interp1d interpolation function which represents the p-y curve at the depth of interest. 
    'p' (lb/in) and 'y' (in).
    '''
    
    from scipy.interpolate import interp1d
    
    #Convert to psi
    Su  = Su/144.
    Su0 = Su0/144.
    sigma_v_eff = sigma_v_eff/144.
    
    psi = Su/sigma_v_eff
    
    #Calculate alpha based on the API method
    if alpha=='API':
        if psi<1.0:
            alpha = min(0.5*psi**(-0.5),1.0)
        elif psi>1.0:
            alpha = min(0.5*psi**(-0.25),1.0)
        elif z<z_0:
            alpha=0.0 #Assign default value to alpha above the mudline to avoid numerical errors.
        else:
            print 'psi = %2.2f' %psi
            raise Exception('Failed to calculate alpha based on API method!')
    
    #Calculate N_p_max based on pile-soil interface adhesion factor
    N_p_max = 9.0 + 3.0*alpha
    
    #p-y curve properties
    if gapping=='No':
        N_p0 = 4.0*sqrt(3**alpha)
        N_p1 = 0.0
        J    = (Su0 + Su)/Su * 2*sqrt(2)
    else:
        N_p0 = 2.0*sqrt(3**alpha)
        N_p1 = sigma_v_eff/Su
        J    = (Su0 + Su)/Su * sqrt(2)
    
    if (z-z_0)<=0:
        #p-y curves for the virtual soil layer between the pile head and the mudline should have p=0
        N_p  = 0.0
        z_cr = 1.0 #Dummy value to keep program from crashing
    
    else:
        try:
            N_p   = N_p0 + N_p1 + J*(z-z_0)/D 
            
            if N_p > N_p_max: N_p = N_p_max

            z_cr  = (6.0 - sigma_v_eff/Su)*D/J  #This condition is implemented to avoid zero division errors.

        except ZeroDivisionError:
            print "Division by zero! Su = 0.0 so z_cr cannot be calculated."
    
    if epsilon_50=='Auto' and Su!=0:
        #epsilon_50 = min(0.02, 0.004 + 0.0032*(sigma_v_eff/Su))
        
        P_a = 14.7 #psi, atmospheric pressure
        psi = Su/(P_a + sigma_v_eff) #Where (P_a + sigma_v_eff) is the confining stress
        epsilon_50 = -0.0318*psi**0.109 + 0.0395 #This relationship was obtained by fitting y = a*x**b + c
                                                #to Su and epsilon_50 data from Reese et al (1975)

        if epsilon_50 > 0.02:
            epsilon_50 = 0.02
        elif epsilon_50 < 0.004:
            epsilon_50 = 0.004

    elif epsilon_50=='Auto' and Su==0:
        epsilon_50=0.02
        
    p_ult = Su*N_p*D
    y_50  = 2.5*epsilon_50*D
    
    #Normalized lateral displacement
    Y = concatenate((-logspace(3,-4,100),[0],logspace(-4,3,100)))
    
    #Normalized depths
    Z    = z/D
    Z_cr = z_cr/D
    
    #Normalized p-y curves
    P = 0.5*sign(Y)*abs(Y)**(1.0/3.0)  #sign(Y) and abs(Y) used since negative numbers cannot be raised to fractional powers
                                           #Expression equivalent to P = 0.5*Y**(1.0/3.0) for Y>=0
    for i in range(0,len(Y)): 
        if P[i] > 1.0:    P[i] = 1.0
        elif P[i] < -1.0: P[i] = -1.0
            
    #Un-normallized p-y curves
    p = P*p_ult
    y = Y*y_50
    
    f = interp1d(y,p, kind='linear')   #Interpolation function for p-y curve
    
    if print_curves=='Yes':
        #Plot of p-y curve and check if 'k' is calculated correctly
        plot(y,p,ls)
        xlabel('y (in)'), ylabel('p (lb/in)')
        grid(True)

    if return_Np == 'Yes':
        return f, N_p
    else:
        return f

def MM_2_py_curves(z, D, Su, sigma_v_eff, z_0=0.0, Su0=0.0, epsilon_50=0.02, 
                           gapping='No', N_p_max = 12.0, print_curves='No',ls='-'):
    '''Returns an interp1d interpolation function which represents the Matlock (1970) p-y curve at the depth of interest.
    
    Important: Make sure to import the interp1 function by running 'from scipy.interpolate import interp1d' in the main program.

    Input:
    -----
    z            - Depth relative to pile head (in)
    D            - Pile diameter (in)
    Su           - Undrained shear strength at depth 'z' (psf)
    Su0          - Undrained shear strength at depth 'z0' (psf)
                   Note: If setting Su0 based on the 'interp1d' function for Su from 'design_soil_profile' then, 
                   it is safer to set Su0=f_Su(z0+0.01) rather than Su0=f_Su(z0) since f_Su(z0) could be zero 
                   and lead to numerical instabilities in the code.
    sigma_v_eff  - Effectve vertical stress (psf)
    z_0          - Load eccentricity above the mudline or depth to mudline relative to the pile head (in)
    epsilon_50   - Strain at half the strength as defined by Matlock (1970).
                   Typically ranges from 0.005 (stiff clay) to 0.02 (soft clay).
    gapping      - 'Yes' -> N_p = 2.0 + gamma*z/Su + (Su0 + Su)/Su * sqrt(2)*(z/D)
                   'No'  -> N_p = 4.0              + (Su0 + Su)/Su * 2*sqrt(2)*(z/D)
    N_p_max      - Equals to 9.0 for a smooth pile and 12.0 for a rought pile based on Randolph & Houlsby (1984)
    
    Output:
    ------
    Returns an interp1d interpolation function which represents the p-y curve at the depth of interest. 
    'p' (lb/in) and 'y' (in).
    '''
    
    from scipy.interpolate import interp1d
    
    #Convert to psi
    Su  = Su/144.
    Su0 = Su0/144.
    sigma_v_eff = sigma_v_eff/144.
    
    #p-y curve properties
    if gapping=='No':
        N_p0 = 4.0
        N_p1 = 0.0
        J    = (Su0 + Su)/Su * 2*sqrt(2)
    else:
        N_p0 = 2.0
        N_p1 = sigma_v_eff/Su
        J    = (Su0 + Su)/Su * sqrt(2)
        
    #Cavity expansion gapping
    P_a = 2120. #Atmospheric pressure, psf
    N_p_cavity1  = 7.0 + sigma_v_eff/Su
    N_p_cavity2  = 7.0 + sigma_v_eff/Su + 62.4/Su + P_a/Su
    N_p_cavity   = min(N_p_cavity1, N_p_cavity2)
    
    if (z-z_0)<=0:
        #p-y curves for the virtual soil layer between the pile head and the mudline should have p=0
        N_p  = 0.0
        z_cr = 1.0 #Dummy value to keep program from crashing
    
    else:
        try:
            N_p   = N_p0 + N_p1 + J*(z-z_0)/D 
            
            if N_p > min(N_p_max,N_p_cavity): N_p = min(N_p_max, N_p_cavity)

            z_cr  = (6.0 - sigma_v_eff/Su)*D/J  #This condition is implemented to avoid zero division errors.

        except ZeroDivisionError:
            print "Division by zero! Su = 0.0 so z_cr cannot be calculated."
    
    
    p_ult = Su*N_p*D
    y_50  = 2.5*epsilon_50*D
    
    #Normalized lateral displacement
    Y = concatenate((-logspace(3,-4,100),[0],logspace(-4,3,100)))
    
    #Normalized depths
    Z    = z/D
    Z_cr = z_cr/D
    
    #Normalized p-y curves
    P = 0.5*sign(Y)*abs(Y)**(1.0/3.0)  #sign(Y) and abs(Y) used since negative numbers cannot be raised to fractional powers
                                           #Expression equivalent to P = 0.5*Y**(1.0/3.0) for Y>=0
    for i in range(0,len(Y)): 
        if P[i] > 1.0:    P[i] = 1.0
        elif P[i] < -1.0: P[i] = -1.0
            
    #Un-normallized p-y curves
    p = P*p_ult
    y = Y*y_50
    
    f = interp1d(y,p, kind='linear')   #Interpolation function for p-y curve
    
    if print_curves=='Yes':
        #Plot of p-y curve and check if 'k' is calculated correctly
        plot(y,p,ls)
        xlabel('y (in)'), ylabel('p (lb/in)')
        grid(True)
    
    return f


def jeanjean_py_curves(z,D, Su, sigma_v_eff, Su_0=0.0, z_0=0.0, A=550, print_curves='No', return_Np='No',ls='-'):
    '''
    Returns an interp1d interpolation function which represents the Jeanjean (2009) p-y curve at the depth of interest.
    
    Important: Make sure to import the interp1 function by running 'from scipy.interpolate import interp1d' in the main program.

    Input:
    -----
    z            - Depth relative to pile head (in)
    D            - Pile diameter (in)
    Su           - Undrained shear strength (psf)
    sigma_v_eff  - Effectve vertical stress (psf)
    Su_0         - Undrained shear strength at the mudline (psf)
    z_0          - Load eccentricity above the mudline or depth to mudline relative to the pile head (in)
    A            - G_max/Su (default = 550)
    
    Optional argument:
    return_Np    - Returns the Np values that in addtion to the p-y curve. This option was added to check how well visualize the
                   effect and depth of the gap and should only be used when this function is used by itself.
                   DO NOT set it to 'Yes' for p-y analysis as the program will crash!
    
    Output:
    ------
    Returns an interp1d interpolation function which represents the p-y curve at the depth of interest. 
    'p' (lb/in) and 'y' (in).
    '''
    
    from scipy.interpolate import interp1d
    
    #Change units to psi
    Su = Su/144.0  
    Su_0 = Su_0/144.0
    sigma_v_eff = sigma_v_eff/144.
    G_max = A*Su
    
    #Normalized p-y curve
    Y = linspace(-3,3,1000) 
    P = tanh(A/100.0*abs(Y)**(0.5))*sign(Y)
    
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

        N_p = 12 - 4*exp(-xi*(z-z_0)/D)
    
    #Un-normalize p-y curves
    p_ult = N_p*Su*D
    
    p = P*p_ult
    y = Y*D
    
    f = interp1d(y,p, kind='linear')   #Interpolation function for p-y curve
    
    #Print curves
    if print_curves=='Yes':
        plot(y,p,ls)
        xlabel('y (in)'), ylabel('p (lb/in)')
        grid(True)
        
    if return_Np == 'Yes':
        return f, N_p
    else:
        return f


def modified_matlock_py_curves(z, D, Su, sigma_v_eff, z_0=0.0, epsilon_50=0.02, loading_type='static', print_curves='No'):
    '''Returns an interp1d interpolation function which represents the Matlock (1970) with the range for Np updated from 
    3 - 9 to 8 - 12 as proposed by Jeanjean (2009). These p-y curves have been named 'Modified Matlock' curves.
    
    Important: Make sure to import the interp1 function by running 'from scipy.interpolate import interp1d' in the main program.

    Input:
    -----
    z            - Depth relative to pile head (in)
    D            - Pile diameter (in)
    Su           - Undrained shear strength (psf)
    sigma_v_eff  - Effectve vertical stress (psf)
    z_0          - Load eccentricity above the mudline or depth to mudline relative to the pile head (in)
    epsilon_50   - Strain at half the strength as defined by Matlock (1970).
                   Typically ranges from 0.005 (stiff clay) to 0.02 (soft clay).
    loading_type - Either 'static' or 'cyclic'
    
    Output:
    ------
    Returns an interp1d interpolation function which represents the p-y curve at the depth of interest. 
    'p' (lb/in) and 'y' (in).
    '''
    
    from scipy.interpolate import interp1d
    
    #Convert to psi
    Su = Su/144.
    sigma_v_eff = sigma_v_eff/144.
    
    #p-y curve properties
    J     = 0.5
    
    if (z-z_0)<=0:
        #p-y curves for the virtual soil layer between the pile head and the mudline should have p=0
        N_p  = 0.0
        z_cr = 1.0 #Dummy value to keep program from crashing
        
    else:
        try:
            N_p   = 8.0 + sigma_v_eff/Su + J*(z-z_0)/D 

            if N_p > 12.0: N_p = 12.0
                
            z_cr  = (4.0 - sigma_v_eff/Su)*D/J + z_0  #This condition is implemented to avoid zero division errors.

        except ZeroDivisionError:
            print "Division by zero! Su = 0.0 so z_cr cannot be calculated."
    
    p_ult = Su*N_p*D
    y_50  = 2.5*epsilon_50*D
    
    #Normalized lateral displacement
    Y = concatenate((-logspace(3,-4,100),[0],logspace(-4,3,100)))
    
    #Normalized depths
    Z    = z/D
    Z_cr = z_cr/D
    
    #Normalized p-y curves
    P = 0.5*sign(Y)*abs(Y)**(1.0/3.0)  #sign(Y) and abs(Y) used since negative numbers cannot be raised to fractional powers
                                           #Expression equivalent to P = 0.5*Y**(1.0/3.0) for Y>=0
    for i in range(0,len(Y)): 
        if P[i] > 1.0:    P[i] = 1.0
        elif P[i] < -1.0: P[i] = -1.0
    
    if loading_type=='cyclic':
        
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
            
    
    #Un-normallized p-y curves
    p = P*p_ult
    y = Y*y_50
    
    f = interp1d(y,p, kind='linear')   #Interpolation function for p-y curve
    
    #Secant stiffness
    #k = f(y1)/y1
    
    if print_curves=='Yes':
        #Plot of p-y curve and check if 'k' is calculated correctly
        plot(y,p), xlabel('y (in)'), ylabel('p (lb/in)')
        grid(True)
        #plot(y1, k*y1, 'x')
    
    return f


def kodikara_py_curves(z, D, Su, sigma_v_eff, z_0=0.0, alpha=0.0, R1=0.5, A=250, print_curves='No'):
    '''
    Returns an interp1d interpolation function which represents a Bezier-type p-y curve by Kodikara et al. (2009), 
    at the depth of interest.
    
    Important: Make sure to import the interp1 function by running 'from scipy.interpolate import interp1d' in the 
    main program.

    Input:
    -----
    z            - Depth (in)
    D            - Pile diameter (in)
    Su           - Undrained shear strength (psf)
    sigma_v_eff  - Effective vertical stress (psf)
    z_0          - Load eccentricity above the mudline or depth to mudline relative to the pile head (in)
    A            - Shear modulus to undrained shear strength ratio, G/Su (default = 250)
    R1           - Tensile strength to undrained shear strength ratio, sigma_t = R1*Su (default=0.5)
    alpha        - Pile-soil adhesion in shear w.r.t. Su, 0 <= alpha <= 1 (alpha = 0.0 by default)
    
    Note: Pile-soil adhesion in tension is assumed to be equal to the tensile strengh of the soil
    
    Output:
    ------
    Returns an interp1d interpolation function which represents the p-y curve at the depth of interest. 
    'p' (lb/in) and 'y' (in).
    '''
    
    from scipy.interpolate import interp1d
    
    #Convert to psi
    Su = Su/144.
    sigma_v_eff = sigma_v_eff/144.

    """
    The total vertical stress is needed for the gapping criterion but it is not currently available
    in the soil profile input used in the py_analysis_2 function. Therefore let the total overburden
    stress be calculated as follows. This approximation should be updated in a future version of this
    function.
    """
    sigma_v = sigma_v_eff + (62.4/12**3)*(z-z_0)
    
    #Bezier curve parameters
    sigma_t = R1*Su #Tensile strength of the soil
    G       = A*Su  #Shear modulus of the soil
    
    #print "z = %.2f, sigma_t = %.2f, sigma_v = %.2f, 7*Su = %.2f" %(z, sigma_t, sigma_v, 7*Su)
    
    if (sigma_t + sigma_v) >= 7*Su:
        #Gap will not be created
        a = interp1d([0,1], [6.615,7.142], kind='linear')
        b = interp1d([0,1], [1.065,1.093], kind='linear')
        B = 0.4144
        C = 3.7881
        #print 'Gap not created at z = %.1f-in, sigma_v_eff = %.1f-psf, Su = %.1f-psf' %(z,sigma_v_eff,Su)
    
    elif (sigma_t + sigma_v) < 7*Su:
        #Gap will be created
        a = interp1d([0,1], [52.96,54.60], kind='linear')
        b = interp1d([0,1], [1.169,1.290], kind='linear')      
        B = 0.8317
        C = 2.1190
        #print 'Gap created at z = %.1f-in, sigma_v_eff = %.1f-psf, Su = %.1f-psf' %(z,sigma_v_eff,Su)
    
    
    K_i = G*(alpha*B + C)
    y_u = a(alpha)*D/A
    y_e = b(alpha)*D/A
    
    
    #print 'K_i = %.2e, y_e = %.2e, y_u = %.2e' %(K_i, y_e, y_u)
    
    
    #Calculate p_u according to Matlock (1970) with Jeanjan (2009) bounding values for N_p
    J     = 0.5
    
    if (z-z_0)<=0:
        #p-y curves for the virtual soil layer between the pile head and the mudline should have p=0
        y = linspace(-5*D, 5*D, 100)
        p = zeros(len(y))
    
    else:
        try:
            N_p   = 8.0 + sigma_v_eff/Su + J*(z-z_0)/D 

            if N_p > 12.0: N_p = 12.0

            z_cr  = (4.0 - sigma_v_eff/Su)*D/J  #This condition is implemented to avoid zero division errors.

        except ZeroDivisionError:
            print "Division by zero! Su = 0.0 so z_cr cannot be calculated."
    
        p_u = Su*N_p*D

        #p-y curves
        u = linspace(0,1,20)       #Dummy variable

        y_non_lin_range = (1-u)**2*y_e     + 2*u*(1-u)*(p_u/K_i) + u**2*y_u
        p_non_lin_range = (1-u)**2*K_i*y_e + 2*u*(1-u)*p_u       + u**2*p_u

        y = concatenate([array([min(-y_u,-5*D)]), -y_non_lin_range[::-1], array([0]), y_non_lin_range, array([5*D])])
        p = concatenate([array([-p_u]), -p_non_lin_range[::-1], array([0]), p_non_lin_range, array([p_u])])
    
 
    if print_curves=='Yes':
        #Plot of p-y curve and check if 'k' is calculated correctly
        plot(y,p, '-'), xlabel('y (in)'), ylabel('p (lb/in)')
        grid(True)
        
    #Interpolation function for p-y curve
    f = interp1d(y,p, kind='linear')   
    
    return f


def jeanjean_2017_py_curves(z, D, Su, sigma_v_eff, z_0=0.0, Su_0=0.0, gapping='No', alpha = 1.0, TE_DSS_ratio=0.9,
                            custom_py='No', a=0.0,strain_f=0.0, print_curves='No',ls='-', return_Np='No'):
    '''Returns an interp1d interpolation function which represents the Matlock (1970) p-y curve at the depth of interest.
    
    Important: Make sure to import the interp1 function by running 'from scipy.interpolate import interp1d' in the main program.

    Input:
    -----
    z            - Depth relative to pile head (in)
    D            - Pile diameter (in)
    Su           - Undrained shear strength at depth 'z' (psf)
    Su_0         - Undrained shear strength at depth 'z0' (psf)
                   Note: If setting Su0 based on the 'interp1d' function for Su from 'design_soil_profile' then, 
                   it is safer to set Su0=f_Su(z0+0.01) rather than Su0=f_Su(z0) since f_Su(z0) could be zero 
                   and lead to numerical instabilities in the code. 'py_analysis_2()' uses Su0=f_Su(z0+0.01) by default.
    sigma_v_eff  - Effectve vertical stress (psf)
    z_0          - Load eccentricity above the mudline or depth to mudline relative to the pile head (in)
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
    'p' (lb/in) and 'y' (in).
    '''
    
    from scipy.interpolate import interp1d
    
    #Convert to psi
    Su   = Su/144.
    Su_0 = Su_0/144.
    sigma_v_eff = sigma_v_eff/144.
    
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
            print 'psi = %2.2f' %psi
            raise Exception('Failed to calculate alpha based on API method!')
    
    N_1 = 12.0
    N_2 = 3.22
    
    #Normalized rate of undrained shear strength increase
    lamda = Su_0/(Su_1*D)
                          
    d = max(16.8 - 2.3*log(lamda), 14.5)
    
    #Maximum value of N_p under plane strain/full flow conditions 
    N_p_max = 9.0 + 3.0*alpha  #Refered to as 'N_pd' by Jeanjean et al (2017)
    
    if (z-z_0)<=0:
        #p-y curves for the virtual soil layer between the pile head and the mudline should have p=0
        N_p  = 0.0
        
    else:
        #P-y curves for the actual soil 
        N_p0 = min(N_1 - (1-alpha) - (N_1 - N_2)*(1.0 - ((z-z_0)/d/D)**0.6)**1.35, N_p_max)
        
        if isnan(N_p0)==True:
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
            print 'a = %2.2f, strain_f = %2.2f' %(a,strain_f)
            raise Exception('Input values for a and strain_f required!')
            
            
        else:
            A = 1.33 + 0.45*log(a)
            Y_max = strain_f*(2.5-1.2*log(a))
            
            Y = concatenate((-logspace(1,-4,25),[0],logspace(-4,1,25)))
            
            P = sign(Y)*tanh(A*(abs(Y)/Y_max)**0.5) / tanh(A)
            
            for i in range(0,len(P)):
                if P[i] > 1.0:
                    P[i] = 1.0
                elif P[i] < -1.0:
                    P[i] = -1.0
            
    else:
        P = array([-1.0,-1.0,-0.975,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.05,
                       0,0.05,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.975,1.0,1.0])
        
        if Su <= 14.5: #i.e. Su < 14.5psi, 100kPa
            Y = array([-10.,-0.25,-0.15,-0.082,-0.05,-0.032,-0.022,-14.5e-3,-9e-3,-5.3e-3,-3e-3,-3e-4,
                       0,3e-4,3e-3,5.3e-3,9e-3,14.5e-3,0.022,0.032,0.05,0.082,0.15,0.25,10.])
            
        else: #i.e Su > 14.5psi, 100kPa
            Y = array([-10.,-0.25,-0.16,-0.114,-0.07,-0.045,-0.03,-0.02,-12.5e-3,-7e-3,-3.5e-3,-4e-4,
                       0,4e-4,3.5e-3,7e-3,12.5e-3,0.02,0.03,0.045,0.07,0.114,0.16,0.25,10.])

    #Un-normallized p-y curves
    p = P*p_ult
    y = Y*D
    
    f = interp1d(y,p, kind='linear')   #Interpolation function for p-y curve
    
    if print_curves=='Yes':
        #Plot of p-y curve and check if 'k' is calculated correctly
        plot(y,p,ls)
        xlabel('y (in)'), ylabel('p (lb/in)')
        grid(True)

    if return_Np == 'Yes':
        return f, N_p
    else:
        return f
        

###################
#### t-z curve ####
###################

def tz_curve(Su, D, z_0=0.1):
    '''Returns a 'interp1d' function which represents the t-z curve.
    
    Input:
    -----
    Su     - Undrained shear strength (psf)
    D      - Diameter of pile (in)
    z_0    - Displacement at which 't' is fully mobilized (in) (0.1-in by default)
    
    Output:
    ------
    A t-z curve as an 'interp1d' function.
    '''
    
    from scipy.interpolate import interp1d
    
    t = D*(Su/144.0)*array([-1.0,-1.0, 0.0, 1.0, 1.0 ])
    z = array([-D, -z_0, 0.0, z_0, D])
    
    f = interp1d(z,t, kind='linear')
    
    return f


#######################
#### Soil Profile #####
#######################

def design_soil_profile(soil_profile, plot_profile='No', units='inches',
                        plot_relative_to_mudline='No'):
    '''Define the soil profile used by the p-y analyzer. Outputs 'interp1d' functions containing Su and sigma'_v 
    profiles to be used by the p-y curve functions.
    
    Input:
    -----
    soil_profile - A 2D tuple in the following format: ([Depth (in), Su (psf), gamma_sub (pcf), py-model, model parameter])
                   The soil profile should be defined relative to the pile head (i.e. point of lateral load application) so
                   that any load eccentricities can be taken into account. An example soil profile is shown below.
                   Eg: array([[z0,Su0,gamma_sub0,  'Matlock', 0.02],
                              [z1,Su1,gamma_sub1,  'Matlock', 0.01],
                              [z2,Su2,gamma_sub2, 'Jeanjean',  550],
                              ...])
    plot_profile - Plot Su vs depth profile. Choose 'Yes' to plot.
    units        - The units of depth in soil profile plot can switched between 'inches','feet', and SI units 'm, kPa'.
                   This option will NOT affect the output array 'z0'.
    plot_relative_to_mudline - Plot the soil profile relative to the mudline instead of relative to the pile head.
                               'Yes' or 'No'
    
    Output:
    ------
    z0            - Depth of mudline relative to the pile head (in)
    f_Su          - 'interp1d' function containing undrained shear strength profile (psf)
    f_sigma_v_eff - 'interp1d' function containing effective vertical stress profile (psf)
    '''
    
    from scipy.interpolate import interp1d
    
    #Depth of mudline relative to pile head
    z0 = soil_profile[0,0].astype(float)
    
    #Extract data from soil_profile array and zero strength virtual soil layer 
    #from the pile head down to the mudline
    depth     = concatenate([array([0,z0]), soil_profile[:,0].astype(float)])  #inches
    Su        = concatenate([array([0, 0]), soil_profile[:,1].astype(float)])  #psf
    gamma_sub = concatenate([array([0, 0]), soil_profile[:,2].astype(float)])  #psf
   
    if plot_profile=='Yes':
        
        if plot_relative_to_mudline=='Yes': temp=z0
        else: temp = 0.0
        
        if plot_relative_to_mudline=='Yes':
            y_axis_label = 'Depth below mudline'
        else:
            y_axis_label = 'Depth below pile head'
        
        #Plot Su vs z profile for confirmation
        if units=='inches':
            plot(Su, depth-temp, '-')
            xlabel('Undrained shear strength (psf)'), ylabel('%s (in)' %y_axis_label), grid(True)
        elif units=='feet':
            plot(Su, (depth-temp)/12.0, '-')
            xlabel('Undrained shear strength (psf)'), ylabel('%s (ft)' %y_axis_label), grid(True)
        elif units=='SI':
            plot(Su/20.89,(depth-temp)/39.4,'-')
            xlabel('Undrained shear strength (kPa)'), ylabel('%s (m)' %y_axis_label), grid(True)
            
        ax = gca()
        ax.invert_yaxis(), ax.xaxis.tick_top(), ax.xaxis.set_label_position('top')

    #Calculate sigma_v_eff at each depth
    sigma_v_eff = zeros(len(depth))
    
    for i in range(1,len(depth)):
        sigma_v_eff[i] = sigma_v_eff[i-1] + gamma_sub[i-1]*(depth[i]-depth[i-1])/12.0
    
    #Define interpolation functions
    f_Su          = interp1d(depth, Su,          kind='linear')
    f_sigma_v_eff = interp1d(depth, sigma_v_eff, kind='linear')
    
    return z0, f_Su, f_sigma_v_eff


################
### Plot Np ###
###############

def plot_Np_gapping(soil_profile,D=1.0,L=10., alpha=1.0):

    #subplot(1,2,1)
    z0, f_Su, f_sigma_v_eff = design_soil_profile(soil_profile, plot_profile='No')

    z_Np = linspace(z0*1.001,L,200)
    Np_with_gap    = zeros(len(z_Np))
    Np_without_gap = zeros(len(z_Np))
    
    #N_p_max = 9.0 + 3.0*alpha
    
    for i in range(0,len(z_Np)):
        f, Np_with_gap[i] = MM_1_py_curves(z_Np[i],D,f_Su(z_Np[i]),f_sigma_v_eff(z_Np[i]),z_0=z0,Su0=f_Su(z0+0.02),
                                                             epsilon_50=0.02,gapping='Yes',alpha=alpha, return_Np='Yes',)
        f, Np_without_gap[i] = MM_1_py_curves(z_Np[i],D,f_Su(z_Np[i]),f_sigma_v_eff(z_Np[i]),z_0=z0,Su0=f_Su(z0+0.02),
                                                             epsilon_50=0.02,gapping='No',alpha=alpha, return_Np='Yes')

    #figure(figsize=(6,4))
    plot(Np_with_gap,(z_Np-z0)/D)
    plot(Np_without_gap,(z_Np-z0)/D)
    xlabel(r'$N_p$', fontsize=14), ylabel(r'$z/D$', fontsize=14), grid(True)
    #xlim([0,N_p_max+4])

    ax = gca()
    ax.invert_yaxis(), ax.xaxis.tick_top(), ax.xaxis.set_label_position('top')


def Np_murff_1993(z, D, Su, sigma_v_eff, Su_0=0.0, z_0=0.0, alpha=0.0, gapping='Yes'):
    '''
    Returns the lateral bearing capacity factor (N_p) at the depth of interest for a given soil profile.

    Input:
    -----
    z            - Depth relative to pile head (in)
    D            - Pile diameter (in)
    Su           - Undrained shear strength (psf)
    sigma_v_eff  - Effectve vertical stress (psf)
    Su_0         - Undrained shear strength at the mudline (psf)
    z_0          - Load eccentricity above the mudline or depth to mudline relative to the pile head (in)
    alpha        - Pile-soil adhesion factor ranging from 0.0 to 1.0 (0.0 by default)
    
    Output:
    ------
    N_p
    '''
    
    from scipy.interpolate import interp1d
    
    #Change units to psi
    Su = Su/144.0  
    Su_0 = Su_0/144.0
    sigma_v_eff = sigma_v_eff/144.
    
    N_1 = 9.0 + 3*alpha
    N_2 = 7.0 + 2*alpha
    
    if (z-z_0)<=0:
        #p-y curves for the virtual soil layer between the pile head and the mudline should have p=0
        N_p  = 0.0
        
    else:
        #P-y curves for the actual soil 
        k = (Su - Su_0)/(z - z_0) #Secant gradient of the Su versus depth profile

        if k == 0:
            lamda = 6
        else:
            lamda = Su_0/(k*D)

        if lamda < 6:
            xi = 0.25 + 0.05*lamda
        else:
            xi = 0.55
    
        N_p = min(N_1 - N_2*exp(-xi*(z-z_0)/D) + sigma_v_eff/Su, N_1)
        
        #Need to get the proper formulation for no gapping condition
        if gapping=='No':
            N_p = min(2*(N_1 - N_2*exp(-xi*(z-z_0)/D)), N_1)
            
    
    return N_p
    
    
def plot_Np(soil_profile,D=1.0,L=10., py_model='Matlock', invert_y_axis='No',**kwargs):
    
    #Extract optional keyword arguments
    alpha, gapping, ls = 0.0, 'Yes', '' #Default parameters if no **kwargs are defined
    for arg in kwargs:
        if arg=='gapping':
            gapping = kwargs[arg]        
        if arg=='alpha':
            alpha = kwargs[arg]
        if arg=='ls':
            ls = kwargs[arg]

    z0, f_Su, f_sigma_v_eff = design_soil_profile(soil_profile, plot_profile='No')


    z_Np = concatenate((logspace(-5,0,50),linspace(1,L,100)))
    Np   = zeros(len(z_Np))
    
    #N_p_max = 9.0 + 3.0*alpha
    
    for i in range(0,len(z_Np)):
        if py_model=='Matlock':
            f, Np[i] = matlock_py_curves(z_Np[i],D,f_Su(z_Np[i]),f_sigma_v_eff(z_Np[i]),z_0=z0,return_Np='Yes')
        elif py_model=='MM-1':
            f, Np[i] = MM_1_py_curves(z_Np[i],D,f_Su(z_Np[i]),f_sigma_v_eff(z_Np[i]),z_0=z0,Su0=f_Su(z0+0.02),
                                        epsilon_50=0.02,gapping=gapping,alpha=alpha, return_Np='Yes')
        elif py_model=='Jeanjean':
            f,Np[i] = jeanjean_py_curves(z_Np[i],D,f_Su(z_Np[i]),f_sigma_v_eff(z_Np[i]),z_0=z0,Su_0=f_Su(z0),
                                         return_Np='Yes')
        elif py_model=='Yu et al 2015':
            Np[i] = min(N_p_max,11.94 - (1.0-alpha) - (11.94-3.22)*(1.0 - (z_Np[i]/14.5/D)**0.6)**1.35)
        elif py_model=='Murff & Hamilton 1993':
            Np[i] = Np_murff_1993(z_Np[i],D,f_Su(z_Np[i]),f_sigma_v_eff(z_Np[i]),z_0=z0,Su_0=f_Su(z0),alpha=alpha,
                                  gapping=gapping)
        elif py_model=='Jeanjean_etal_2017':
            f,Np[i] = jeanjean_2017_py_curves(z_Np[i],D,f_Su(z_Np[i]),f_sigma_v_eff(z_Np[i]),z_0=z0,Su_0=f_Su(z0),
                                            alpha=alpha,gapping=gapping,return_Np='Yes')

    #figure(figsize=(6,4))
    plot(Np,(z_Np-z0)/D,ls,label=py_model)
    xlabel(r'$N_p$', fontsize=14), ylabel(r'$z/D$', fontsize=14), grid(True)
    #xlim([0,N_p_max+4])

    if invert_y_axis=='Yes':
        ax = gca()
        ax.invert_yaxis(), ax.xaxis.tick_top(), ax.xaxis.set_label_position('top')