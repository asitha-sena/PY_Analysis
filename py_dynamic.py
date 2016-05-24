#Copyright Asitha Senanayake 2016

from pylab import *
from PY_Analysis import py_method


##################################
### Dynamic Analysis Functions ###
##################################

def dyn_lat_tower_1(L=10., D=1., A=1., I=1., t=0.1, E=200e9, rho=7850., n=20, modes=[1,2], 
                   lumped_mass=0.0, lumped_mass_loc=0):

    '''Analyze the free vibration of a circular tower with a constant cross-section and outputs specified mode and frequency. 
    It allows the addition of a single lumped mass any where along the length of the tower.

    Input:
    -----
    L   - Length (m)
    D   - Diameter (m)
    t   - Wall thickness (m)
    E   - Elastic modulus of material (N/m2)
    rho - Material density (kg/m3)
    n   - Number of elements (default=20)
    modes - List of the modes whose results that should be printed
    
    Optional:
    lumped_mass     - Assign a lumped mass to any element (kg)
    lumped_mass_loc - Number of element to which lumped masss is assigned (0 to n-1)

    Output:
    ------
    omega - Fundamental natural frequency (Hz)
    u     - Displacement vector representing the mode shape (m)
    '''

    from scipy import linalg

    #FE Model
    n_elem  = n
    l       = (L/n_elem)*ones(n_elem)    #Element lengths (m)
    I       = I*ones(n_elem) #(pi*(D**4 - (D-2*t)**4)/64.)*ones(n_elem)   #Second moment of area, (m4)
    A       = A*ones(n_elem) #(pi*(D**2 - (D-2*t)**2)/4.)*ones(n_elem)    #Area (m2)
    
    #Array with node locations
    node_loc = linspace(0,L,n+1)

    #Material
    E       = E*ones(n_elem)   #Elastic Modulus (N/m2)
    rho     = rho*ones(n_elem)
    k       = zeros(n_elem)         #Winkler spring stiffness (F/L^2)
    
    #Add mass of nacelle to top most element by modifying the mass density
    j = lumped_mass_loc
    element_vol  = A[j]*l[j]
    element_mass = A[j]*l[j]*rho[j]
    rho[j]       = (element_mass + lumped_mass)/element_vol
    
    w, vr = fe_solver_dyn_2(n,l,E,I,A,rho,k, base_fixity='Fixed')

    u = zeros(n_elem + 1)

    for i in modes:
        omega = w[-i]**0.5
        
        print 'Mode %2d:  Frequency = %8.3f rad/sec = %6.3f Hz, Period = %8.4f s' %(i, omega, omega/(2*pi), 2*pi/omega)
        for j in range(0,len(vr[:,-i])/2):
            u[j] = vr[2*j,-i]

        plot(u/u[-1],node_loc), xlabel('Relative Displacement'), ylabel('Depth below tower top (m)')
        grid(True)
        ax = gca()
        ax.invert_yaxis(), ax.xaxis.tick_top(), ax.xaxis.set_label_position('top')
        
    return w**0.5, vr


def dyn_lat_tower_2(L=10., D_b=6.0, D_t= 3.0, t_b=0.025, t_t=0.020, E=200e9, rho=7840., n=20, modes=[1,2], 
                   lumped_mass=0.0, lumped_mass_loc=0):

    '''Analyze the free vibration of a circular tower with a tapering cross-section and outputs specified mode and frequency. 
    It allows the addition of a single lumped mass any where along the length of the tower.

    Input:
    -----
    L   - Height of tower (m)
    D_b - Diameter at base of tower (m)
    D_t - Diameter at top of tower  (m)
    t_b - Wall thickness at base of tower (m)
    t_t - Wall thickness at top of tower  (m)
    E   - Elastic modulus of material (N/m2)
    rho - Material density (kg/m3)
    n   - Number of elements (default=20)
    modes - List of the modes whose results that should be printed
    
    Optional:
    lumped_mass     - Assign a lumped mass to any element (kg)
    lumped_mass_loc - Number of element to which lumped masss is assigned (0 to n-1)

    Output:
    ------
    omega - Fundamental natural frequency (Hz)
    u     - Displacement vector representing the mode shape (m)
    '''

    from scipy import linalg

    #FE Model
    n_elem,l,A,I,E,rho,node_loc = tower_pile_geom(D_tower_bot=D_b, D_tower_top=D_t, t_tower_bot=t_b, t_tower_top=t_t, L_tower=L, n_tower=n,
                                                  D_pile=D_b, t_pile=t_b, L_pile=0.0, n_pile=0, rho=rho, lumped_mass=lumped_mass, 
                                                  lumped_mass_loc=lumped_mass_loc)

    k       = ones(n_elem)

    w,vr = fe_solver_dyn_2(n,l,E,I,A,rho,k,base_fixity='Fixed')
    
    u = zeros(n_elem + 1)

    figure()
    rcParams['figure.figsize'] = 3,5
    for i in modes:
        omega = w[-i]**0.5
        
        print 'Mode %2d:  Frequency = %8.3f rad/sec = %6.3f Hz, Period = %8.4f s' %(i, omega, omega/(2*pi), 2*pi/omega)
        for j in range(0,len(vr[:,-i])/2):
            u[j] = vr[2*j,-i]
        
        plot(u/u[-1],node_loc), xlabel('Relative Displacement'), ylabel('Depth below tower top (m)')
        grid(True)
        ax = gca()
        ax.invert_yaxis(), ax.xaxis.tick_top(), ax.xaxis.set_label_position('top')
        
    return w**0.5, vr


def dyn_lat_tower_3(D_tower_bot=4.5, D_tower_top=4.5, t_tower_bot=0.027, t_tower_top=0.027, L_tower=90., n_tower=50, 
                        D_pile=6.0, t_pile=0.050, L_pile=60., n_pile=50, rho=7850., E=200.e9, modes=[1,2], k_secant=array([]),
                        lumped_mass=1.0, lumped_mass_loc=-1):
    '''Analyze the free vibration of a circular tower with a tapering cross-section and attached to a pile foundation supported by p-y springs.
    The output consists of specified modes and natural frequencies. 
    
    The addition of a single lumped mass anywhere along the length of the tower is allowed.

    Input:
    -----
    D_tower_bot - Diameter at base of tower (m)
    D_tower_top - Diameter at top of tower  (m)
    t_tower_bot - Wall thickness at base of tower (m)
    t_tower_top - Wall thickness at top of tower  (m)
    L_tower     - Height of tower (m)
    n_tower     - Number elements in to which the tower should be divided. 
    D_pile      - Diameter of the pile foundation (m)
    t_pile      - Wall thickness of the pile foundation (m)
    L_pile      - Length of the pile foundation (m)
    n_pile      - Number of element in to which the pile should be divided
    E           - Elastic modulus of material (N/m2)
    rho         - Material mass density (kg/m3) 
    modes       - List of the modes whose results that should be printed
    k_secant    - 1-d array with modulus of subgrade reaction (i.e. p-y spring stiffnesses) (N/m^2)
                  An empty array is assigned by default but the function will not work with this since its
                  length has to be equal to (n_pile+n_tower). Therefore, this parameter always has to be
                  explicitly defined.
    
    Optional:
    lumped_mass     - Assign a lumped mass to any element (kg)
    lumped_mass_loc - Number of element to which lumped masss is assigned (0 to n-1)

    Output:
    ------
    omega - Fundamental natural frequency (Hz)
    u     - Displacement vector representing the mode shape (m)
    '''
    
    n_elem, l, A, I, E, rho,node_loc = tower_pile_geom(D_tower_bot=D_tower_bot, D_tower_top=D_tower_top, t_tower_bot=t_tower_bot, t_tower_top=t_tower_top, 
                                                       L_tower=L_tower, n_tower=n_tower, D_pile=D_pile, t_pile=t_pile, L_pile=L_pile, n_pile=n_pile, 
                                                       rho=rho, E=E, lumped_mass=lumped_mass, lumped_mass_loc=lumped_mass_loc)

    w,vr = fe_solver_dyn_2(n_elem,l,E,I,A,rho,k_secant,base_fixity='Free')
    #print 'f_0 = %.2f' %((min(w))**0.5)

    u = zeros(n_elem + 1)

    figure()
    rcParams['figure.figsize'] = 3,5
    for i in modes:
        omega = w[-i]**0.5

        print 'Mode %2d:  Frequency = %8.3f rad/sec = %6.3f Hz, Period = %8.4f s' %(i, omega, omega/(2*pi), 2*pi/omega)
        for j in range(0,len(vr[:,-i])/2):
            u[j] = vr[2*j,-i]
        plot(u/u[-1],node_loc), xlabel('Relative Displacement'), ylabel('Depth below tower top (m)')
    
    grid(True)
    ax = gca()
    ax.invert_yaxis(), ax.xaxis.tick_top(), ax.xaxis.set_label_position('top')
        
    return w**0.5, vr


def dyn_lat_tower_4(soil_profile, D_tower_bot=4.5, D_tower_top=4.5, t_tower_bot=0.027, t_tower_top=0.027, 
                    L_tower=90., n_tower=50, D_pile=6.0, t_pile=0.050, L_pile=60., n_pile=50, rho=7850., E=200.e9, modes=[1,2],
                    Y_secant=0.01, py_model='Matlock', plot_profile='No', print_output='Yes',lumped_mass=1.0, lumped_mass_loc=-1,
                    water_depth=0.0, **kwargs):
    '''Analyze the free vibration of a circular tower with a tapering cross-section and attached to a pile foundation supported by p-y springs.
    The output consists of specified modes and natural frequencies. This accepts the specified Su vs depth soil profile and p-y model and 
    calculates the Winkler spring stiffnesses internally as unlike 'dyn_lat_tower_3()' for which the Winkler spring stiffness array should be
    provided as an input.
    
    The addition of a single lumped mass anywhere along the length of the tower is allowed.

    Input:
    -----
    soil_profile - A 2D tuple in the following format: ([Depth (in), Su (psf), gamma_sub (pcf), py-model, model parameter])
                   The soil profile should be defined relative to the pile head (i.e. point of lateral load application) so
                   that any load eccentricities can be taken into account. An example soil profile is shown below.
                   Eg: array([[z0,Su0,gamma_sub0,  'Matlock', 0.02],
                              [z1,Su1,gamma_sub1,  'Matlock', 0.01],
                              [z2,Su2,gamma_sub2, 'Jeanjean',  550]])
                              py_model      - p-y model to be used. Options: 'Matlock', 'Jeanjean', 'Modified Matlock'
    D_tower_bot - Diameter at base of tower (m)
    D_tower_top - Diameter at top of tower  (m)
    t_tower_bot - Wall thickness at base of tower (m)
    t_tower_top - Wall thickness at top of tower  (m)
    L_tower     - Height of tower (m)
    n_tower     - Number elements in to which the tower should be divided. 
    D_pile      - Diameter of the pile foundation (m)
    t_pile      - Wall thickness of the pile foundation (m)
    L_pile      - Length of the pile foundation (m)
    n_pile      - Number of element in to which the pile should be divided
    E           - Elastic modulus of material (N/m2)
    rho         - Material mass density (kg/m3) 
    modes       - List of the modes whose results that should be printed
    py_model    - Select which p-y model to use, 'Matlock', 'Jeanjean', 'Modified Matlock', or 'Kodikara'.
    print_output - Print and plot results from dynamic analysis. 'Yes' or 'No'
    plot_profile  - Plot Su vs depth profile and the location of the structure relative to that soil profile.
                              ...]). 'Yes' or 'No'
    Y_secant      - The lateral displacement at which to calculate the secant stiffness of the p-y curve.
    water_depth   - Depth of water (m) [0.0m by default]
    
    Optional:
    lumped_mass     - Assign a lumped mass to any element (kg)
    lumped_mass_loc - Number of element to which lumped masss is assigned (0 to n-1)

    Optional keywords: **kwargs
    epsilon_50  - Define 'epsilon_50' if 'Matlock' p-y model is selected. (epsilon_50=0.02 by default)
    Gmax_Su_ratio - Define G_mas/Su if Jeanjean or Kodikara p-y models are chosen. (G_max_Su_ratio = 550 by default)

    Output:
    ------
    omega - Fundamental natural frequency (Hz)
    u     - Displacement vector representing the mode shape (m)
    '''

    #Extract optional keyword arguments
    epsilon_50, Gmax_Su_ratio = 0.02, 550 #Default parameters if no **kwargs are defined
    for arg in kwargs:
        if arg=='epsilon_50':
            epsilon_50 = kwargs[arg]        
        if arg=='Gmax_Su_ratio':
             Gmax_Su_ratio= kwargs[arg]
    
    n_elem, l, A, I, E, rho,node_loc = tower_pile_geom(D_tower_bot=D_tower_bot, D_tower_top=D_tower_top, t_tower_bot=t_tower_bot, t_tower_top=t_tower_top, 
                                                       L_tower=L_tower, n_tower=n_tower, D_pile=D_pile, t_pile=t_pile, L_pile=L_pile, n_pile=n_pile, 
                                                       rho=rho, E=E, lumped_mass=lumped_mass, lumped_mass_loc=-1, print_output=print_output)
    
    k_secant = calc_k_secant(soil_profile, node_loc, n_elem, n_pile, n_tower, D_pile, L_pile, L_tower, py_model=py_model,
                             Y_secant=Y_secant, plot_profile=plot_profile, epsilon_50=epsilon_50, Gmax_Su_ratio=Gmax_Su_ratio,
                             water_depth=water_depth)
    
    #Convert units
    k_secant = 6923.*k_secant #psi to Pa, 1psi = 6.923kPa

    w,vr = fe_solver_dyn_2(n_elem,l,E,I,A,rho,k_secant,base_fixity='Free')
    #print 'f_0 = %.2f' %((min(w))**0.5)

    u = zeros(n_elem + 1)

    if print_output=='Yes':
        figure()
        rcParams['figure.figsize'] = 6,4
        for i in modes:
            omega = w[-i]**0.5

            print 'Mode %2d:  Frequency = %8.3f rad/sec = %6.3f Hz, Period = %8.4f s' %(i, omega, omega/(2*pi), 2*pi/omega)
            for j in range(0,len(vr[:,-i])/2):
                u[j] = vr[2*j,-i]
            plot(u/u[0],(L_pile+L_tower)-node_loc)

        xlabel('Relative Displacement'), ylabel('Depth below top of turbine tower (m)'), grid(True)
        ylim(ymin=0)
        ax = gca()
        ax.invert_yaxis()
        
    return w**0.5, vr

def dyn_lat_tower_4_SI(soil_profile, D_tower_bot=4.5, D_tower_top=4.5, t_tower_bot=0.027, t_tower_top=0.027, 
                    L_tower=90., n_tower=50, D_pile=6.0, t_pile=0.050, L_pile=60., n_pile=50, rho=7850., E=200.e9, modes=[1,2],
                    Y_secant=0.01, py_model='Matlock', plot_profile='Yes', print_output='Yes',lumped_mass=1.0, lumped_mass_loc=-1,
                    water_depth=0.0, **kwargs):
    '''Analyze the free vibration of a circular tower with a tapering cross-section and attached to a pile foundation supported by p-y springs.
    The output consists of specified modes and natural frequencies. This accepts the specified Su vs depth soil profile and p-y model and 
    calculates the Winkler spring stiffnesses internally as unlike 'dyn_lat_tower_3()' for which the Winkler spring stiffness array should be
    provided as an input.
    
    The addition of a single lumped mass anywhere along the length of the tower is allowed.

    Input:
    -----
    soil_profile - A 2D tuple in the following format: ([Depth (in), Su (psf), gamma_sub (pcf), py-model, model parameter])
                   The soil profile should be defined relative to the pile head (i.e. point of lateral load application) so
                   that any load eccentricities can be taken into account. An example soil profile is shown below.
                   Eg: array([[z0,Su0,gamma_sub0,  'Matlock', 0.02],
                              [z1,Su1,gamma_sub1,  'Matlock', 0.01],
                              [z2,Su2,gamma_sub2, 'Jeanjean',  550]])
                              py_model      - p-y model to be used. Options: 'Matlock', 'Jeanjean', 'Modified Matlock'
    D_tower_bot - Diameter at base of tower (m)
    D_tower_top - Diameter at top of tower  (m)
    t_tower_bot - Wall thickness at base of tower (m)
    t_tower_top - Wall thickness at top of tower  (m)
    L_tower     - Height of tower (m)
    n_tower     - Number elements in to which the tower should be divided. 
    D_pile      - Diameter of the pile foundation (m)
    t_pile      - Wall thickness of the pile foundation (m)
    L_pile      - Length of the pile foundation (m)
    n_pile      - Number of element in to which the pile should be divided
    E           - Elastic modulus of material (N/m2)
    rho         - Material mass density (kg/m3) 
    modes       - List of the modes whose results that should be printed
    py_model    - Select which p-y model to use, 'Matlock', 'Jeanjean', 'Modified Matlock', or 'Kodikara'.
    print_output - Print and plot results from dynamic analysis. 'Yes' or 'No'
    plot_profile  - Plot Su vs depth profile and the location of the structure relative to that soil profile.
                              ...]). 'Yes' or 'No'
    Y_secant      - The lateral displacement at which to calculate the secant stiffness of the p-y curve.
    water_depth   - Depth of water (m) [0.0m by default]
    
    Optional:
    lumped_mass     - Assign a lumped mass to any element (kg)
    lumped_mass_loc - Number of element to which lumped masss is assigned (0 to n-1)

    Optional keywords: **kwargs
    epsilon_50  - Define 'epsilon_50' if 'Matlock' p-y model is selected. (epsilon_50=0.02 by default)
    Gmax_Su_ratio - Define G_mas/Su if Jeanjean or Kodikara p-y models are chosen. (G_max_Su_ratio = 550 by default)

    Output:
    ------
    omega - Fundamental natural frequency (Hz)
    u     - Displacement vector representing the mode shape (m)
    '''

    #Extract optional keyword arguments
    epsilon_50, Gmax_Su_ratio = 0.02, 550 #Default parameters if no **kwargs are defined
    for arg in kwargs:
        if arg=='epsilon_50':
            epsilon_50 = kwargs[arg]        
        if arg=='Gmax_Su_ratio':
             Gmax_Su_ratio= kwargs[arg]
    
    n_elem, l, A, I, E, rho,node_loc = tower_pile_geom(D_tower_bot=D_tower_bot, D_tower_top=D_tower_top, t_tower_bot=t_tower_bot, t_tower_top=t_tower_top, 
                                                       L_tower=L_tower, n_tower=n_tower, D_pile=D_pile, t_pile=t_pile, L_pile=L_pile, n_pile=n_pile, 
                                                       rho=rho, E=E, lumped_mass=lumped_mass, lumped_mass_loc=lumped_mass_loc, print_output=print_output)
    
    z_0, f_Su, f_sigma_v_eff = design_soil_profile_SI(soil_profile, plot_profile='No')
    #Plot structure
    if plot_profile=='Yes':
        Su_max = f_Su(1.2*(L_tower+L_pile))/1000 #kPa
        #print 'Su_max = %.f' %Su_max
        plot([-0.2*Su_max,-0.2*Su_max], [0,L_tower], 'g-', lw=2)
        plot([-0.2*Su_max,-0.2*Su_max], [L_tower,(L_tower+L_pile)], 'g-', lw=5)

        plot([-0.5*Su_max, Su_max], [L_tower,L_tower], 'b--')
        text(-0.5*Su_max, 0.95*L_tower, 'Sea Level', color='b')
        #xlim([-0.6*Su_max,1.1*Su_max])
        
    k_secant = calc_k_secant_SI(soil_profile, node_loc, n_elem, n_pile, n_tower, D_pile, L_pile, L_tower, py_model=py_model,
                             Y_secant=Y_secant, plot_profile=plot_profile, epsilon_50=epsilon_50, Gmax_Su_ratio=Gmax_Su_ratio,
                             water_depth=water_depth)
    
    #figure()
    #plot(k_secant), grid(True)
    
    #print 'z(m)    k_secant (N/m^2)'
    #print '------------------------'
    
    #for i in range(n_elem-5,n_elem):
    #    print '%5.1f %6.3e' %((node_loc[i]+node_loc[i+1])/2., k_secant[i])

    w,vr = fe_solver_dyn_2(n_elem,l,E,I,A,rho,k_secant,base_fixity='Free')
    print 'Fundamental natural frequency, f_0 = %6.3f-Hz\n' %((min(w))**0.5/2/pi)

    u = zeros(n_elem + 1)
    

    if print_output=='Yes':
        figure()
        rcParams['figure.figsize'] = 6,4
        for i in modes:
            omega = w[-i]**0.5

            print 'Mode %2d:  Frequency = %8.3f rad/sec = %6.3f Hz, Period = %8.4f s' %(i, omega, omega/(2*pi), 2*pi/omega)
            for j in range(0,len(vr[:,-i])/2):
                u[j] = vr[2*j,-i]
            plot(u/u[-1] - 1,node_loc)

        xlabel('Relative Displacement'), ylabel('Depth below top of turbine tower (m)'), grid(True)
        ylim(ymin=0)
        ax = gca()
        ax.invert_yaxis()
        
    return w**0.5, vr


#######################
### Static Analysis ###
#######################

def py_fe_3(soil_profile, D_tower_bot=4.5, D_tower_top=4.5, t_tower_bot=0.027, t_tower_top=0.027, L_tower=90., n_tower=50,
            D_pile=6.0, t_pile=0.050, L_pile=60., n_pile=50, water_depth=0.0, E=200.e9, V_0=0.0, V_n=0.0, M_0=0.0, M_n=0.0, 
            iterations=10, py_model='Matlock', plot_profile='No', convergence_tracker='No', loc=2, **kwargs):

    '''Models a laterally loaded pile using the p-y method. The solution for lateral displacements 
    is obtained by solving the 4th order ODE, EI*d4y/dz4 + ky = 0 using the finite element method. 
    
    This function is different from 'py_fe_1' and 'py_fe_1' because it takes the pile head load as an input 
    rather than the pile head displacement. Moreover, it follows the format of 'py_dynamic.dyn_lat_tower_4'
    by using 'py_dynamic.tower_pile_geom' to generate the geometry of the structure than to generate it 
    internally. This provides a lot more flexibility with regard to the types of structures to be modelled.
    Moreover, SI units are used instead of Imperial units.
    

    ***Axial forces are neglected in the current form of this code. "fe_solver_3" will have to be upgraded
    with frame elements in order to account for axial loads.
    
    Assumes that EI remains constant with respect to curvature i.e. pile material remains in the elastic region.

    Uses 'fe_solver_3()'.
    
    Input:
    -----
    soil_profile - A 2D tuple in the following format: ([Depth (m), Su (kPa), gamma_sub (kN/m^3), py-model, model parameter])
                   The soil profile should be defined relative to the pile head (i.e. point of lateral load application) so
                   that any load eccentricities can be taken into account. An example soil profile is shown below.
                   Eg: array([[z0,Su0,gamma_sub0,  'Matlock', 0.02],
                              [z1,Su1,gamma_sub1,  'Matlock', 0.01],
                              [z2,Su2,gamma_sub2, 'Jeanjean',  550]])
                              
                   *This program is still unable to analyze soil profile with different p-y models for layers. This functionality
                   will be added in the future.
                  
    D_tower_bot - Diameter at base of tower (m)
    D_tower_top - Diameter at top of tower  (m)
    t_tower_bot - Wall thickness at base of tower (m)
    t_tower_top - Wall thickness at top of tower  (m)
    L_tower     - Height of tower (m)
    n_tower     - Number elements in to which the tower should be divided. 
    D_pile      - Diameter of the pile foundation (m)
    t_pile      - Wall thickness of the pile foundation (m)
    L_pile      - Length of the pile foundation (m)
    n_pile      - Number of element in to which the pile should be divided
    water_depth - Depth of water (m). The water level is alway defined at the tower-pile interface. The mudline
                  is set to this depth below the tower-pile interface.
    E           - Elastic modulus of material (N/m2)
    V_0         - Lateral load at pile head (N)
    V_n         - Force at pile head/tip  (N),  shear causing clockwise rotation of pile is positive.
    M_0, M_n    - Moment at pile head/tip (Nm), moments causing tension on left side of pile is positive.
    iterations  - Number of iterations to repeat calculation in order obtain convergence of 'y'
                  (A better approach is to iterate until a predefined tolerance is achieved but this requires additional
                  coding so, I will implement this later.)
    py_model    - Select which p-y model to use, 'Matlock', 'Jeanjean', 'Modified Matlock', or 'Kodikara'.
    plot_profile  - Plot Su vs depth profile and the location of the structure relative to that soil profile.
                              ...]). 'Yes' or 'No'
    water_depth   - Depth of water (m) [0.0m by default]
    
    Optional:
    convergence_tracker - Track how k_secant converges to actual p-y curve at a selected node
    loc                 - Node number at which k_secant to be tracked (0 to n+1)
    
    Optional keywords: **kwargs
    epsilon_50  - Define 'epsilon_50' if 'Matlock' p-y model is selected. (epsilon_50=0.02 by default)
    Gmax_Su_ratio - Define G_mas/Su if Jeanjean or Kodikara p-y models are chosen. (G_max_Su_ratio = 550 by default)
    
    Output:
    ------
    y           - Lateral displacement at each node, length of array = (n+1)
    rot         - Rotation at each node, lengthh of array = (n+1)
    l           - Vector of node locations along pile
    '''
    
    from scipy import linalg
    
    #Extract optional keyword arguments
    epsilon_50, Gmax_Su_ratio = 0.02, 550 #Default parameters if no **kwargs are defined
    for arg in kwargs:
        if arg=='epsilon_50':
            epsilon_50 = kwargs[arg]        
        if arg=='Gmax_Su_ratio':
             Gmax_Su_ratio= kwargs[arg]
    
    n, l, A, I, E, rho,node_loc = tower_pile_geom(D_tower_bot=D_tower_bot, D_tower_top=D_tower_top, t_tower_bot=t_tower_bot, 
                                                  t_tower_top=t_tower_top, L_tower=L_tower, n_tower=n_tower, D_pile=D_pile, 
                                                  t_pile=t_pile, L_pile=L_pile, n_pile=n_pile, rho=0.0, E=E, lumped_mass=0.0, 
                                                  lumped_mass_loc=-1, print_output='No')
    
    #Number of nodes
    N = n+1
    
    #Initialize and assemble array/list of p-y curves at each real node
    z = zeros(n)
    py_funs  = []
    k_secant = zeros(n)
    
    #Extract soil profile data
    z_0, f_Su, f_sigma_v_eff = design_soil_profile_SI(soil_profile, plot_profile=plot_profile)
    
    #Plot structure
    if plot_profile=='Yes':
        Su_max = f_Su(1.2*(L_tower+L_pile))/1000 #kPa
        
        plot([-0.2*Su_max,-0.2*Su_max], [0,L_tower], 'g-', lw=2)
        plot([-0.2*Su_max,-0.2*Su_max], [L_tower,L_tower+L_pile], 'g-', lw=5)

        plot([-0.5*Su_max, Su_max], [L_tower,L_tower], 'b--')
        text(-0.5*Su_max, 0.95*L_tower, 'Sea Level', color='b')
        xlim([-0.6*Su_max,1.1*Su_max])
        
    
    #Array for displacement
    y = ones(N)*(0.01*D_pile)   #An initial value of 0.01D was arbitrarily chosen
    
    #fe_solver_dyn_3() applies springs to each element as opposed to each node as in the fd_solver.
    #Therefore, p-y curves should be calculated at the mid-point of each element
    for i in range(0,n):
        z[i] = (node_loc[i]+node_loc[i+1])/2.0
    
        Su, sigma_v_eff = f_Su(z[i]), f_sigma_v_eff(z[i])
    
        if py_model=='Matlock':
            py_funs.append(matlock_py_curves_SI(z[i], D_pile, Su, sigma_v_eff, z_0=z_0, epsilon_50=epsilon_50, print_curves='No'))
        elif py_model=='Jeanjean':
            py_funs.append(jeanjean_py_curves(z[i], D_pile,Su, sigma_v_eff, z_0=z_0, Su_0=f_Su(z_0), A=Gmax_Su_ratio))
        elif py_model=='Modified Matlock':
            py_funs.append(modified_matlock_py_curves(z[i], D, Su, sigma_v_eff, z_0=z_0, epsilon_50=epsilon_50, print_curves='No'))
        elif py_model=='Kodikara':
            py_funs.append(kodikara_py_curves(z[i], D_pile, Su, sigma_v_eff, z_0=z_0, R1=0.5, A=Gmax_Su_ratio, print_curves='No'))
        else:
            print "P-y model not properly defined. Please select one of the following:"
            print "'Matlock', 'Jeanjean', 'Modified Matlock', 'Kodikara'"
        
        k_secant[i]     = py_funs[i](y[i])/y[i]
        
    
    #Track k_secant and current displacements
    if convergence_tracker=='Yes':
        y1 = linspace(-2.*D_pile,2.*D_pile,500)
        figure()
        plot(y1, py_funs[loc](y1))
        xlabel('y (m)'), ylabel('p (N/m)'), grid(True)
    
    
    for j in range(iterations):
            
        y,rot = fe_solver_3(n,l,E,I,V_0,V_n,M_0,M_n,k_secant)
        
        if convergence_tracker=='Yes':
            plot(y[loc], k_secant[loc]*y[loc], 'x')
            
        for i in range(0,n):
            k_secant[i] = py_funs[i](y[i])/y[i]

    
    return y,rot,node_loc



#####################
### Pile Geometry ###
#####################

def tower_pile_geom(D_tower_bot=6.0, D_tower_top=3.87, t_tower_bot=0.027, t_tower_top=0.019, L_tower=90.0, n_tower=50,
                    D_pile=6.0, t_pile=0.05, L_pile=60.0, n_pile=50, rho=7850.0, lumped_mass=1e3, lumped_mass_loc=0, E=200.e9,
                    print_output='Yes'):
    '''Calculates the geometrical and material inputs for 'py_method.py_fe_dyn()'. The very top element will
    be converted to represent the mass of the nacelle by changing its mass density. 
    
    The node and element count start from the pile tip and increases upwards.
    
    Input:
    -----
    D_tower_bot  - Diameter of the tower at the bottom (m)
    D_tower_top  - Diameter of the tower at the top    (m)
    t_tower_bot  - Wall thickness at the tower bottom  (m)
    t_tower_top  - Wall thickness at the tower top     (m)
    L_tower      - Length/height of tower              (m)
    n_tower      - Number elements in to which the tower is discretized
    D_pile       - Diameter of the monopile foundation (m)
    t_pile       - Wall thickness of the monopile foundation (m)
    L_pile       - Length of pile (m)
    n_pile       - Number of elements in to which the pile is discretized
    rho          - Mass density of the tower and pile material (7850 kg/m^3 by default i.e. steel)
    lumped_mass  - Magnitude of lumped mass (kg)
    lumped_mass_loc - Element location of lumped mass (-1 for lumped mass at the top)
    E            - Elastic modulus of material (Pa) (200GPa for steel by default)
    print_output - Print the total mass of tower and top element. 'Yes' or 'No'
    
    Output:
    ------
    A    - 1-d array with cross-sectional area of each element (in^2)
    I    - 1-d array with second moment of area of each element (in^4)
    rho  - 1-d array with mass density of each element (lb-m/in^3)
    l    - 1-d array with length of each element (in)
    node_loc - 1-d array with location of each node relative to bottom of pile (i.e. node 0)
    '''
    """
    #Unit conversion from SI to US Customary
    D_tower_bot = D_tower_bot*3.28*12 #in
    D_tower_top = D_tower_top*3.28*12 #in
    t_tower_bot = t_tower_bot*3.28*12 #in
    t_tower_top = t_tower_top*3.28*12 #in
    L_tower     = L_tower*3.28*12 #in
    D_pile      = D_pile*3.28*12 #in
    t_pile      = t_pile*3.28*12 #in
    L_pile      = L_pile*3.28*12 #in
    rho         = rho*2.2/39.36**3/386.4 #lb-m/in^3
    M_nacelle   = M_nacelle*2.2/386.4    #lb-m
    """
    
    n_elem = n_tower + n_pile #Number of elements
    
    #Initialize arrays for L, A, I, and rho
    l   = zeros(n_elem)
    A   = zeros(n_elem)
    I   = zeros(n_elem)
    
    #Array to specify location of each node
    nodes    = n_elem + 1
    node_loc = zeros(nodes)

    #Length vector
    l_e_tower = L_tower/n_tower*ones(n_tower)
    
    #Add pile only if it is defined (i.e. n_pile > 0)
    if n_pile>0: 
        l_e_pile = L_pile/n_pile*ones(n_pile)
        l = concatenate([l_e_tower, l_e_pile])
    else:
        l = l_e_tower
    
    #Density and elastic modulus vectors
    rho = rho*ones(n_elem)
    E   = E*ones(n_elem)

    #Cross-sectional area vector
    D_gradient = (D_tower_bot - D_tower_top)/L_tower
    t_gradient = (t_tower_bot - t_tower_top)/L_tower
    
    total_mass = 0.0
    cumulative_mass = zeros(nodes) #Mass of the tower section upto below each node
    
    cumulative_length = 0.0
    pile_mass = 0.0
    
    #Populate arrays
    for i in range(0,n_tower):
        cumulative_length += l[i]
        node_loc[i+1] = cumulative_length #Assign location of each node
        
        D = D_tower_top + D_gradient*(cumulative_length - l[i]/2.)
        t = t_tower_top + t_gradient*(cumulative_length - l[i]/2.)
        
        #print '%6.2f  %6.2f  %6.3f  %6.4f  %6.3f' %(l[i], node_loc[i], D, t, D_gradient*(cumulative_length- L_pile))

        A[i] = pi*(D**2 - (D-2*t)**2)/4.0
        I[i] = pi*(D**4 - (D-2*t)**4)/64.0
        
        cumulative_mass[i] += A[i]*l[i]*rho[i]
        total_mass         += A[i]*l[i]*rho[i]
        
    for i in range(n_tower, n_tower+n_pile):
        cumulative_length += l[i]
        node_loc[i+1] = cumulative_length #Assign location of each node

        D = D_pile
        t = t_pile

        A[i] = pi*(D**2 - (D-2*t)**2)/4.0
        I[i] = pi*(D**4 - (D-2*t)**4)/64.0
        
        pile_mass += l[i]*A[i]*rho[i]


    #Add mass of nacelle to top most element by modifying the mass density
    j = lumped_mass_loc
    element_vol  = A[j]*l[j]
    element_mass = A[j]*l[j]*rho[j]
    rho[j]       = (element_mass + lumped_mass)/element_vol

    if print_output=='Yes':
        #print 'Volume of top most element = %.2f-in^3' %vol
        print 'Mass of pile = %.f kg' %pile_mass
        print 'Mass of element with lumped mass   = %.f kg' %(rho[j]*element_vol)
        print 'Total mass of only the tower       = %.f kg\n' %total_mass

        #rcParams['figure.figsize'] = 18, 4
        #figure()
        #subplot(1,3,1), plot(rho) , ylabel('$\rho$')
        #subplot(1,3,2), plot(A), ylabel('A')
        #subplot(1,3,3), plot(I), ylabel('I')
        #print 'node_loc = %d' %len(node_loc)
        #print 'Nodes = %d' %nodes
    
    return n_elem, l, A, I, E, rho, node_loc


###############
### Solvers ###
###############

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
    The calculated natural frequencies are based on the k_secant values estimated from the input p-y curves. 
    Therefore, the k_secant values should be carefully selected. The values of k_secant calculated based on 
    p-y curves at each node is dependent on the initial assumed displacement.
    
    Specified boundary conditions: Displacement and moment at pile head. Shear and moment at pile tip. 
                                   Axial force at pile head.
                                   
    The fixity at the base (i.e. node 0) can be controlled by the keyword argument 'base_fixity'. This is the
    only difference between 'fe_solver_dyn_2()' and 'fe_solver_dyn_1()'
    

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
        boundary_dof     = array([2*N-2,2*N-1])
        boundary_dof_val = array([0.,0.])
    elif base_fixity =='Pinned':
        boundary_dof     = array([2*N-1])
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


def fe_solver_3(n,l,E,I,V_0,V_n,M_0,M_n,k_secant):
    '''Solves the finite element model from 'py_fe_2' using the elimination method for boundary condidition.
    This function should be run iteratively for non-linear p-y curves by updating 'k_secant' using 'y'.
    A single iteration is sufficient if the p-y curves are linear.
    
    Specified boundary conditions: Displacement and moment at pile head. Shear and moment at pile tip. 
                                   Axial force at pile head.

    Note:
    The reduced stiffness matrix is derived based on specified displacement at the pile head and then solved.
    V_0 is provided as input rather than y_0 as in the case of 'fe_solver_1' and 'fe_solver_2'.
    

     Input:
    -----
    n        - Number of elements
    l        - 1-d array of element lengths (in)
    E        - 1-d array of elastic modulus of material in each element (lb/in^2)
    I        - 1-d array of second moment of area of each element (in^4)
    A        - 1-d array of cross sectional area of each element (in^2)
    V_0      - Shear at pile head (lb)
    V_n      - Shear at pile tip (lb)
    M_0, M_n - Moment at pile head/tip (lb-in)
    k_secant - Vector of secant stiffness from p-y curves for each element (lb/in^2)
    
    Output:
    ------
    u         - Displacement vector (in)
    slope     - Slope vector
    '''
    
    from scipy import linalg
    
    #FE Model
    N       = n+1               #Number of nodes

    #External loads
    f = 0*ones(2*N)   #Nodal actions (point loads on even dofs and moments on odd dofs)
    p = 0*ones(n)     #Distributed load
    k = k_secant      #Winkler spring stiffness, F/L^2
    
    f[0]       = V_0
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

    q     = linalg.solve(K,F)

    y     = zeros(N)
    slope = zeros(N)

    for i in range(0,N):
        y[i]     = q[2*i]
        slope[i] = q[2*i+1]
    
    return y,slope


###############################################
### k_secant functions for dynamic analysis ###
###############################################


def calc_k_secant(soil_profile, node_loc, n_elem, n_pile, n_tower, D_pile, L_pile, L_tower, py_model='Matlock',
                  Y_secant=0.01, plot_profile='No', water_depth=0.0, **kwargs):
    '''Calculates the stiffness of the Winkler springs (i.e. secant stiffness of a p-y curve) for a wind turbine structure 
    defined by 'py_dynamic.tower_pile_geom()' based on the specified soil profile ('py_dynamic.design_soil_profile()') and 
    p-y model.
    
    *** Note: The soil profile data is provided in US customary units which pile geometry is specified in SI units. ***
    
    Input:
    -----
    soil_profile - A 2D tuple in the following format: ([Depth (in), Su (psf), gamma_sub (pcf), py-model, model parameter])
                   The soil profile should be defined relative to the pile head (i.e. point of lateral load application) so
                   that any load eccentricities can be taken into account. An example soil profile is shown below.
                   Eg: array([[z0,Su0,gamma_sub0,  'Matlock', 0.02],
                              [z1,Su1,gamma_sub1,  'Matlock', 0.01],
                              [z2,Su2,gamma_sub2, 'Jeanjean',  550],
                              ...])
    node_loc      - 1-d array with locations of all nodes in the structure (m). 
                    Output from 'py_dynamic.tower_pile_geom()'
    n_elem        - Total number of elements in the structure. Output from 'py_dynamic.tower_pile_geom()'
    D_pile        - Diameter of the pile (m)
    py_model      - p-y model to be used. Options: 'Matlock', 'Jeanjean', 'Modified Matlock'
    Y_secant      - The lateral displacement at which to calculate the secant stiffness of the p-y curve.
    plot_profile  - Plot Su vs depth profile and the location of the structure relative to that soil profile.
    water_depth   - Depth of water at this location (m) [0.0m by default]

    Optional keywords: **kwargs
    epsilon_50  - Define 'epsilon_50' if 'Matlock' p-y model is selected. (epsilon_50=0.02 by default)
    Gmax_Su_ratio - Define G_mas/Su if Jeanjean or Kodikara p-y models are chosen. Default values are:
                     Jeanjean (2009) -> G_max_Su_ratio = 550
                     Kodikara (2009) -> G_max_Su_ratio = 250
    '''

    #Extract optional keyword arguments
    epsilon_50, Gmax_Su_ratio = 0.02, 550 #Default parameters if no **kwargs are defined
    for arg in kwargs:
        if arg=='epsilon_50':
            epsilon_50 = kwargs[arg]        
        if arg=='Gmax_Su_ratio':
            Gmax_Su_ratio = kwargs[arg]
             
    z_0, f_Su, f_sigma_v_eff = design_soil_profile(soil_profile, plot_profile=plot_profile, y_axis_label='Depth below top of turbine tower (ft)')

    #P-y curves
    z = zeros(n_elem)
    k_secant = zeros(n_elem)

    #fe_solver_dyn_3() applies springs to each element as opposed to each node as in the fd_solver.
    #Therefore, p-y curves should be calculated at the mid-point of each element
    for i in range(0,n_elem):
        z[i] = (L_tower+L_pile)*39.4 - (node_loc[i]+node_loc[i+1])*39.4/2.0

        Su, sigma_v_eff = f_Su(z[i]), f_sigma_v_eff(z[i])

        if py_model=='Matlock':
            k_secant[i] = matlock_k_secant(z[i], D_pile*39.4, Su, sigma_v_eff, Y_secant=Y_secant, z_0=z_0, epsilon_50=epsilon_50)
        elif py_model=='Jeanjean':
            k_secant[i] = jeanjean_k_secant(z[i], D_pile*39.4, Su, sigma_v_eff, Y_secant=Y_secant, z_0=z_0, Su_0=f_Su(z_0), A=Gmax_Su_ratio)
        elif py_model=='Modified Matlock':
            k_secant[i] = modified_matlock_k_secant(z[i], D_pile*39.4, Su, sigma_v_eff, Y_secant=Y_secant, z_0=z_0, epsilon_50=epsilon_50)
        elif py_model=='Kodikara':
            k_secant[i] = kodikara_k_secant(z[i], D_pile*39.4, Su, sigma_v_eff, Y_secant=Y_secant, z_0=z_0, R1=0.5, A=Gmax_Su_ratio, print_curves='No')
        else:
            print "P-y model not properly defined. Please select one of the following:"
            print "'Matlock', 'Jeanjean', 'Modified Matlock', 'Kodikara'"
            break

    if plot_profile=='Yes':
        plot(500*ones(n_pile), z[0:n_pile]/12., lw=5)
        plot(500*ones(n_tower),z[n_pile:n_elem]/12.,'g-',lw=2)
        ylim([0,1.2*(L_tower+L_pile+water_depth)*3.28]) #limit of y-axis in feet
        ax1 = gca()
        ax1.invert_yaxis()
        
        twinx()
        plot([0,0],[0,1.2*(L_tower+L_pile+water_depth)],'.')
        ylim([0,1.2*(L_tower+L_pile+water_depth)])     #limit of y-axis in meters
        ax2 = gca()
        ax2.invert_yaxis()
        ylabel('(m)')
        
    return k_secant


def calc_k_secant_SI(soil_profile, node_loc, n_elem, n_pile, n_tower, D_pile, L_pile, L_tower, py_model='Matlock',
                  Y_secant=0.01, plot_profile='No', water_depth=0.0, **kwargs):
    '''Calculates the stiffness of the Winkler springs (i.e. secant stiffness of a p-y curve) for a wind turbine structure 
    defined by 'py_dynamic.tower_pile_geom()' based on the specified soil profile ('py_dynamic.design_soil_profile()') and 
    p-y model.
    
    *** Note: The soil profile data is provided in US customary units which pile geometry is specified in SI units. ***
    
    Input:
    -----
    soil_profile - A 2D tuple in the following format: ([Depth (m), Su (kPa), gamma_sub (kN/m^3), py-model, model parameter])
                   The soil profile should be defined relative to the pile head (i.e. point of lateral load application) so
                   that any load eccentricities can be taken into account. An example soil profile is shown below.
                   Eg: array([[z0,Su0,gamma_sub0,  'Matlock', 0.02],
                              [z1,Su1,gamma_sub1,  'Matlock', 0.01],
                              [z2,Su2,gamma_sub2, 'Jeanjean',  550],
                              ...])
    node_loc      - 1-d array with locations of all nodes in the structure (m). 
                    Output from 'py_dynamic.tower_pile_geom()'
    n_elem        - Total number of elements in the structure. Output from 'py_dynamic.tower_pile_geom()'
    D_pile        - Diameter of the pile (m)
    py_model      - p-y model to be used. Options: 'Matlock', 'Jeanjean', 'Modified Matlock'
    Y_secant      - The lateral displacement at which to calculate the secant stiffness of the p-y curve.
    plot_profile  - Plot Su vs depth profile and the location of the structure relative to that soil profile.
    water_depth   - Depth of water at this location (m) [0.0m by default]

    Optional keywords: **kwargs
    epsilon_50  - Define 'epsilon_50' if 'Matlock' p-y model is selected. (epsilon_50=0.02 by default)
    Gmax_Su_ratio - Define G_mas/Su if Jeanjean or Kodikara p-y models are chosen. (G_max_Su_ratio = 550 by default)
    '''

    #Extract optional keyword arguments
    epsilon_50, Gmax_Su_ratio = 0.02, 550 #Default parameters if no **kwargs are defined
    for arg in kwargs:
        if arg=='epsilon_50':
            epsilon_50 = kwargs[arg]        
        if arg=='Gmax_Su_ratio':
            Gmax_Su_ratio = kwargs[arg]
             
    z_0, f_Su, f_sigma_v_eff = design_soil_profile_SI(soil_profile, plot_profile=plot_profile)

    #P-y curves
    z = zeros(n_elem)
    k_secant = zeros(n_elem)

    #fe_solver_dyn_3() applies springs to each element as opposed to each node as in the fd_solver.
    #Therefore, p-y curves should be calculated at the mid-point of each element
    for i in range(0,n_elem):
        z[i] = (node_loc[i]+node_loc[i+1])/2.0

        Su, sigma_v_eff = f_Su(z[i]), f_sigma_v_eff(z[i])

        if py_model=='Matlock':
            k_secant[i] = matlock_k_secant_SI(z[i], D_pile, Su, sigma_v_eff, Y_secant=Y_secant, z_0=z_0, epsilon_50=epsilon_50)
        #elif py_model=='Jeanjean':
        #    k_secant[i] = jeanjean_k_secant(z[i], D_pile*39.4, Su, sigma_v_eff, Y_secant=Y_secant, z_0=z_0, Su_0=f_Su(z_0), A=Gmax_Su_ratio)
        #elif py_model=='Modified Matlock':
        #    k_secant[i] = modified_matlock_k_secant(z[i], D_pile*39.4, Su, sigma_v_eff, Y_secant=Y_secant, z_0=z_0, epsilon_50=epsilon_50)
        #elif py_model=='Kodikara':
        #    k_secant[i] = kodikara_k_secant(z[i], D_pile*39.4, Su, sigma_v_eff, Y_secant=Y_secant, z_0=z_0, R1=0.5, A=Gmax_Su_ratio, print_curves='No')
        else:
            print "P-y model not properly defined. Please select one of the following:"
            print "'Matlock', 'Jeanjean', 'Modified Matlock', 'Kodikara'"
            break
        
    return k_secant


def matlock_k_secant_SI(z, D, Su, sigma_v_eff, Y_secant=0.01, z_0=0.0, epsilon_50=0.02, loading_type='static'):
    '''Returns secant modulus of the Matlock (1970) p-y curve at the depth of interest at the specified displacement. This
    value can be used to calculate the stiffness of the pile-soil response at very low displacement which in turn is needed
    to find the natural frequencies of a wind turbine attached to the said pile.
    
    
    Important: Make sure to import the interp1 function by running 'from scipy.interpolate import interp1d' in the main program.

    Input:
    -----
    z            - Depth relative to pile head (m)
    D            - Pile diameter (m)
    Su           - Undrained shear strength (Pa)
    sigma_v_eff  - Effectve vertical stress (Pa)
    Y_secant     - The lateral displacement at which the secant modulus is calculated normalized by D
                   Default: Y_secant = 0.01 -> y_secant = 0.01*D
    z_0          - Load eccentricity above the mudline or depth to mudline relative to the pile head (m)
    epsilon_50   - Strain at half the strength as defined by Matlock (1970).
                   Typically ranges from 0.005 (stiff clay) to 0.02 (soft clay).
    loading_type - Either 'static' or 'cyclic'
    
    Output:
    ------
    Returns the secant modulus of the p-y curve at the specified displacement 'y' in units of N/m^2.
    '''
    
    from scipy.interpolate import interp1d
    
    
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
    y_secant = Y_secant*D
    Y = y_secant/y_50
    
    #Normalized depths
    Z    = z/D
    Z_cr = z_cr/D
    
    #Normalized p-y curves
    P = 0.5*Y**(1.0/3.0)  
    
    if P > 1.0:    P = 1.0
    
    if loading_type=='cyclic':

        if Z<=Z_cr:
            if Y <= 3: 
                P = P
            elif 3 <= Y <= 15:
                P = 0.72*(Z/Z_cr - 1)/(15.-3.) * Y[i] + 0.72*(15-3*Z/Z_cr)/(15.-3.)
            elif Y > 15:
                P = 0.72*Z/Z_cr

        else:
            if Y <= 3: 
                P = P
            elif Y>=3:
                P = 0.72
            
    #Un-normallized p-y curves
    p = P*p_ult
    y = Y*y_50
    
    k_secant = p/y
    
    #plot(y, p, 'x')
    #xlim([0., 1.1*y]), ylim([0., 1.1*p])
    
    return k_secant


def matlock_k_secant(z, D, Su, sigma_v_eff, Y_secant=0.01, z_0=0.0, epsilon_50=0.02, loading_type='static'):
    '''Returns secant modulus of the Matlock (1970) p-y curve at the depth of interest at the specified displacement. This
    value can be used to calculate the stiffness of the pile-soil response at very low displacement which in turn is needed
    to find the natural frequencies of a wind turbine attached to the said pile.
    
    
    Important: Make sure to import the interp1 function by running 'from scipy.interpolate import interp1d' in the main program.

    Input:
    -----
    z            - Depth relative to pile head (in)
    D            - Pile diameter (in)
    Su           - Undrained shear strength (psf)
    sigma_v_eff  - Effectve vertical stress (psf)
    Y_secant     - The lateral displacement at which the secant modulus is calculated normalized by D
                   Default: Y_secant = 0.01 -> y_secant = 0.01*D
    z_0          - Load eccentricity above the mudline or depth to mudline relative to the pile head (in)
    epsilon_50   - Strain at half the strength as defined by Matlock (1970).
                   Typically ranges from 0.005 (stiff clay) to 0.02 (soft clay).
    loading_type - Either 'static' or 'cyclic'
    
    Output:
    ------
    Returns the secant modulus of the p-y curve at the specified displacement 'y' in units of lb/in^2.
    ***VERY IMPORTANT: Make sure to convert the output to units of N/m^2 when using with functions in the 
                       'py_dynamic' module!!!
    
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
    y_secant = Y_secant*D
    Y = y_secant/y_50
    
    #Normalized depths
    Z    = z/D
    Z_cr = z_cr/D
    
    #Normalized p-y curves
    P = 0.5*Y**(1.0/3.0)  
    
    if P > 1.0:    P = 1.0
    
    if loading_type=='cyclic':

        if Z<=Z_cr:
            if Y <= 3: 
                P = P
            elif 3 <= Y <= 15:
                P = 0.72*(Z/Z_cr - 1)/(15.-3.) * Y[i] + 0.72*(15-3*Z/Z_cr)/(15.-3.)
            elif Y > 15:
                P = 0.72*Z/Z_cr

        else:
            if Y <= 3: 
                P = P
            elif Y>=3:
                P = 0.72
            
    #Un-normallized p-y curves
    p = P*p_ult
    y = Y*y_50
    
    k_secant = p/y

    return k_secant


def jeanjean_k_secant(z,D, Su, sigma_v_eff, Y_secant=0.01, Su_0=0.0, z_0=0.0, A=550, print_curves='No'):
    '''
    Returns an interp1d interpolation function which represents the Jeanjean (2009) p-y curve at the depth of interest.
    
    Important: Make sure to import the interp1 function by running 'from scipy.interpolate import interp1d' in the main program.

    Input:
    -----
    z            - Depth relative to pile head (in)
    D            - Pile diameter (in)
    Su           - Undrained shear strength (psf)
    sigma_v_eff  - Effectve vertical stress (psf)
    Y_secant     - The lateral displacement at which the secant modulus is calculated normalized by D
                   Default: Y_secant = 0.01 -> y_secant = 0.01*D
    Su_0         - Undrained shear strength at the mudline (psf)
    z_0          - Load eccentricity above the mudline or depth to mudline relative to the pile head (in)
    A            - G_max/Su (default = 550)
    
    Output:
    ------
    Returns 'k_secant' the secant stiffness of the p-y curves i.e. modulus of subgrade reaction (lb/in^2).
    ***VERY IMPORTANT: Make sure to convert the output to units of N/m^2 when using with functions in the 
                       'py_dynamic' module!!!
    '''
    
    from scipy.interpolate import interp1d
    
    #Change units to psi
    Su = Su/144.0  
    G_max = A*Su
    
    #Normalized p-y curve
    Y = Y_secant
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
    y = Y_secant*D
    
    k_secant = p/y
        
    return k_secant

def modified_matlock_k_secant(z, D, Su, sigma_v_eff, Y_secant=0.01, z_0=0.0, epsilon_50=0.02, loading_type='static', print_curves='No'):
    '''Returns an interp1d interpolation function which represents the Matlock (1970) with the range for Np updated from 
    3 - 9 to 8 - 12 as proposed by Jeanjean (2009). These p-y curves have been named 'Modified Matlock' curves.
    
    Important: Make sure to import the interp1 function by running 'from scipy.interpolate import interp1d' in the main program.

    Input:
    -----
    z            - Depth relative to pile head (in)
    D            - Pile diameter (in)
    Su           - Undrained shear strength (psf)
    sigma_v_eff  - Effectve vertical stress (psf)
    Y_secant     - The lateral displacement at which the secant modulus is calculated normalized by D
                   Default: Y_secant = 0.01 -> y_secant = 0.01*D
    z_0          - Load eccentricity above the mudline or depth to mudline relative to the pile head (in)
    epsilon_50   - Strain at half the strength as defined by Matlock (1970).
                   Typically ranges from 0.005 (stiff clay) to 0.02 (soft clay).
    loading_type - Either 'static' or 'cyclic'
    
    Output:
    ------
    Returns 'k_secant' the secant stiffness of the p-y curves i.e. modulus of subgrade reaction (lb/in^2).
    ***VERY IMPORTANT: Make sure to convert the output to units of N/m^2 when using with functions in the 
                       'py_dynamic' module!!!
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
    y_secant = Y_secant*D
    Y        = y_secant/y_50
    
    
    #Normalized depths
    Z    = z/D
    Z_cr = z_cr/D
    
    #Normalized p-y curves
    P = 0.5*Y**(1.0/3.0)
    
    if P > 1.0:    P = 1.0
    
    if loading_type=='cyclic':   

        if Z<=Z_cr:
            if Y <= 3: 
                P = P
            elif 3 <= Y <= 15:
                P = 0.72*(Z/Z_cr - 1)/(15-3) * Y + 0.72*(15-3*Z/Z_cr)/(15-3)
            elif Y > 15:
                P = 0.72*Z/Z_cr

        else:
            if Y <= 3: 
                P = P
            elif Y>=3:
                P = 0.72
            
    
    #Un-normallized p-y curves
    p = P*p_ult
    y = Y*y_50
    
    k_secant = p/y
    
    return k_secant


def kodikara_k_secant(z, D, Su, sigma_v_eff, Y_secant=0.01,z_0=0.0, alpha=0.0, R1=0.5, A=250, print_curves='No'):
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
    Y_secant     - The lateral displacement at which the secant modulus is calculated normalized by D
                   Default: Y_secant = 0.01 -> y_secant = 0.01*D
    z_0          - Load eccentricity above the mudline or depth to mudline relative to the pile head (in)
    A            - Shear modulus to undrained shear strength ratio, G/Su (default = 250)
    R1           - Tensile strength to undrained shear strength ratio, sigma_t = R1*Su (default=0.5)
    alpha        - Pile-soil adhesion in shear w.r.t. Su, 0 <= alpha <= 1 (alpha = 0.0 by default)
    
    Note: Pile-soil adhesion in tension is assumed to be equal to the tensile strengh of the soil
    
    Output:
    ------
    Returns 'k_secant' the secant stiffness of the p-y curves i.e. modulus of subgrade reaction (lb/in^2).
    ***VERY IMPORTANT: Make sure to convert the output to units of N/m^2 when using with functions in the 
                       'py_dynamic' module!!!
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

            z_cr  = (6.0 - sigma_v_eff/Su)*D/J  #This condition is implemented to avoid zero division errors.

        except ZeroDivisionError:
            print "Division by zero! Su = 0.0 so z_cr cannot be calculated."
    
        p_u = Su*N_p*D

        #p-y curves
        u = linspace(0,1,20)       #Dummy variable

        y_non_lin_range = (1-u)**2*y_e     + 2*u*(1-u)*(p_u/K_i) + u**2*y_u
        p_non_lin_range = (1-u)**2*K_i*y_e + 2*u*(1-u)*p_u       + u**2*p_u

        y = concatenate([array([min(-y_u,-5*D)]), -y_non_lin_range[::-1], array([0]), y_non_lin_range, array([5*D])])
        p = concatenate([array([-p_u]), -p_non_lin_range[::-1], array([0]), p_non_lin_range, array([p_u])])

    f = interp1d(y,p, kind='linear')
    k_secant = f(Y_secant*D)/(Y_secant*D)
    
    return k_secant


##################
### p-y curves ###
##################

def matlock_py_curves_SI(z, D, Su, sigma_v_eff, z_0=0.0, epsilon_50=0.02, loading_type='static', print_curves='No'):
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
    
    Output:
    ------
    Returns an interp1d interpolation function which represents the p-y curve at the depth of interest. 
    'p' (N/m) and 'y' (m).
    '''
    
    from scipy.interpolate import interp1d
    
    
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
    Y = linspace(-200,200,1000)
    
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
        plot(y,p), xlabel('y (m)'), ylabel('p (N/m)')
        grid(True)
        xlim([-2*D,2*D])
    
    return f


###########################
### Design Soil Profile ###
###########################

def design_soil_profile(soil_profile, plot_profile='No', y_axis_label='Depth below pile head (ft)'):
    '''Define the soil profile used by the p-y analyzer. Outputs 'interp1d' functions containing Su and sigma'_v 
    profiles to be used by the p-y curve functions.

    Note: Unlike the corresponding function in the 'py_method' module, the plots from this function show the depth
    in units of feet instead of inches. All other output is the same between this and 'py_method.design_soil_profile()'.
    
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
    y_axis_label - Label for y-axis
    
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
        #Plot Su vs z profile for confirmation
        plot(Su, depth/12., '-o')
        xlabel('Undrained shear strength (psf)'), ylabel(y_axis_label), grid(True)

        #Plot mudline/ground surface
        plot([-0.05*max(Su),max(Su)], [z0/12.,z0/12.], '--', color='brown')
        text(0.5*max(Su), 0.95*z0/12., 'Mudline', color='brown')
        
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


def design_soil_profile_SI(soil_profile, plot_profile='No'):
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
    
    #Depth of mudline relative to pile head
    z0 = soil_profile[0,0].astype(float)
    
    #Extract data from soil_profile array and zero strength virtual soil layer 
    #from the pile head down to the mudline
    depth     = concatenate([array([0,z0]), soil_profile[:,0].astype(float)])  #m
    Su        = concatenate([array([0, 0]), soil_profile[:,1].astype(float)])  #kPa
    gamma_sub = concatenate([array([0, 0]), soil_profile[:,2].astype(float)])  #kN/m^3
   
    if plot_profile=='Yes':
        #Plot Su vs z profile for confirmation
        plot(Su, depth, '-o', label=r'$S_u$')
        legend(loc='lower left')
        xlabel('Undrained shear strength (kPa)'), ylabel('Depth below pile head (m)'), grid(True)

        #Plot mudline/ground surface
        plot([-0.5*max(Su),max(Su)], [z0,z0], '--', color='brown')
        text(-0.5*max(Su), 0.95*z0, 'Mudline', color='brown')
        
        ax = gca()
        ax.invert_yaxis(), ax.xaxis.tick_top(), ax.xaxis.set_label_position('top')

    #Calculate sigma_v_eff at each depth
    sigma_v_eff = zeros(len(depth))
    
    for i in range(1,len(depth)):
        sigma_v_eff[i] = sigma_v_eff[i-1] + gamma_sub[i-1]*(depth[i]-depth[i-1])
    
    #Define interpolation functions
    f_Su          = interp1d(depth, Su*1000,          kind='linear') #Pa
    f_sigma_v_eff = interp1d(depth, sigma_v_eff*1000, kind='linear') #Pa
    
    return z0, f_Su, f_sigma_v_eff
