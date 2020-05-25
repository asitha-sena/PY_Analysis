import unittest
import numpy as np
import py_monotonic as pym


class Test_py_monotonic(unittest.TestCase):

    def setUp(self):
        """Field test from Matlock & Tucker (1961) is used as the test case to
        verify that the code is running correctly.
        """

        # Design soil profile
        self.soil_profile = np.array([[0.304, 9.58,  10., 'Matlock', 0.02],
                                      [15.23, 33.52, 10., 'Matlock', 0.02]])

        # Pile dimensions
        self.L = 516./39.4   # m, pile length (516 in, 43 ft)
        self.D = 12.75/39.4  # m, pile diameter (12.75 in)
        self.t = 0.5/39.4    # m, pile wall thickness (0.5 in)
        self.E = 210e9       # Pa, Elastic modulus of steel (210 GPa)

    def tearDown(self):
        pass

    def test_sigma_v_eff(self):
        z0, f_Su, f_sigma_v_eff = pym.design_soil_profile_SI(self.soil_profile)

        self.assertEqual(z0, 0.304)
        self.assertEqual(np.float(f_sigma_v_eff(10.0)), 96960.)

    def test_pile_head_load_calc(self):
        """The pile head load at 0.033m (~1.3 in) of lateral displacement is
        compared to the result of the field test.
        """

        z0, f_Su, f_sigma_v_eff = pym.design_soil_profile_SI(self.soil_profile)

        _, _, V0 = pym.py_analysis_2_SI(self.soil_profile, L=self.L, D=self.D,
                                        t=self.t, E=self.E, y_0=0.033, n=50,
                                        iterations=10,
                                        convergence_tracker='No',
                                        py_model='MM-1', print_output='No',
                                        epsilon_50=0.01)

        self.assertEqual(np.round(V0, 0), 51417)  # Result should be 51417 N

    def test_pile_head_disp_calc(self):
        """The pile head displament under a lateral load of 51.417 kN is
        compared to the result of the field test.
        """

        z0, f_Su, f_sigma_v_eff = pym.design_soil_profile_SI(self.soil_profile)

        y, _, = pym.py_analysis_1_SI(self.soil_profile, L=self.L, D=self.D,
                                     t=self.t, E=self.E, V_0=51417, n=50,
                                     iterations=10,
                                     convergence_tracker='No',
                                     py_model='MM-1', print_output='No',
                                     epsilon_50=0.01)

        self.assertEqual(np.round(y[0], 3), 0.033)  # Result should be 0.033 m


if __name__ == '__main__':
    unittest.main()
