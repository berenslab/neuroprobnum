from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import RK23, RK45


class ButcherTableau:

    def __init__(self, method):

        """Create Butcher Tableau for method.
        method (str) : keyword for method, e.g. RK45"""

        self.method = method

        # Set RK type.
        # Note: E = Bstar - B, Bstar = E + B
        if method == 'FE':
            # Solution: Forward/Explicit Euler (1st)
            # Error: Heun (2nd)
            # https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Embedded_methods

            self.C = np.array([0])
            self.A = np.array([[0]])
            self.B = np.array([1])
            self.Bstar = np.array([0.5, 0.5])
            self.E = np.array([-0.5, 0.5])
            self.P = np.array([[1.]])
            self.n_stages = 2
            self.err_ex = -1 / 2
            self.order_B = 1
            self.order_Bstar = 2

        elif method in 'HN':
            # Solution: Heun (2nd)
            # Error: Forward/Explicit Euler (1st)
            # https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Embedded_methods

            self.C = np.array([0, 1])
            self.A = np.array([[0, 0], [1, 0]])
            self.B = np.array([0.5, 0.5])
            self.Bstar = np.array([1, 0])
            self.E = np.array([0.5, -0.5])
            self.P = np.array([[1, -0.5], [0, 0.5]])
            self.n_stages = 2
            self.err_ex = -1 / 2
            self.order_B = 2
            self.order_Bstar = 1

        elif method == 'RKBS':
            # Third order solution, second order error estimate.
            # https://en.wikipedia.org/wiki/Bogacki%E2%80%93Shampine_method

            rk = RK23(fun=lambda t, y: 0, t0=0, y0=np.array([0]), t_bound=0)

            self.C = rk.C
            self.A = rk.A
            self.B = rk.B  # 3rd order
            self.E = rk.E
            self.P = rk.P
            self.Bstar = np.array([7 / 24, 1 / 4, 1 / 3, 1 / 8])  # 2nd order
            self.n_stages = 4
            self.err_ex = -1 / 3
            self.order_B = 3
            self.order_Bstar = 2

        elif method == 'RKCK':
            # Fourth order solution, fifth order error estimate.
            # https://en.wikipedia.org/wiki/Cash%E2%80%93Karp_method

            self.C = np.array([0, 1 / 5, 3 / 10, 3 / 5, 1, 7 / 8])
            self.A = np.array([
                [0, 0, 0, 0, 0],
                [1 / 5, 0, 0, 0, 0],
                [3 / 40, 9 / 40, 0, 0, 0],
                [3 / 10, -9 / 10, 6 / 5, 0, 0],
                [-11 / 54, 5 / 2, -70 / 27, 35 / 27, 0],
                [1631 / 55296, 175 / 512, 575 / 13824, 44275 / 110592, 253 / 4096],
            ])
            self.B = np.array([2825 / 27648, 0, 18575 / 48384, 13525 / 55296, 277 / 14336, 1 / 4])  # 4th
            self.Bstar = np.array([37 / 378, 0, 250 / 621, 125 / 594, 0, 512 / 1771])  # 5th
            self.E = None
            self.n_stages = 6
            self.err_ex = -1 / 4
            self.order_B = 4
            self.order_Bstar = 5

        elif method == 'RKDP':
            # Fifth order solution, fourth order error estimate.
            # https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method

            rk = RK45(fun=lambda t, y: 0, t0=0, y0=np.array([0]), t_bound=0)

            self.C = rk.C
            self.A = rk.A
            self.B = rk.B
            self.E = rk.E
            self.P = rk.P
            self.Bstar = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])
            self.n_stages = 7
            self.err_ex = -1 / 5
            self.order_B = 5
            self.order_Bstar = 4

        else:
            raise NotImplementedError(method)

        self.complete_E_and_Bstar()

    def complete_E_and_Bstar(self):
        """If Bstar is given, either check given E or compute new E.
        If E is given and not Bstar, compute Bstar.
        E = Bstar - B, Bstar = E + B"""

        if self.Bstar is not None:
            Bstar_ = np.pad(self.Bstar, (0, np.max([self.Bstar.size, self.B.size]) - self.Bstar.size))
            B_ = np.pad(self.B, (0, np.max([self.Bstar.size, self.B.size]) - self.B.size))
            if self.E is not None:
                assert np.allclose(self.E, (Bstar_ - B_))
            else:
                self.E = Bstar_ - B_

        # E given but not Bstar. Compute Bstar from E and B.
        elif self.E is not None:
            E_ = np.pad(self.E, (0, np.max([self.E.size, self.B.size]) - self.E.size))
            B_ = np.pad(self.B, (0, np.max([self.E.size, self.B.size]) - self.B.size))
            self.Bstar = E_ + B_

    def get_tableau(self):
        """Return tableau.
        Returns:
        tableau (dict) : Butcher tableau as a dict.
        """

        tableau = dict()
        tableau['A'] = self.A.copy()
        tableau['B'] = self.B.copy()
        tableau['C'] = self.C.copy()
        tableau['n_stages'] = self.n_stages

        # Optional values.
        tableau['Bstar'] = self.Bstar.copy()
        tableau['E'] = self.E.copy()
        tableau['P'] = self.P if hasattr(self, "P") else None  # For dense output
        tableau['err_ex'] = self.err_ex
        tableau['order_B'] = self.order_B
        tableau['order_Bstar'] = self.order_Bstar

        return tableau

    def plot(self):
        """Plot tableau."""

        plt.figure(1, (12, 3))

        plt.subplot(131)
        plt.title('C')
        plt.plot(self.C, '.-')
        plt.xticks(np.arange(0, self.C.size))

        plt.subplot(132)
        plt.title('B')
        plt.plot(self.B, '.-', label='B')
        plt.plot(self.Bstar, '.-', label='B*')
        plt.legend()
        plt.xticks(np.arange(0, np.max([self.B.size, self.Bstar.size])))

        plt.subplot(133)
        plt.title('A')
        plt.imshow(self.A, origin='bottom')
        plt.colorbar()
        plt.xticks(np.arange(0, self.A.shape[0]))
        plt.yticks(np.arange(0, self.A.shape[0]))

        plt.show()
