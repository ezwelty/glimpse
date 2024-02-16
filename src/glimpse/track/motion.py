"""Model particle motion."""
import datetime
from typing import Iterable, Optional, Union

import numpy as np

from ..raster import Raster

Index = Union[slice, Iterable[int]]
Number = Union[int, float]


class Motion:
    """
    Illustration of the motion model interface required by :class:`Tracker`.

    :class:`Tracker` requires a motion model to have the following methods:

        - `initialize_particles()`: Initializes particles, typically around an
          initial position.
        - `evolve_particles(particles, dt)`: Evolves particles forward or backward
          in time.
        - `compute_log_likelihoods(particles)`: Computes particle log likelihoods
          (optional). If provided, these are added to the Observer log
          likelihoods computed by a `Tracker` to calculate the final particle
          likelihoods.

    This minimal example initializes all particles at the same position
    (x, y, 0) with velocity components in x, y, z drawn from normal
    distributions with zero mean. Particles are evolved based only on their
    initial velocities since accelerations are absent.

    Attributes:
        xy (iterable): Mean initial position (x, y).
        time_unit (timedelta): Length of time unit for temporal arguments.
        n (int): Number of particles.
        vxyz_sigma (iterable): Standard deviation of velocity
            (dx/dt, dy/dt, dz/dt) in `time_unit` time units.
    """

    def __init__(
        self,
        xy: Iterable[Number],
        time_unit: datetime.timedelta,
        n: int = 1000,
        vxyz_sigma: Iterable[Number] = (0, 0, 0),
    ) -> None:
        self.xy = xy
        self.time_unit = time_unit
        self.n = n
        self.vxyz_sigma = vxyz_sigma

    def initialize_particles(self) -> np.ndarray:
        """
        Initialize particles around an initial mean position.

        Returns:
            Particle positions and velocities (x, y, z, vx, vy, vz).
        """
        particles = np.zeros((self.n, 6), dtype=float)
        particles[:, 0:2] = self.xy
        particles[:, 3:6] = self.vxyz_sigma * np.random.randn(self.n, 3)
        return particles

    def evolve_particles(self, particles: np.ndarray, dt: datetime.timedelta) -> None:
        """
        Evolve particles through time by stochastic differentiation.

        Arguments:
            particles: Particle positions and velocities (x, y, z, vx, vy, vz).
            dt: Time step to evolve particles forward or backward.
        """
        time_units = dt.total_seconds() / self.time_unit.total_seconds()
        particles[:, 0:3] += time_units * particles[:, 3:6]

    def compute_log_likelihoods(self, particles: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute particle log likelihoods.

        If specified, these are added to the Observer log likelihoods computed
        by a :class:`Tracker` to calculate the final particle likelihoods.

        Arguments:
            particles: Particle positions and velocities (x, y, z, vx, vy, vz).

        Returns:
            Particle log likelihoods, or `None`.
        """
        return None


class CartesianMotion(Motion):
    """
    Evolves particles following a Cartesian motion model.

    Initial particle positions and velocities, and random accelerations, are
    specified by independent and normally distributed x, y, z components.
    Temporal arguments (e.g. :attr:`vxyz`, :attr:`axyz`) are assumed to be
    in :attr:`time_unit` time units.

    Particle heights (z) are initialized based on a mean surface (:attr:`dem`) and its
    associated uncertainty (:attr:`dem_sigma`), then evolved following the motion
    model. Particles are weighted based on their distance (dz) from the surface
    and its uncertainty.

    Attributes:
        xy (iterable): Mean initial position (x, y).
        time_unit (timedelta): Length of time unit for temporal arguments.
        dem: Elevation of the surface on which to track points.
        dem_sigma: Elevation standard deviations.
        n (int): Number of particles.
        xy_sigma (iterable): Standard deviation of initial position (x, y).
        vxyz (iterable): Mean initial velocity (dx/dt, dy/dt, dz/dt).
        vxyz_sigma (iterable): Standard deviation of initial velocity
            (dx/dt, dy/dt, dz/dt).
        axyz (iterable): Mean acceleration (d^2x/dt^2, d^2y/dt^2, d^2z/dt^2).
        axyz_sigma (iterable): Standard deviation of acceleration
            (d^2x/dt^2, d^2y/dt^2, d^2z/dt^2).
    """

    def __init__(
        self,
        xy: Iterable[Number],
        time_unit: datetime.timedelta,
        dem: Union[Number, Raster],
        dem_sigma: Union[Number, Raster] = None,
        n: int = 1000,
        xy_sigma: Iterable[Number] = (0, 0),
        vxyz: Iterable[Number] = (0, 0, 0),
        vxyz_sigma: Iterable[Number] = (0, 0, 0),
        axyz: Iterable[Number] = (0, 0, 0),
        axyz_sigma: Iterable[Number] = (0, 0, 0),
    ) -> None:
        self.xy = xy
        self.time_unit = time_unit
        if not isinstance(dem, Raster):
            dem = Raster(dem)
        self.dem = dem
        if not isinstance(dem_sigma, Raster):
            dem_sigma = Raster(dem_sigma)
        self.dem_sigma = dem_sigma
        self.n = n
        self.xy_sigma = xy_sigma
        self.vxyz = vxyz
        self.vxyz_sigma = vxyz_sigma
        self.axyz = axyz
        self.axyz_sigma = axyz_sigma

    def initialize_particles(self) -> np.ndarray:
        """
        Initialize particles around an initial mean position.

        Returns:
            particles: Particle positions and velocities (x, y, z, vx, vy, vz).
        """
        particles = np.zeros((self.n, 6), dtype=float)
        particles[:, 0:2] = self.xy + self.xy_sigma * np.random.randn(self.n, 2)
        particles[:, 2] = self.dem.sample(particles[:, 0:2])
        if self.dem_sigma is not None:
            z_sigma = self.dem_sigma.sample(particles[:, 0:2])
            particles[:, 2] += z_sigma * np.random.randn(self.n)
        particles[:, 3:6] = self.vxyz + self.vxyz_sigma * np.random.randn(self.n, 3)
        return particles

    def evolve_particles(self, particles: np.ndarray, dt: datetime.timedelta) -> None:
        """
        Evolve particles through time by stochastic differentiation.

        Arguments:
            particles: Particle positions and velocities (x, y, z, vx, vy, vz).
            dt: Time step to evolve particles forward or backward.
        """
        n = len(particles)
        time_units = dt.total_seconds() / self.time_unit.total_seconds()
        axyz = self.axyz + self.axyz_sigma * np.random.randn(n, 3)
        particles[:, 0:3] += (
            time_units * particles[:, 3:6] + 0.5 * axyz * time_units ** 2
        )
        particles[:, 3:6] += time_units * axyz

    def compute_log_likelihoods(self, particles: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute particle log likelihoods.

        Particles are weighted based on their distance from the mean surface
        (:attr:`dem`) and its associated uncertainty (:attr:`dem_sigma`).

        Arguments:
            particles: Particle positions and velocities (x, y, z, vx, vy, vz).

        Returns:
            Particle log likelihoods, or `None`.
        """
        if self.dem_sigma is None:
            return None
        z = self.dem.sample(particles[:, 0:2])
        z_sigma = self.dem_sigma.sample(particles[:, 0:2])
        # Avoid division by zero
        nonzero = np.nonzero(z_sigma)[0]
        log_likelihoods = np.zeros(len(particles), dtype=float)
        log_likelihoods[nonzero] = (
            1 / (2 * z_sigma[nonzero] ** 2) * (z[nonzero] - particles[nonzero, 2]) ** 2
        )
        return log_likelihoods


class CylindricalMotion(CartesianMotion):
    """
    Evolves particles following a cylindrical motion model.

    Identical to :class:`CartesianMotion`, except that particle motion is
    specified by independent and normally distributed components of magnitude
    (radius), direction (theta), and elevation (z). Angular arguments are
    assumed to be in radians counterclockwise from the +x axis.

    Attributes:
        xy (iterable): Mean initial position (x, y).
        time_unit (timedelta): Length of time unit for temporal arguments.
        dem: Elevation of the surface on which to track points.
        dem_sigma: Elevation standard deviations.
        n (int): Number of particles.
        xy_sigma (iterable): Standard deviation of initial position (x, y).
        vrthz (iterable): Mean initial velocity (d radius/dt, theta, dz/dt).
        vrthz_sigma (iterable): Standard deviation of initial velocity
            (d radius/dt, theta, dz/dt).
        arthz (iterable): Mean acceleration
            (d^2 radius/dt^2, d theta/dt, d^2z/dt^2).
        arthz_sigma (iterable): Standard deviation of acceleration
            (d^2 radius/dt^2, d theta/dt, d^2z/dt^2).
    """

    def __init__(
        self,
        xy: Iterable[Number],
        time_unit: datetime.timedelta,
        dem: Union[Number, Raster],
        dem_sigma: Union[Number, Raster] = None,
        n: int = 1000,
        xy_sigma: Iterable[Number] = (0, 0),
        vrthz: Iterable[Number] = (0, 0, 0),
        vrthz_sigma: Iterable[Number] = (0, 0, 0),
        arthz: Iterable[Number] = (0, 0, 0),
        arthz_sigma: Iterable[Number] = (0, 0, 0),
    ) -> None:
        self.xy = xy
        self.time_unit = time_unit
        if not isinstance(dem, Raster):
            dem = Raster(dem)
        self.dem = dem
        if not isinstance(dem_sigma, Raster):
            dem_sigma = Raster(dem_sigma)
        self.dem_sigma = dem_sigma
        self.n = n
        self.xy_sigma = xy_sigma
        self.vrthz = vrthz
        self.vrthz_sigma = vrthz_sigma
        self.arthz = arthz
        self.arthz_sigma = arthz_sigma

    def initialize_particles(self) -> np.ndarray:
        """
        Initialize particles around an initial mean position.

        Returns:
            particles: Particle positions and velocities (x, y, z, vx, vy, vz).
        """
        particles = np.zeros((self.n, 6), dtype=float)
        particles[:, 0:2] = self.xy + self.xy_sigma * np.random.randn(self.n, 2)
        particles[:, 2] = self.dem.sample(particles[:, 0:2])
        if self.dem_sigma is not None:
            z_sigma = self.dem_sigma.sample(particles[:, 0:2])
            particles[:, 2] += z_sigma * np.random.randn(self.n)
        v = self.vrthz + self.vrthz_sigma * np.random.randn(self.n, 3)
        particles[:, 3:6] = np.column_stack(
            (
                # r' * cos(th)
                v[:, 0] * np.cos(v[:, 1]),
                # r' * sin(th)
                v[:, 0] * np.sin(v[:, 1]),
                v[:, 2],
            )
        )
        return particles

    def evolve_particles(self, particles: np.ndarray, dt: datetime.timedelta) -> None:
        """
        Evolve particles through time by stochastic differentiation.

        Arguments:
            particles: Particle positions and velocities (x, y, z, vx, vy, vz).
            dt: Time step to evolve particles forward or backward.
        """
        n = len(particles)
        time_units = dt.total_seconds() / self.time_unit.total_seconds()
        vx = particles[:, 3]
        vy = particles[:, 4]
        vr = np.sqrt(vx ** 2 + vy ** 2)
        arthz = self.arthz + self.arthz_sigma * np.random.randn(n, 3)
        axyz = np.column_stack(
            (
                # r'' * cos(th) - r' * sin(th) * th'
                arthz[:, 0] * (vx / vr) - vy * arthz[:, 1],
                # r'' * sin(th) - r' * cos(th) * th'
                arthz[:, 0] * (vy / vr) + vx * arthz[:, 1],
                arthz[:, 2],
            )
        )
        particles[:, 0:3] += (
            time_units * particles[:, 3:6] + 0.5 * axyz * time_units ** 2
        )
        particles[:, 3:6] += time_units * axyz


class TangentCartesianMotion(Motion):
    """
    Evolves particles tangent to a mean surface.

    Initial particle positions and velocities, and random accelerations, are
    specified by independent and normally distributed x and y components.
    Temporal arguments (e.g. :attr:`vxy`, :attr:`axy`) are assumed to be
    in :attr:`time_unit` time units.

    Particle heights (z) are initialized based on a mean surface (:attr:`dem`)
    and its uncertainty (:attr:`dem_sigma`), then maintain the same relative
    height adjusted by a random walk proportional to the particle's horizontal
    distance and a characteristic slope of small-scale features
    (:attr:`slope_sigma`).

    This is the model used in Brinkerhoff (2017): Bayesian methods in glaciology
    (http://hdl.handle.net/11122/8113), chapter 4.

    Attributes:
        xy (iterable): Mean initial position (x, y).
        time_unit (timedelta): Length of time unit for temporal arguments.
        dem: Elevation of the surface on which to track points.
        dem_sigma: Elevation standard deviations.
        n (int): Number of particles.
        xy_sigma (iterable): Standard deviation of initial position (x, y).
        vxy (iterable): Mean initial velocity (dx/dt, dy/dt).
        vxy_sigma (iterable): Standard deviation of initial velocity
            (dx/dt, dy/dt).
        axy (iterable): Mean acceleration (d^2x/dt^2, d^2y/dt^2).
        axy_sigma (iterable): Standard deviation of acceleration
            (d^2x/dt^2, d^2y/dt^2).
        slope_sigma (float): Standard deviation of the characteristic slope
            of small-scale features.
    """

    def __init__(
        self,
        xy: Iterable[Number],
        time_unit: datetime.timedelta,
        dem: Union[Number, Raster],
        dem_sigma: Union[Number, Raster] = 0,
        n: int = 1000,
        xy_sigma: Iterable[Number] = (0, 0),
        vxy: Iterable[Number] = (0, 0),
        vxy_sigma: Iterable[Number] = (0, 0),
        axy: Iterable[Number] = (0, 0),
        axy_sigma: Iterable[Number] = (0, 0),
        slope_sigma: Number = 0,
    ) -> None:
        self.xy = xy
        self.time_unit = time_unit
        if not isinstance(dem, Raster):
            dem = Raster(dem)
        self.dem = dem
        if not isinstance(dem_sigma, Raster):
            dem_sigma = Raster(dem_sigma)
        self.dem_sigma = dem_sigma
        self.n = n
        self.xy_sigma = xy_sigma
        self.vxy = vxy
        self.vxy_sigma = vxy_sigma
        self.axy = axy
        self.axy_sigma = axy_sigma
        self.slope_sigma = slope_sigma

    def initialize_particles(self) -> np.ndarray:
        """
        Initialize particles around an initial mean position.

        Returns:
            particles: Particle positions and velocities (x, y, z, vx, vy, vz).
        """
        particles = np.zeros((self.n, 6), dtype=float)
        particles[:, 0:2] = self.xy + self.xy_sigma * np.random.randn(self.n, 2)
        z_offsets = self.dem_sigma.sample(particles[:, 0:2]) * np.random.randn(self.n)
        particles[:, 2] = self.dem.sample(particles[:, 0:2]) + z_offsets
        particles[:, 3:5] = self.vxy + self.vxy_sigma * np.random.randn(self.n, 2)
        return particles

    def evolve_particles(self, particles: np.ndarray, dt: datetime.timedelta) -> None:
        """
        Evolve particles through time by stochastic differentiation.

        Arguments:
            particles: Particle positions and velocities (x, y, z, vx, vy, vz).
            dt: Time step to evolve particles forward or backward.
        """
        n = len(particles)
        time_units = dt.total_seconds() / self.time_unit.total_seconds()
        axy = self.axy + self.axy_sigma * np.random.randn(n, 2)
        dxy = time_units * particles[:, 3:5] + 0.5 * axy * time_units ** 2
        # HACK: Recover z_offsets (since particles may have been resampled)
        z_offsets = particles[:, 2] - self.dem.sample(particles[:, 0:2])
        z_offsets += (
            self.slope_sigma * np.random.randn(n) * (dxy ** 2).sum(axis=1) ** 0.5
        )
        particles[:, 0:2] += dxy
        particles[:, 2] = self.dem.sample(particles[:, 0:2]) + z_offsets
        particles[:, 3:5] += time_units * axy


class TangentCylindricalMotion(Motion):
    """
    Evolves particles tangent to a mean surface using cylindrical motion.

    Identical to :class:`TangentCartesianMotion`, except that particle motion is
    specified by independent and normally distributed components of magnitude
    (radius) and direction (theta). Angular arguments are
    assumed to be in radians counterclockwise from the +x axis.

    Attributes:
        xy (iterable): Mean initial position (x, y).
        time_unit (timedelta): Length of time unit for temporal arguments.
        dem: Elevation of the surface on which to track points.
        dem_sigma: Elevation standard deviations.
        n (int): Number of particles.
        xy_sigma (iterable): Standard deviation of initial position (x, y).
        vrth (iterable): Mean initial velocity (d radius/dt, theta).
        vrth_sigma (iterable): Standard deviation of initial velocity
            (d radius/dt, theta).
        arth (iterable): Mean acceleration
            (d^2 radius/dt^2, d theta/dt).
        arth_sigma (iterable): Standard deviation of acceleration
            (d^2 radius/dt^2, d theta/dt).
    """

    def __init__(
        self,
        xy: Iterable[Number],
        time_unit: datetime.timedelta,
        dem: Union[Number, Raster],
        dem_sigma: Union[Number, Raster] = None,
        n: int = 1000,
        xy_sigma: Iterable[Number] = (0, 0),
        vrth: Iterable[Number] = (0, 0),
        vrth_sigma: Iterable[Number] = (0, 0),
        arth: Iterable[Number] = (0, 0),
        arth_sigma: Iterable[Number] = (0, 0),
        slope_sigma: Number = 0,
    ) -> None:
        self.xy = xy
        self.time_unit = time_unit
        if not isinstance(dem, Raster):
            dem = Raster(dem)
        self.dem = dem
        if not isinstance(dem_sigma, Raster):
            dem_sigma = Raster(dem_sigma)
        self.dem_sigma = dem_sigma
        self.n = n
        self.xy_sigma = xy_sigma
        self.vrth = vrth
        self.vrth_sigma = vrth_sigma
        self.arth = arth
        self.arth_sigma = arth_sigma
        self.slope_sigma = slope_sigma

    def initialize_particles(self) -> np.ndarray:
        """
        Initialize particles around an initial mean position.

        Returns:
            particles: Particle positions and velocities (x, y, z, vx, vy, vz).
        """
        particles = np.zeros((self.n, 6), dtype=float)
        particles[:, 0:2] = self.xy + self.xy_sigma * np.random.randn(self.n, 2)
        z_offsets = self.dem_sigma.sample(particles[:, 0:2]) * np.random.randn(self.n)
        particles[:, 2] = self.dem.sample(particles[:, 0:2]) + z_offsets
        vrth = self.vrth + self.vrth_sigma * np.random.randn(self.n, 2)
        particles[:, 3:5] = np.column_stack(
            (
                # r' * cos(th)
                vrth[:, 0] * np.cos(vrth[:, 1]),
                # r' * sin(th)
                vrth[:, 0] * np.sin(vrth[:, 1]),
            )
        )
        return particles

    def evolve_particles(self, particles: np.ndarray, dt: datetime.timedelta) -> None:
        """
        Evolve particles through time by stochastic differentiation.

        Arguments:
            particles: Particle positions and velocities (x, y, z, vx, vy, vz).
            dt: Time step to evolve particles forward or backward.
        """
        n = len(particles)
        time_units = dt.total_seconds() / self.time_unit.total_seconds()
        vx = particles[:, 3]
        vy = particles[:, 4]
        vr = np.sqrt(vx ** 2 + vy ** 2)
        arth = self.arth + self.arth_sigma * np.random.randn(n, 2)
        axy = np.column_stack(
            (
                # r'' * cos(th) - r' * sin(th) * th'
                arth[:, 0] * (vx / vr) - vy * arth[:, 1],
                # r'' * sin(th) - r' * cos(th) * th'
                arth[:, 0] * (vy / vr) + vx * arth[:, 1],
            )
        )
        dxy = time_units * particles[:, 3:5] + 0.5 * axy * time_units ** 2
        # HACK: Recover z_offsets (since particles may have been resampled)
        z_offsets = particles[:, 2] - self.dem.sample(particles[:, 0:2])
        z_offsets += (
            self.slope_sigma * np.random.randn(n) * (dxy ** 2).sum(axis=1) ** 0.5
        )
        particles[:, 0:2] += dxy
        particles[:, 2] = self.dem.sample(particles[:, 0:2]) + z_offsets
        particles[:, 3:5] += time_units * axy
