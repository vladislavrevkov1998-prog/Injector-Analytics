"""Axisymmetric irrotational carrier closure on a profiled annular duct.

This is a prescribed one-way carrier, not a turbulent-jet, RANS, gas mass, or
two-phase momentum solution. It solves the Stokes streamfunction equation

    div((1/r) grad(psi)) = 0,  uz = psi_r/r,  ur = -psi_z/r.

Coordinates passed to this module are millimetres; velocities are m/s and psi
is m^3/s. The total volume flux is 2*pi*(psi_outer-psi_inner). P1 triangles on
the chord approximation have exactly constant wall psi, continuous normal
volume flux across element edges, and identically zero axisymmetric divergence
inside each triangle. Physical circular walls remain available for contact
tests elsewhere. Evaluation in the small exact-wall/chord-mesh sliver extends
the boundary element velocity; it never moves a particle or admits a particle
outside the exact chamber. Exact-wall evaluations use an impermeable tangent
trace. A cylindrical duct uses its exact uniform-flow solution.
"""
from dataclasses import dataclass, field
from functools import lru_cache
import json
import math
from types import MappingProxyType
import warnings

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from .chamber import Chamber, TOL, number


MAX_NODES = 80000
MAX_RADIAL_CELLS = 128


def _readonly(value, dtype=float):
    result = np.asarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _canonical_geometry(model):
    primitives = []
    for wall in ('inner', 'outer'):
        for primitive in model.profiles[wall]:
            record = dict(id=f'{wall}{len(primitives)}', kind=primitive.kind,
                          wall=wall, unit='mm', start=primitive.start.tolist(),
                          end=primitive.end.tolist())
            if primitive.kind == 'arc':
                record.update(center=primitive.center.tolist(),
                              clockwise=primitive.raw.get('clockwise', False))
            primitives.append(record)
    return dict(schema_version=1, enabled=True, base_z_mm=model.base,
                outlet_z_mm=model.outlet, primitives=primitives, objects=[],
                groups=[], sections=[dict(id='outlet', z_mm=model.outlet,
                                         is_outlet=True)])


def _stations(model, step, tolerance, radial_cells):
    ratio = (model.outlet-model.base)/step
    if not math.isfinite(ratio) or ratio > MAX_NODES:
        raise ValueError('Несущий поток: сетка превышает 80000 узлов; увеличьте осевой шаг')
    uniform_count = math.ceil(ratio)
    if (uniform_count+1)*(radial_cells+1) > MAX_NODES:
        raise ValueError('Несущий поток: сетка превышает 80000 узлов; увеличьте осевой шаг')
    # Anchor the step to the physical inlet. Extending a cylindrical fallback
    # with a farther output plane must not rescale all preceding integration
    # steps (or the seeded OU history).
    candidates = [*(model.base+np.arange(uniform_count)*step),model.outlet]
    for wall in ('inner', 'outer'):
        for primitive in model.profiles[wall]:
            candidates.extend(primitive.discretize(tolerance)[:, 0])
    values = np.sort(np.asarray(candidates))
    z = values[np.r_[True, np.diff(values) > TOL]]
    z[0], z[-1] = model.base, model.outlet
    if len(z)*(radial_cells+1) > MAX_NODES:
        raise ValueError('Несущий поток: профиль и допуск требуют более 80000 узлов')
    return z


def _wall_values(model, z):
    """Vectorized exact walls, including primitive joints and vertical tangents."""
    result = []
    for wall in ('inner', 'outer'):
        radii = np.full(np.shape(z), np.nan, dtype=float)
        for primitive in model.profiles[wall]:
            mask = (z >= primitive.lo-TOL) & (z <= primitive.hi+TOL)
            if np.any(mask):
                radii[mask] = primitive.at(np.clip(z[mask], primitive.lo, primitive.hi))
        if not np.isfinite(radii).all():
            raise ValueError('Несущий поток: координата вне покрытого профилем интервала')
        result.append(radii)
    return tuple(result)


@dataclass(frozen=True, eq=False)
class Carrier:
    """Immutable, cacheable streamfunction field; all public arrays are readonly."""
    z_mm: np.ndarray
    r_mm: np.ndarray
    eta: np.ndarray
    streamfunction: np.ndarray
    triangles: np.ndarray
    inlet_velocity_m_s: float
    inlet_flux_m3_s: float
    weak_residual: float
    wall_tolerance_mm: float
    diagnostics: object
    _model: object = field(repr=False)
    _coordinates_m: np.ndarray = field(repr=False)
    _basis_gradient: np.ndarray = field(repr=False)
    _psi_gradient: np.ndarray = field(repr=False)
    _cylindrical: bool = field(repr=False)

    def _arguments(self, z_mm, r_mm):
        z, r = np.broadcast_arrays(np.asarray(z_mm, dtype=float),
                                   np.asarray(r_mm, dtype=float))
        shape = z.shape
        z, r = z.ravel(), r.ravel()
        if not np.isfinite(z).all() or not np.isfinite(r).all():
            raise ValueError('Несущий поток: координаты должны быть конечными')
        if np.any(z < self.z_mm[0]-TOL) or np.any(z > self.z_mm[-1]+TOL):
            raise ValueError('Несущий поток: z вне жаровой трубы')
        z = np.clip(z, self.z_mm[0], self.z_mm[-1])
        inner, outer = _wall_values(self._model, z)
        if np.any(r < inner-TOL) or np.any(r > outer+TOL):
            raise ValueError('Несущий поток: точка вне точных стенок жаровой трубы')
        # Only tolerate floating-point error at the exact wall, never a finite
        # outside fuel position. The separate mesh lookup extends by its chord
        # error into a physically valid near-wall sliver.
        r = np.minimum(np.maximum(r, inner), outer)
        return z, r, inner, outer, shape

    def mesh_bounds(self, z_mm):
        z = np.asarray(z_mm, dtype=float)
        if (not np.isfinite(z).all() or np.any(z < self.z_mm[0]-TOL)
                or np.any(z > self.z_mm[-1]+TOL)):
            raise ValueError('Несущий поток: z вне сетки')
        return (np.interp(z, self.z_mm, self.r_mm[:, 0]),
                np.interp(z, self.z_mm, self.r_mm[:, -1]))

    def _locate(self, z, r):
        axial = np.clip(np.searchsorted(self.z_mm, z, side='right')-1,
                        0, len(self.z_mm)-2)
        fraction = (z-self.z_mm[axial])/(self.z_mm[axial+1]-self.z_mm[axial])
        lower = (1-fraction)*self.r_mm[axial, 0]+fraction*self.r_mm[axial+1, 0]
        upper = (1-fraction)*self.r_mm[axial, -1]+fraction*self.r_mm[axial+1, -1]
        # The arc discretization bounds the *normal geometric distance* to a
        # chord, not its radial gap at a fixed z. Near a vertical arc tangent
        # a tiny normal gap can have a much larger radial component. Compare
        # against the nearest point of that actual chord segment instead.
        outside = (r < lower) | (r > upper)
        if np.any(outside):
            indices = np.flatnonzero(outside)
            side = np.where(r[indices] < lower[indices], 0, -1)
            cell = axial[indices]
            start = np.column_stack((self.z_mm[cell], self.r_mm[cell, side]))
            end = np.column_stack((self.z_mm[cell+1], self.r_mm[cell+1, side]))
            segment = end-start
            position = np.column_stack((z[indices], r[indices]))
            along = np.clip(np.einsum('ij,ij->i', position-start, segment)
                            / np.einsum('ij,ij->i', segment, segment), 0., 1.)
            distance = np.linalg.norm(position-(start+along[:, None]*segment), axis=1)
            if np.any(distance > self.wall_tolerance_mm+4*TOL):
                raise ValueError('Несущий поток: точка вне допуска аппроксимации стенки')
        lookup_r = np.minimum(np.maximum(r, lower), upper)
        count = self.r_mm.shape[1]-1
        radial = np.clip(np.floor((lookup_r-lower)/(upper-lower)*count).astype(int),
                         0, count-1)
        diagonal = ((1-fraction)*self.r_mm[axial, radial]
                    + fraction*self.r_mm[axial+1, radial+1])
        element = 2*(axial*count+radial)+(lookup_r > diagonal)
        return element.astype(int), lookup_r

    def streamfunction_at(self, z_mm, r_mm):
        z, r, _, _, shape = self._arguments(z_mm, r_mm)
        if self._cylindrical:
            value = .5*self.inlet_velocity_m_s*((r*.001)**2
                                                -(self.r_mm[0, 0]*.001)**2)
        else:
            elements, lookup_r = self._locate(z, r)
            vertex = self.triangles[elements, 0]
            delta = np.column_stack((z*.001, lookup_r*.001))-self._coordinates_m[vertex]
            value = (self.streamfunction.ravel()[vertex]
                     + np.einsum('ij,ij->i', self._psi_gradient[elements], delta))
        return value.reshape(shape)

    def meridional_velocity(self, z_mm, r_mm):
        """Return [..., uz, ur] in m/s; no azimuthal/swirl component is modeled."""
        z, r, inner, outer, shape = self._arguments(z_mm, r_mm)
        if self._cylindrical:
            velocity = np.column_stack((np.full_like(z, self.inlet_velocity_m_s),
                                        np.zeros_like(z)))
        else:
            elements, _ = self._locate(z, r)
            gradient = self._psi_gradient[elements]
            velocity = np.column_stack((gradient[:, 1], -gradient[:, 0]))/(r*.001)[:, None]
            # The FEM boundary is a chord approximation. On the *exact* curved
            # wall use its impermeable tangent trace, avoiding an artificial
            # normal velocity solely from the chord/arc difference.
            for wall, radii in (('inner', inner), ('outer', outer)):
                mask = np.abs(r-radii) <= TOL
                if not np.any(mask):continue
                for primitive in self._model.profiles[wall]:
                    selected = mask & (z >= primitive.lo-TOL) & (z <= primitive.hi+TOL)
                    indices = np.flatnonzero(selected)
                    if not len(indices):continue
                    if primitive.kind == 'line':
                        normal = primitive.normal(float(np.clip(z[indices[0]], primitive.lo, primitive.hi)))
                        velocity[indices] -= np.outer(velocity[indices]@normal, normal)
                    else:
                        normals = np.column_stack((z[indices]-primitive.center[0],
                                                   r[indices]-primitive.center[1]))
                        normals /= np.linalg.norm(normals, axis=1)[:, None]
                        velocity[indices] -= np.einsum('ij,ij->i', velocity[indices], normals)[:, None]*normals
        return velocity.reshape(shape+(2,))

    def velocity(self, z_mm, x_mm, y_mm):
        """Return [..., vz, vx, vy] in Cartesian coordinates, in that order."""
        z, x, y = np.broadcast_arrays(np.asarray(z_mm, dtype=float),
                                      np.asarray(x_mm, dtype=float),
                                      np.asarray(y_mm, dtype=float))
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Несущий поток: координаты должны быть конечными')
        radius = np.hypot(x, y)
        meridional = self.meridional_velocity(z, radius)
        return np.stack((meridional[..., 0], meridional[..., 1]*x/radius,
                         meridional[..., 1]*y/radius), axis=-1)


@lru_cache(maxsize=6)
def _build(canonical, inlet_velocity, step, radial_cells, tolerance):
    model = Chamber(json.loads(canonical))
    z = _stations(model, step, tolerance, radial_cells)
    eta = np.linspace(0., 1., radial_cells+1)
    inner, outer = _wall_values(model, z)
    r = inner[:, None]+(outer-inner)[:, None]*eta
    zz = np.broadcast_to(z[:, None], r.shape)
    coordinates = np.column_stack((zz.ravel(), r.ravel()))*.001
    stride = radial_cells+1
    lower_left = (np.arange(len(z)-1)[:, None]*stride
                  + np.arange(radial_cells)[None, :]).ravel()
    triangles = np.stack((np.column_stack((lower_left, lower_left+stride, lower_left+stride+1)),
                          np.column_stack((lower_left, lower_left+stride+1, lower_left+1))), axis=1).reshape(-1, 3)
    points = coordinates[triangles]
    a = points[:, 1]-points[:, 0]
    b = points[:, 2]-points[:, 0]
    determinant = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
    if np.any(determinant <= 0) or not np.isfinite(determinant).all():
        raise ValueError('Несущий поток: вырожденная/перевёрнутая ячейка сетки')
    gradients = np.empty((len(triangles), 3, 2))
    gradients[:, 0] = np.column_stack((points[:, 1, 1]-points[:, 2, 1], points[:, 2, 0]-points[:, 1, 0]))
    gradients[:, 1] = np.column_stack((points[:, 2, 1]-points[:, 0, 1], points[:, 0, 0]-points[:, 2, 0]))
    gradients[:, 2] = np.column_stack((points[:, 0, 1]-points[:, 1, 1], points[:, 1, 0]-points[:, 0, 0]))
    gradients /= determinant[:, None, None]
    inlet_inner, inlet_outer = r[0, 0]*.001, r[0, -1]*.001
    outer_psi = .5*inlet_velocity*(inlet_outer**2-inlet_inner**2)
    if not math.isfinite(outer_psi) or outer_psi <= 0:
        raise ValueError('Несущий поток: объёмный расход выходит за численный диапазон')
    cylindrical = (np.ptp(inner) <= TOL and np.ptp(outer) <= TOL)
    if cylindrical:
        psi = .5*inlet_velocity*((r*.001)**2-inlet_inner**2)
        if not np.isfinite(psi).all():
            raise ValueError('Несущий поток: решение выходит за численный диапазон')
        residual = 0.
    else:
        # Three-point barycentric quadrature for the positive coefficient 1/r.
        # The gradients are constant per P1 triangle; no nodal radial squeezing
        # or prescribed velocity interpolation replaces this elliptic solve.
        barycentric = np.full((3, 3), 1/6)
        np.fill_diagonal(barycentric, 2/3)
        radii_q = points[:, :, 1]@barycentric.T
        integrated_coefficient = determinant*.5*np.mean(1/radii_q, axis=1)
        stiffness = integrated_coefficient[:, None, None]*np.einsum('tik,tjk->tij', gradients, gradients)
        rows = np.broadcast_to(triangles[:, :, None], stiffness.shape).ravel()
        columns = np.broadcast_to(triangles[:, None, :], stiffness.shape).ravel()
        matrix = coo_matrix((stiffness.ravel(), (rows, columns)),
                            shape=(len(coordinates), len(coordinates))).tocsr()
        fixed = np.zeros(r.shape, dtype=bool)
        fixed[:, 0] = True;fixed[:, -1] = True;fixed[0] = True
        psi = np.zeros(r.shape)
        psi[:, -1] = outer_psi
        psi[0] = .5*inlet_velocity*((r[0]*.001)**2-inlet_inner**2)
        known = fixed.ravel();free = ~known
        right = -(matrix[free][:, known]@psi.ravel()[known])
        with warnings.catch_warnings():
            warnings.simplefilter('error', MatrixRankWarning)
            try:psi.ravel()[free] = spsolve(matrix[free][:, free], right)
            except (MatrixRankWarning, RuntimeError, ValueError) as exc:
                raise ValueError('Несущий поток: не удалось решить систему функции тока') from exc
        if not np.isfinite(psi).all():
            raise ValueError('Несущий поток: решение содержит нечисловые значения')
        # A dimensionless free-node algebraic residual. It is not an empirical
        # error bar, a turbulent closure error, or a pointwise PDE residual.
        algebraic = matrix@psi.ravel()
        denominator = max(float(np.max(np.abs(matrix.data)))*abs(outer_psi), 1e-300)
        residual = float(np.max(np.abs(algebraic[free])))/denominator
        if residual > 1e-9:
            raise ValueError('Несущий поток: слишком большая невязка системы')
    psi_gradient = np.einsum('ti,tij->tj', psi.ravel()[triangles], gradients)
    # Independent discrete side-wall check: normal flux is zero on every
    # straight mesh edge because both endpoint values of psi are identical.
    normals = [];velocities = []
    for wall_index, side in ((0, 0), (radial_cells-1, -1)):
        cell = 2*(np.arange(len(z)-1)*radial_cells+wall_index)+(1 if side == -1 else 0)
        tangent = np.column_stack((np.diff(z), np.diff(r[:, side])))
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        radius = .5*(r[:-1, side]+r[1:, side])*.001
        gradient = psi_gradient[cell]
        velocity = np.column_stack((gradient[:, 1], -gradient[:, 0]))/radius[:, None]
        if cylindrical:velocity[:] = (inlet_velocity, 0.)
        normals.append(normal);velocities.append(velocity)
    wall_normal = float(np.max(np.abs(np.einsum('ij,ij->i', np.vstack(normals), np.vstack(velocities)))))
    diagnostics = MappingProxyType(dict(
        model='axisymmetric_irrotational_one_way_carrier',
        solution_kind='analytic_cylindrical' if cylindrical else 'P1_streamfunction_FEM',
        node_count=int(r.size), triangle_count=int(len(triangles)),
        max_discrete_wall_normal_velocity_m_s=wall_normal,
        algebraic_relative_residual=residual,
        mesh_wall_tolerance_mm=tolerance,
        carrier_flux_m3_s=2*math.pi*outer_psi,
        interpretation='Prescribed carrier closure; no turbulence, separation or gas backreaction solved'))
    return Carrier(_readonly(z), _readonly(r), _readonly(eta), _readonly(psi),
                   _readonly(triangles, np.int64), inlet_velocity,
                   2*math.pi*outer_psi, residual, tolerance, diagnostics, model,
                   _readonly(coordinates), _readonly(gradients),
                   _readonly(psi_gradient), cylindrical)


def build_carrier(geometry_dict, inlet_velocity_m_s, max_step_mm=2.,
                  radial_cells=16, wall_tolerance_mm=.02):
    """Build/cache an explicitly parameterized conservative carrier closure.

    ``inlet_velocity_m_s`` is supplied, not inferred from fuel flow or pressure.
    Output planes and their radial bins do not determine this internal mesh.
    Unknown metadata, fixed flags and constraint IDs do not affect its cache.
    """
    if any(isinstance(value, (bool, np.bool_)) for value in
           (inlet_velocity_m_s, max_step_mm, wall_tolerance_mm)):
        raise ValueError('Несущий поток: скорость, шаг и допуск должны быть числами, не флажками')
    velocity = number(inlet_velocity_m_s, 'Скорость несущего потока')
    step = number(max_step_mm, 'Осевой шаг несущего потока')
    tolerance = number(wall_tolerance_mm, 'Допуск стенок несущего потока')
    if velocity <= 0 or step <= 0 or tolerance <= 0:
        raise ValueError('Несущий поток: скорость, шаг и допуск должны быть >0')
    if (isinstance(radial_cells, (bool, np.bool_))
            or not isinstance(radial_cells, (int, np.integer))
            or not 2 <= radial_cells <= MAX_RADIAL_CELLS):
        raise ValueError('Несущий поток: число радиальных ячеек — целое 2…128')
    model = Chamber(geometry_dict)
    if not model.enabled:
        raise ValueError('Несущий поток: включите профили жаровой трубы')
    canonical = json.dumps(_canonical_geometry(model), sort_keys=True, separators=(',', ':'))
    return _build(canonical, velocity, step, int(radial_cells), tolerance)


def clear_carrier_cache():
    _build.cache_clear()
