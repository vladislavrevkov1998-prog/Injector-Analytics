"""Empirical, path-length directed air rays in Cartesian coordinates.

There are no velocities or physical momentum in this module. ``s`` is a
geometric path length [mm], not time. Neighbour coupling aligns conditional
directions in overlap regions of their rays; its strength is dimensionless.
It must not be interpreted as conservation of momentum or calibrated CFD.
"""
from dataclasses import dataclass
import numpy as np


def first_wall_contact(model, start, end):
    """Exact line/arc wall intersection for forward, radial AND reverse rays.

    The tested polynomial/root helpers are shared with fuel geometry. Unlike
    the fuel integrator's forward-only wrapper, interval clipping below uses
    either sign of dz. A segment cannot jump across an inside–outside–inside
    throat merely because both numerical endpoints lie in the fluid.
    """
    from .chamber import TOL
    from .transport_contacts import _polynomial, _unit_roots, _refine, _enters_fluid, _ROOT_PAD
    delta = end-start
    dz = float(delta[0])
    best = float('inf')
    for wall_id, name in ((1, 'inner'), (2, 'outer')):
        for primitive in model.profiles[name]:
            if abs(dz) > 1e-14:
                interval = sorted(((primitive.lo-start[0])/dz, (primitive.hi-start[0])/dz))
                lo, hi = max(0., interval[0]), min(1., interval[1])
                if hi <= lo:
                    continue
            elif primitive.lo-TOL <= start[0] <= primitive.hi+TOL:
                lo, hi = 0., 1.
            else:
                continue
            a = start+lo*delta
            movement = (hi-lo)*delta
            axial_end = a[0]+movement[0]
            radii = np.array([np.hypot(*(a[1:]+t*movement[1:])) for t in (0., .5, 1.)])
            at = primitive.at(np.clip(np.array([a[0], (a[0]+axial_end)/2, axial_end]), primitive.lo, primitive.hi))
            # A conditional wall layer may ride exactly on a wall without
            # becoming a captured fuel film. This is a geometric boundary
            # case, not a wall-impact constitutive law.
            if primitive.kind == 'line' and np.all(np.abs(radii-at) <= TOL):
                continue
            # Safe rejection using extrema of the Cartesian radius and the
            # exact primitive range; midpoint alone is never a certificate.
            denom = float(movement[1:]@movement[1:])
            closest = np.clip(-float(a[1:]@movement[1:])/denom, 0., 1.) if denom else 0.
            radial_min = np.hypot(*(a[1:]+closest*movement[1:]))
            radial_max = max(radii[0], radii[-1])
            wall_min, wall_max = primitive.range(*sorted((float(a[0]), float(axial_end))))
            if radial_min > wall_max+TOL or radial_max < wall_min-TOL:
                continue
            coefficients = _polynomial(primitive, a[0], movement[0], a[1:], movement[1:])
            roots = _unit_roots(coefficients)
            for endpoint in (0., 1.):
                p = a+endpoint*movement
                if abs(np.hypot(*p[1:])-float(primitive.at(float(np.clip(p[0], primitive.lo, primitive.hi))))) <= TOL:
                    roots.append(endpoint)
            for local in sorted(set(roots)):
                local = _refine(primitive, local, a[0], movement[0], a[1:], movement[1:])
                fraction = lo+(hi-lo)*local
                if fraction >= best:
                    continue
                point = a+local*movement
                if abs(np.hypot(*point[1:])-float(primitive.at(float(np.clip(point[0], primitive.lo, primitive.hi))))) > TOL:
                    continue
                if fraction <= _ROOT_PAD and _enters_fluid(primitive, wall_id, a[0], movement[0], a[1:], movement[1:]):
                    continue
                best = max(0., float(fraction))
    return best


def local_to_cartesian(direction, theta_deg):
    dz, dr, dt = np.asarray(direction, dtype=float)
    theta = np.deg2rad(theta_deg)
    return np.array([dz, dr*np.cos(theta)-dt*np.sin(theta),
                     dr*np.sin(theta)+dt*np.cos(theta)])


def cartesian_to_local(direction, theta_deg):
    dz, dx, dy = np.asarray(direction, dtype=float)
    theta = np.deg2rad(theta_deg)
    return np.array([dz, dx*np.cos(theta)+dy*np.sin(theta),
                     -dx*np.sin(theta)+dy*np.cos(theta)])


@dataclass
class JetPaths:
    sources: tuple
    path_mm: np.ndarray
    positions: np.ndarray
    directions: np.ndarray
    stop_steps: np.ndarray
    stop_reasons: tuple
    diagnostics: tuple
    ambiguity_path_mm: object = None


def trace_jets(sources, flows, cfg, model):
    """Trace all rays on a fixed physical path grid, independent of output z.

    A neighbour is sampled at the nearest downstream point of its inlet ray,
    so sources with different axial origins do not need an invented common
    travel time. Directions are updated simultaneously, making registry order
    irrelevant. Exactly opposing rays remain symmetric and are reported as
    counterflow instead of being arbitrarily turned downstream.
    """
    from .chamber import chamber
    sources = tuple(sorted(sources, key=lambda s: s['id']))
    count = len(sources)
    steps = int(np.ceil(model['max_path_mm']/model['step_mm']))
    if (steps+1)*max(1, count) > 5_000_000:
        raise ValueError('Воздушные траектории превышают лимит памяти: увеличьте внутренний шаг или уменьшите длину пути')
    ds = model['max_path_mm']/steps
    path = np.arange(steps+1)*ds
    positions = np.zeros((steps+1, count, 3))
    directions = np.zeros_like(positions)
    if not sources:
        return JetPaths(sources, path, positions, directions, np.empty(0, dtype=int), (), ())
    origins = np.array([[s['z_mm'], s['radius_mm']*np.cos(np.deg2rad(s['theta_deg'])),
                         s['radius_mm']*np.sin(np.deg2rad(s['theta_deg']))] for s in sources])
    initial = np.array([local_to_cartesian(s['direction'], s['theta_deg']) for s in sources])
    positions[0] = origins
    directions[0] = initial
    strength = model['interaction_strength']
    interaction_length = model['interaction_length_mm']
    # Only ratios enter coupling. Log scaling avoids overflow from otherwise
    # finite independently entered G and penetration factors.
    intensity_logs = np.array([np.log(flows[s['id']])+np.log(s['penetration_factor'])
                               if flows.get(s['id'], 0.) > 0 and s['penetration_factor'] > 0 else -np.inf
                               for s in sources])
    max_intensity = float(np.max(intensity_logs))
    intensities = np.exp(intensity_logs-max_intensity) if np.isfinite(max_intensity) else np.zeros(count)
    widths = np.array([np.hypot(s['sigma_r_mm'], s['radius_mm']*np.deg2rad(s['sigma_theta_deg'])) for s in sources])
    penetration = np.array([s['penetration_factor'] for s in sources])
    stopped = intensities < 0  # All start active, including zero-penetration rays.
    stops = np.full(count, steps, dtype=int)
    reasons = ['max_path']*count
    ambiguity = np.full(count, np.inf)
    geometry = chamber(cfg)
    diagnostics = []
    if strength:
        diagnostics.append('Взаимодействие струй — эмпирическое выравнивание направлений; не физический баланс импульса.')
    if np.any(initial[:, 0] <= 1e-10):
        diagnostics.append('Есть поперечные/обратные струи: учитываются фактические пересечения траекторий, обратный расход не включён в положительную матрицу α.')
    for step in range(steps):
        previous = positions[step]
        vector = directions[step].copy()
        if strength and count > 1:
            # delta[i,j] is the position of i relative to the inlet of j.
            delta = previous[:, None, :] - origins[None, :, :]
            along = np.maximum(0., np.einsum('ijk,jk->ij', delta, initial))
            closest = origins[None, :, :] + along[:, :, None]*initial[None, :, :]
            distance2 = np.sum((previous[:, None, :]-closest)**2, axis=2)
            width_squared = (widths[:, None]+widths[None, :])**2
            exponent = np.divide(distance2, width_squared,
                                 out=np.where(distance2 == 0, 0., np.inf), where=width_squared > 0)
            overlap = np.exp(-.5*exponent)
            np.fill_diagonal(overlap, 0.)
            overlap[:, stopped] = 0.
            # A zero penetration setting has zero coupling and no NaN. It
            # does not erase its known mass flow from the independent ledger.
            fraction = np.divide(intensities[None, :], intensities[:, None]+intensities[None, :],
                                 out=np.zeros((count, count)), where=(intensities[:, None]+intensities[None, :]) > 0)
            weights = overlap*fraction*(penetration[:, None] > 0)
            dot = np.clip(vector@vector.T, -1., 1.)
            # Exact/near antiparallel symmetry has no preferred turn axis.
            # Record the unresolved encounter instead of manufacturing one.
            center_distance2 = np.sum((previous[:, None, :]-previous[None, :, :])**2, axis=2)
            unresolved = ((dot < -1.+1e-8) & (center_distance2 <= np.maximum(width_squared, 1e-20))
                          & (intensities[:, None] > 0) & (intensities[None, :] > 0))
            affected = unresolved.any(axis=1) & ~stopped
            ambiguity[affected] = np.minimum(ambiguity[affected], path[step])
            # The angle factor is dimensionless; projection below supplies
            # zero torque for exactly parallel or antiparallel directions.
            weights *= .25+.75*(1-dot)/2
            target = np.einsum('ij,jk->ik', weights, vector)
            total_weights = weights.sum(axis=1)
            target = np.divide(target, total_weights[:, None], out=np.zeros_like(target), where=total_weights[:, None] > 0)
            # Project the alignment force onto the unit sphere tangent. The
            # update is bounded and cannot create an arbitrary direction from
            # a symmetric exact cancellation.
            projection = target - np.sum(target*vector, axis=1)[:, None]*vector
            rate = np.minimum(1., strength*ds/interaction_length*total_weights)
            proposed = vector + rate[:, None]*projection
            norms = np.linalg.norm(proposed, axis=1)
            vector = np.divide(proposed, norms[:, None], out=vector, where=norms[:, None] > 1e-14)
        proposed = previous + ds*vector
        proposed[stopped] = previous[stopped]
        for index in np.flatnonzero(~stopped):
            source = sources[index]
            if source['kernel'] != 'wall':
                continue
            theta = np.rad2deg(np.arctan2(previous[index, 2], previous[index, 1]))
            local = cartesian_to_local(vector[index], theta)
            wall = source['wall']
            normal_zr = geometry.primitive_at(wall, previous[index, 0]).normal(previous[index, 0]) if geometry else np.array([0., 1. if wall == 'inner' else -1.])
            normal = np.array([normal_zr[0], normal_zr[1], 0.])
            tangent = local-float(local@normal)*normal
            norm = float(np.linalg.norm(tangent))
            if norm > 1e-12:
                vector[index] = local_to_cartesian(tangent/norm, theta)
            proposed[index] = previous[index]+ds*vector[index]
            z = proposed[index, 0]
            if not geometry or geometry.base <= z <= geometry.outlet:
                radius = cfg.bounds_at(float(z))[0 if wall == 'inner' else 1]
                theta = np.arctan2(proposed[index, 2], proposed[index, 1])
                proposed[index, 1:] = radius*np.array([np.cos(theta), np.sin(theta)])
        if geometry:
            for index in np.flatnonzero(~stopped):
                start = previous[index]
                end = proposed[index]
                axial_fraction = 1.
                if end[0] > geometry.outlet and end[0] > start[0]:
                    axial_fraction = (geometry.outlet-start[0])/(end[0]-start[0])
                elif end[0] < geometry.base and end[0] < start[0]:
                    axial_fraction = (geometry.base-start[0])/(end[0]-start[0])
                bounded_end = start+axial_fraction*(end-start)
                if sources[index]['kernel'] == 'wall':
                    wall = sources[index]['wall']
                    radius = geometry.bounds(float(bounded_end[0]))[0 if wall == 'inner' else 1]
                    theta = np.arctan2(bounded_end[2], bounded_end[1])
                    bounded_end[1:] = radius*np.array([np.cos(theta), np.sin(theta)])
                    proposed[index] = bounded_end
                    contact = float('inf')  # This path follows the analytic wall.
                else:
                    contact = first_wall_contact(geometry, start, bounded_end)
                if np.isfinite(contact):
                    proposed[index] = start+contact*(bounded_end-start)
                    stopped[index] = True
                    stops[index] = step+1
                    reasons[index] = 'wall_exit'
                elif axial_fraction < 1.:
                    proposed[index] = bounded_end
                    stopped[index] = True
                    stops[index] = step+1
                    reasons[index] = 'axial_exit'
        directions[step+1] = vector
        positions[step+1] = proposed
        radius = np.hypot(proposed[:, 1], proposed[:, 2])
        if geometry:
            from .transport_contacts import wall_radius
            outside_z = (proposed[:, 0] < geometry.base-1e-7) | (proposed[:, 0] > geometry.outlet+1e-7)
            bounded_z = np.clip(proposed[:, 0], geometry.base, geometry.outlet)
            rin = wall_radius(geometry, np.ones(count, dtype=np.int8), bounded_z)
            rout = wall_radius(geometry, np.full(count, 2, dtype=np.int8), bounded_z)
        else:
            outside_z = np.zeros(count, dtype=bool)
            rin, rout = cfg.r_in_mm, cfg.r_out_mm
        outside_wall = (radius < rin-1e-7) | (radius > rout+1e-7)
        ending = ~stopped & (outside_z | outside_wall)
        for index in np.flatnonzero(ending):
            stopped[index] = True
            stops[index] = step+1
            reasons[index] = 'axial_exit' if outside_z[index] else 'wall_exit'
        if stopped.all():
            positions = positions[:step+2]
            directions = directions[:step+2]
            path = path[:step+2]
            break
    if np.isfinite(ambiguity).any():
        diagnostics.append('Встречные почти антипараллельные струи: ось разворота неоднозначна; α после встречи маскируется, произвольного отклонения нет.')
    return JetPaths(sources, path, positions, directions, stops, tuple(reasons), tuple(diagnostics), ambiguity)


def section_crossing(paths, index, z_mm, cfg):
    """First oriented crossing; repeated crossings are explicitly ambiguous."""
    end = min(int(paths.stop_steps[index])+1, len(paths.path_mm))
    points = paths.positions[:end, index]
    delta = np.diff(points[:, 0])
    z0 = points[:-1, 0]
    z1 = points[1:, 0]
    positive = np.flatnonzero((delta > 1e-12) & (z0 <= z_mm+1e-9) & (z1 > z_mm+1e-9))
    negative = np.flatnonzero((delta < -1e-12) & (z0 >= z_mm-1e-9) & (z1 < z_mm-1e-9))
    # Include an exact final endpoint (the usual chamber outlet).
    if len(delta) and abs(points[-1, 0]-z_mm) <= 1e-9 and abs(delta[-1]) > 1e-12:
        pool = positive if delta[-1] > 0 else negative
        if len(pool) == 0 or pool[-1] != len(delta)-1:
            if delta[-1] > 0:
                positive = np.append(positive, len(delta)-1)
            else:
                negative = np.append(negative, len(delta)-1)
    crossings = sorted([(int(i), 1) for i in positive] + [(int(i), -1) for i in negative])
    valid = []
    for step, sign in crossings:
        fraction = (z_mm-points[step, 0])/delta[step]
        point = points[step] + fraction*(points[step+1]-points[step])
        radius = float(np.hypot(point[1], point[2]))
        rin, rout = cfg.bounds_at(float(z_mm))
        if paths.sources[index]['kernel'] == 'wall':
            radius = rin if paths.sources[index]['wall'] == 'inner' else rout
        if rin-1e-7 <= radius <= rout+1e-7:
            direction = paths.directions[step+1, index]
            theta = float(np.rad2deg(np.arctan2(point[2], point[1])) % 360)
            valid.append(dict(sign=sign, center=[float(z_mm), radius, theta],
                              direction=cartesian_to_local(direction, theta).tolist(),
                              path_mm=float(paths.path_mm[step]+fraction*np.linalg.norm(points[step+1]-points[step]))))
    if not valid:
        return None
    result = valid[0]
    result['multiple_crossings'] = len(valid) > 1
    return result
