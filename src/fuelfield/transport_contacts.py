"""Exact contacts and forward wall-film kinematics for an axisymmetric chamber.

Coordinates and lengths are millimetres; velocities are metres per second.
Wall IDs are 1 (inner) and 2 (outer).  Cartesian flight segments are intersected
with the analytic line/circle profiles, never with the drawing's chord mesh.
The contact tolerance is the chamber's physical tolerance (1e-7 mm).

This module supplies geometry only.  Capture, drag and the film constitutive
law belong to the transport integrator; there is no rebound or obstacle law.
"""
from __future__ import annotations

import numpy as np

from .chamber import TOL


_WALLS = {1: 'inner', 2: 'outer'}
_LD = np.longdouble
_EPS = np.finfo(_LD).eps
_ROOT_PAD = 2e-12


def _checked_z(model, values):
    values = np.asarray(values, dtype=float)
    if (np.any(~np.isfinite(values)) or np.any(values < model.base - TOL)
            or np.any(values > model.outlet + TOL)):
        raise ValueError('Wall coordinates must be finite and inside the chamber')
    return np.clip(values, model.base, model.outlet)


def _wall_inputs(model, wall_ids, z):
    ids = np.asarray(wall_ids)
    if np.any(~np.isin(ids, (1, 2))):
        raise ValueError('Wall IDs must be 1 (inner) or 2 (outer)')
    ids, z = np.broadcast_arrays(ids.astype(np.int8), _checked_z(model, z))
    return ids, z


def _profile_groups(model, ids, z):
    """Assign a joint to its downstream primitive, except at the outlet."""
    for identity, wall in _WALLS.items():
        profile = model.profiles[wall]
        for index, primitive in enumerate(profile):
            # Chamber accepts sub-TOL endpoint discrepancies in a join.  Later
            # assignments select the downstream piece in this tiny overlap.
            upper = z <= primitive.hi+TOL if index == len(profile)-1 else z < primitive.hi+TOL
            mask = (ids == identity) & (z >= primitive.lo-TOL) & upper
            if np.any(mask):
                yield primitive, mask


def wall_radius(model, wall_ids, z_mm):
    """Return exact wall radii [mm], broadcasting wall IDs and axial positions.

    ID 0 is deliberately invalid: a free parcel has no associated film wall.
    """
    ids, z = _wall_inputs(model, wall_ids, z_mm)
    result = np.full(ids.shape, np.nan)
    for primitive, mask in _profile_groups(model, ids, z):
        result[mask] = primitive.at(z[mask])
    if np.any(~np.isfinite(result)):
        raise ValueError('A chamber wall does not cover the requested coordinate')
    return result


def film_velocity(model, wall_ids, z_mm, theta_rad, speed_m_s):
    """Forward unit wall tangent times speed, in Cartesian [vz, vx, vy].

    Inputs broadcast.  At an exact circular vertical tangent vz is exactly
    zero; no division by the graph slope is used.  At profile joints the
    downstream tangent is used.  Speeds must be finite and nonnegative.
    """
    ids, z = _wall_inputs(model, wall_ids, z_mm)
    ids, z, theta, speed = np.broadcast_arrays(ids, z,
        np.asarray(theta_rad, dtype=float), np.asarray(speed_m_s, dtype=float))
    if (np.any(~np.isfinite(theta)) or np.any(~np.isfinite(speed))
            or np.any(speed < 0)):
        raise ValueError('Film angle/speed must be finite, with nonnegative speed')
    axial = np.full(ids.shape, np.nan)
    radial = np.full(ids.shape, np.nan)
    for primitive, mask in _profile_groups(model, ids, z):
        if primitive.kind == 'line':
            delta = primitive.end - primitive.start
            vector = delta * (1. if delta[0] > 0 else -1.)
            vector /= np.linalg.norm(vector)
            axial[mask], radial[mask] = vector
        elif primitive.kind == 'arc':
            offset = np.clip(z[mask]-primitive.center[0], -primitive.radius, primitive.radius)
            # The sqrt formula preserves the exact zero at a vertical tangent.
            axial[mask] = np.sqrt(np.maximum(0.,
                primitive.radius**2 - offset**2)) / primitive.radius
            radial[mask] = -primitive.branch * offset / primitive.radius
        else:
            raise ValueError('Unsupported wall primitive: ' + primitive.kind)
    if np.any(~np.isfinite(axial)):
        raise ValueError('A chamber wall does not cover the film coordinate')
    return np.stack((speed * axial, speed * radial * np.cos(theta),
                     speed * radial * np.sin(theta)), axis=-1)


def wall_arc_length(model, wall_id, z0_mm, z1_mm):
    """Exact nonnegative wall distance [mm] between forward axial positions.

    IDs and endpoints broadcast.  Reverse axial travel is rejected; film in
    this reduced model always moves towards the outlet.  Circular distances
    use the analytic circle angle, including finite travel to vertical ends.
    """
    ids, start = _wall_inputs(model, wall_id, z0_mm)
    ids, start, end = np.broadcast_arrays(ids, start, _checked_z(model, z1_mm))
    if np.any(end < start):
        raise ValueError('Wall-film travel must be forward in z')
    result = np.zeros(ids.shape)
    for identity, wall in _WALLS.items():
        for primitive in model.profiles[wall]:
            lo = np.maximum(start, primitive.lo)
            hi = np.minimum(end, primitive.hi)
            mask = (ids == identity) & (hi > lo)
            if not np.any(mask):
                continue
            if primitive.kind == 'line':
                delta = primitive.end - primitive.start
                result[mask] += (hi[mask] - lo[mask]) * np.linalg.norm(delta) / abs(delta[0])
            elif primitive.kind == 'arc':
                # atan2(sqrt(1-u²), u) is acos with better endpoint behaviour.
                a = np.clip((lo[mask] - primitive.center[0]) / primitive.radius, -1., 1.)
                b = np.clip((hi[mask] - primitive.center[0]) / primitive.radius, -1., 1.)
                angle_a = np.arctan2(np.sqrt(np.maximum(0., (1.-a)*(1.+a))), a)
                angle_b = np.arctan2(np.sqrt(np.maximum(0., (1.-b)*(1.+b))), b)
                result[mask] += primitive.radius * np.abs(angle_a - angle_b)
            else:
                raise ValueError('Unsupported wall primitive: ' + primitive.kind)
    return result


def _polyval(coefficients, value):
    result = _LD(0)
    for coefficient in coefficients[::-1]:
        result = result * value + coefficient
    return result


def _unit_roots(coefficients):
    """All real roots on [0,1], including double tangencies.

    Degree is at most four.  Derivative roots partition the interval into
    monotone pieces, so endpoint-inside tests and arbitrary sampling are not
    used to discover a crossing.  Long-double construction/evaluation avoids
    cancellation in the torus quartic; bracketed Newton isolates simple roots.
    """
    coefficients = np.asarray(coefficients, dtype=_LD)
    scale = np.max(np.abs(coefficients))
    if not np.isfinite(scale):
        raise ValueError('Nonfinite contact polynomial')
    if scale == 0:
        return [0., 1.]  # Entire segment lies on the algebraic surface.
    c = coefficients / scale
    while len(c) > 1 and abs(c[-1]) <= 8 * _EPS:
        c = c[:-1]
    degree = len(c) - 1
    if degree == 0:
        return []
    if degree == 1:
        roots = [-c[0] / c[1]]
    elif degree == 2:
        a, b, constant = c[2], c[1], c[0]
        discriminant = b*b - 4*a*constant
        error = 32 * _EPS * (b*b + abs(4*a*constant))
        if discriminant < -error:
            return []
        if discriminant <= error:
            roots = [-b / (2*a)]
        else:
            q = -.5 * (b + np.copysign(np.sqrt(discriminant), b))
            roots = [q/a, constant/q]
    else:
        critical = _unit_roots(c[1:] * np.arange(1, len(c), dtype=_LD))
        knots = [_LD(0), *(_LD(x) for x in critical if 0 < x < 1), _LD(1)]
        knots = sorted(set(knots))
        values = [_polyval(c, knot) for knot in knots]
        roots = []
        # Stationary zeroes include tangencies with no sign change.
        for knot, value in zip(knots, values):
            error = 64 * _EPS * sum(abs(coefficient) * abs(knot)**i
                                   for i, coefficient in enumerate(c))
            if abs(value) <= error:
                roots.append(knot)
        for lo, hi, flo, fhi in zip(knots, knots[1:], values, values[1:]):
            if (flo < 0) == (fhi < 0) or flo == 0 or fhi == 0:
                continue
            value = (lo+hi)/2
            derivative = c[1:]*np.arange(1, len(c), dtype=_LD)
            for _ in range(80):
                fv = _polyval(c, value)
                if fv == 0:
                    break
                if (flo < 0) == (fv < 0):
                    lo, flo = value, fv
                else:
                    hi = value
                slope = _polyval(derivative, value)
                proposed = value-fv/slope if slope != 0 else (lo+hi)/2
                if not lo <= proposed <= hi:
                    proposed = (lo+hi)/2
                # Coefficient evaluation error is magnified near a shallow
                # root; the unsquared physical surface is polished below.
                if abs(proposed-value) <= 8*np.finfo(float).eps:
                    value = proposed
                    break
                value = proposed
            else:
                for _ in range(64):
                    value = (lo+hi)/2
                    fv = _polyval(c, value)
                    if (flo < 0) == (fv < 0):
                        lo, flo = value, fv
                    else:
                        hi = value
                value = (lo+hi)/2
            roots.append(value)
    return sorted(set(float(np.clip(root, 0., 1.)) for root in roots
                      if -_ROOT_PAD <= root <= 1. + _ROOT_PAD))


def _radius_ld(primitive, z):
    if primitive.kind == 'line':
        return (_LD(primitive.start[1]) + (z - _LD(primitive.start[0]))
                * _LD(primitive.end[1] - primitive.start[1])
                / _LD(primitive.end[0] - primitive.start[0]))
    return (_LD(primitive.center[1]) + primitive.branch * np.sqrt(max(_LD(0),
        _LD(primitive.radius)**2 - (z - _LD(primitive.center[0]))**2)))


def _polynomial(primitive, z, dz, xy, delta):
    """Ascending powers in the primitive-overlap parameter t in [0,1]."""
    z, dz = _LD(z), _LD(dz)
    xy, delta = np.asarray(xy, dtype=_LD), np.asarray(delta, dtype=_LD)
    radial = np.array([xy @ xy, 2*(xy @ delta), delta @ delta], dtype=_LD)
    if primitive.kind == 'line':
        radius = _radius_ld(primitive, z)
        change = (dz * _LD(primitive.end[1]-primitive.start[1])
                  / _LD(primitive.end[0]-primitive.start[0]))
        # Factored constant avoids subtracting two nearly equal radii squared.
        initial_radius = np.sqrt(radial[0])
        return np.array([(initial_radius-radius)*(initial_radius+radius),
                         radial[1]-2*radius*change,
                         radial[2]-change*change], dtype=_LD)
    if primitive.kind != 'arc':
        raise ValueError('Unsupported wall primitive: ' + primitive.kind)
    # Scale all distances before forming a quartic.  The circle's meridional
    # equation is squared to eliminate sqrt(x²+y²); its other branch must later
    # be rejected against the actual analytic primitive.
    offset = z - _LD(primitive.center[0])
    cr, radius = _LD(primitive.center[1]), _LD(primitive.radius)
    scale = max(abs(offset), abs(dz), abs(cr), radius,
                np.max(np.abs(xy)), np.max(np.abs(delta)), _LD(1e-200))
    offset, dz, cr, radius = offset/scale, dz/scale, cr/scale, radius/scale
    radial /= scale*scale
    initial_radius = np.sqrt(radial[0])
    h = np.array([offset*offset+radial[0]+cr*cr-radius*radius,
                  2*offset*dz+radial[1], dz*dz+radial[2]], dtype=_LD)
    coefficients = np.convolve(h, h)
    coefficients[:3] -= 4*cr*cr*radial
    # Avoid the largest cancellation, at t=0, by difference of squares.
    coefficients[0] = ((offset*offset+(initial_radius-cr)**2-radius*radius)
                       * (offset*offset+(initial_radius+cr)**2-radius*radius))
    # These algebraically equivalent forms also improve c1/c2 near a large
    # chamber radius compared with the wall's circle radius.
    difference = (offset*offset+(initial_radius-cr)*(initial_radius+cr)-radius*radius)
    coefficients[1] = 4*offset*dz*h[0] + 2*radial[1]*difference
    coefficients[2] = (h[1]*h[1] + 2*dz*dz*h[0]
                       + 2*radial[2]*difference)
    return coefficients


def _refine(primitive, root, z, dz, xy, delta):
    """Polish the actual unsquared surface, without changing root branch."""
    value = _LD(root)
    z, dz = _LD(z), _LD(dz)
    xy, delta = np.asarray(xy, dtype=_LD), np.asarray(delta, dtype=_LD)
    for _ in range(4):
        point = xy + value*delta
        radius = np.sqrt(point @ point)
        if radius == 0:
            break  # Walls have strictly positive radius; filtering rejects it.
        axial = z + value*dz
        dr = point @ delta / radius
        if primitive.kind == 'line':
            f = radius - _radius_ld(primitive, axial)
            derivative = (dr-dz*_LD(primitive.end[1]-primitive.start[1])
                          / _LD(primitive.end[0]-primitive.start[0]))
        else:
            a, b = axial-_LD(primitive.center[0]), radius-_LD(primitive.center[1])
            f = a*a+b*b-_LD(primitive.radius)**2
            derivative = 2*(a*dz+b*dr)
        if derivative == 0:
            break
        correction = f/derivative
        candidate = value-correction
        if candidate < -_ROOT_PAD or candidate > 1+_ROOT_PAD:
            break
        # A near-double root must not jump to a distant root under Newton.
        if abs(correction) > 1e-5:
            break
        value = np.clip(candidate, _LD(0), _LD(1))
        if abs(correction) < 2*_EPS:
            break
    return float(value)


def _enters_fluid(primitive, wall_id, z, dz, xy, delta):
    """One-sided start-boundary classification, including grazing motion."""
    z, dz = _LD(z), _LD(dz)
    xy, delta = np.asarray(xy, dtype=_LD), np.asarray(delta, dtype=_LD)
    radius = np.sqrt(xy @ xy)
    dr = xy @ delta / radius
    if primitive.kind == 'line':
        slope = (_LD(primitive.end[1]-primitive.start[1])
                 / _LD(primitive.end[0]-primitive.start[0]))
        derivative = dr-slope*dz
        scale = abs(dr)+abs(slope*dz)
    else:
        # The circle implicit normal is finite at a vertical tangent.
        a, b = z-_LD(primitive.center[0]), _radius_ld(primitive, z)-_LD(primitive.center[1])
        derivative = primitive.branch*(a*dz+b*dr)
        scale = abs(a*dz)+abs(b*dr)
    sign = 1 if wall_id == 1 else -1
    derivative *= sign
    if abs(derivative) > 64*_EPS*max(scale, _LD(1e-200)):
        return derivative > 0
    # At zero first derivative, curvature decides whether a tangent segment
    # enters the fluid.  Differences are evaluated in extended precision so a
    # tangential Cartesian motion along a cylinder is classified correctly.
    probe = _LD(1e-4)
    point = xy+probe*delta
    change_r = np.sqrt(point @ point)-radius
    change_wall = _radius_ld(primitive, z+probe*dz)-_radius_ld(primitive, z)
    change = sign*(change_r-change_wall)
    error = 64*_EPS*max(abs(radius), abs(change_wall), _LD(1e-200))
    return change > error  # A segment remaining on the wall is a contact.


def first_contacts(model, z0_mm, z1_mm, xy0_mm, xy1_mm):
    """First exact wall contacts of N forward Cartesian flight segments.

    Return ``(fraction, wall_id)`` arrays of length N; ``inf, 0`` means no
    contact.  Fractions parameterize the complete segment, including endpoints.
    Initial points must be inside or on a wall.  A start-wall root is ignored
    only when the forward motion enters the fluid.  A positive-time tangency
    counts as a contact.  z may remain constant but must never decrease.

    All analytic profile pieces crossed by a step are examined.  In particular
    an inside--inside step through a narrow waist still produces a contact.
    """
    z = _checked_z(model, [z0_mm, z1_mm])
    if z.shape != (2,) or z[1] < z[0]:
        raise ValueError('Flight must have scalar forward axial endpoints')
    start, end = np.asarray(xy0_mm, dtype=float), np.asarray(xy1_mm, dtype=float)
    if (start.ndim != 2 or start.shape[1] != 2 or end.shape != start.shape
            or np.any(~np.isfinite(start)) or np.any(~np.isfinite(end))):
        raise ValueError('Flight endpoints must be finite arrays of shape (N,2)')
    count = len(start)
    fractions = np.full(count, np.inf)
    walls = np.zeros(count, dtype=np.int8)
    if not count:
        return fractions, walls
    r0 = np.linalg.norm(start, axis=1)
    inner, outer = model.bounds(float(z[0]))
    if np.any((r0 < inner-TOL) | (r0 > outer+TOL)):
        raise ValueError('A flight starts outside the closed chamber')
    dz, delta = float(z[1]-z[0]), end-start
    for identity, name in _WALLS.items():
        for primitive in model.profiles[name]:
            if dz:
                lo = max(0., (primitive.lo-z[0])/dz)
                hi = min(1., (primitive.hi-z[0])/dz)
                if hi <= lo:
                    continue
            else:
                if not primitive.lo <= z[0] <= primitive.hi:
                    continue
                lo, hi = 0., 1.
            za, axial_change = float(z[0]+lo*dz), (hi-lo)*dz
            xy, movement = start+lo*delta, (hi-lo)*delta
            # Exact radial extrema of each Cartesian chord.  Independent
            # analytic wall ranges are a safe rejection, even when endpoints
            # are inside and the interior of the segment crosses a wall.
            a = np.einsum('ij,ij->i', movement, movement)
            b = np.einsum('ij,ij->i', xy, movement)
            nearest = np.divide(-b, a, out=np.zeros_like(a), where=a > 0)
            nearest = np.clip(nearest, 0., 1.)
            radial_min = np.linalg.norm(xy+nearest[:, None]*movement, axis=1)
            radial_max = np.maximum(np.linalg.norm(xy, axis=1),
                                    np.linalg.norm(xy+movement, axis=1))
            wall_min, wall_max = primitive.range(za, za+axial_change)
            possible = ((radial_min <= wall_max+TOL) & (radial_max >= wall_min-TOL)
                        & (fractions >= lo-_ROOT_PAD))
            for index in np.flatnonzero(possible):
                coefficients = _polynomial(primitive, za, axial_change, xy[index], movement[index])
                roots = _unit_roots(coefficients)
                # Explicit physical endpoints also cover polynomial round-off
                # exactly at a join or a vertical circular tangent.
                for endpoint in (0., 1.):
                    point = xy[index]+endpoint*movement[index]
                    radius = np.linalg.norm(point)
                    if abs(radius-float(primitive.at(za+endpoint*axial_change))) <= TOL:
                        roots.append(endpoint)
                for local in sorted(set(roots)):
                    local = _refine(primitive, local, za, axial_change, xy[index], movement[index])
                    fraction = lo+(hi-lo)*local
                    if fraction >= fractions[index]:
                        continue
                    point = xy[index]+local*movement[index]
                    axial = za+local*axial_change
                    radius = np.linalg.norm(point)
                    if abs(radius-float(primitive.at(axial))) > TOL:
                        continue  # Wrong circle branch introduced by squaring.
                    if fraction <= _ROOT_PAD and _enters_fluid(
                            primitive, identity, za, axial_change, xy[index], movement[index]):
                        continue
                    fractions[index], walls[index] = max(0., fraction), identity
    # A missed crossing must fail visibly instead of silently losing fuel.
    final_radius = np.linalg.norm(end, axis=1)
    inner, outer = model.bounds(float(z[1]))
    missed = (walls == 0) & ((final_radius < inner-TOL) | (final_radius > outer+TOL))
    if np.any(missed):
        raise ValueError('Analytic wall contact could not be resolved for flight '
                         + str(int(np.flatnonzero(missed)[0])))
    return fractions, walls
