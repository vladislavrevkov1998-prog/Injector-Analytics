"""Read draft wall intersections without substituting legacy cylinder radii."""
import numpy as np
from .chamber import Primitive,TOL,UNITS
from .chamber_editing import arc_circle


def draft_bounds(raw,z):
    """Return unambiguous intersections of the *drawn* walls with a section.

    A bad arc direction elsewhere must not lengthen every section. For an
    invalid arc use its actual displayed sweep, including multiple hits; never
    silently choose a different circle branch. Missing/ambiguous hits hide the
    section rather than inventing a wall.
    """
    radii=[]
    for wall in ('inner','outer'):
        hits=[]
        for p in raw.get('primitives',[]):
            if p['wall']!=wall or p['kind']=='point':continue
            try:
                primitive=Primitive(p)
                if primitive.lo-TOL<=z<=primitive.hi+TOL:hits.append(float(primitive.at(np.clip(z,primitive.lo,primitive.hi))))
            except ValueError:
                if p['kind']!='arc':continue
                try:
                    a,b,center,radius,angle,sweep=arc_circle(p)
                except ValueError:continue
                factor=UNITS[p.get('unit','mm')];center=center*factor;radius*=factor
                x=(z-center[0])/radius
                if abs(x)>1+TOL/max(radius,TOL):continue
                theta=np.arccos(np.clip(x,-1,1))
                for t in (theta,-theta):
                    distance=(angle-t)%(2*np.pi) if sweep<0 else (t-angle)%(2*np.pi)
                    if distance<=abs(sweep)+1e-10 or abs(distance-2*np.pi)<1e-10:
                        hits.append(float(center[1]+radius*np.sin(t)))
        unique=[]
        for value in hits:
            if not any(abs(value-old)<=TOL for old in unique):unique.append(value)
        if len(unique)!=1:return None
        radii.append(unique[0])
    return tuple(radii) if radii[1]>radii[0]+TOL else None
