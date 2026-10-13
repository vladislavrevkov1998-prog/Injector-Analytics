"""Independent reference: RK4 momentum/work equations + maximum-flow search.

No application imports or Gauss quadrature. Equations/graph readings are from
Dityakin et al., 1977, pp. 38, 39, 42, 45, 47, 51, 57. This is a numerical
verification of the implemented short-chamber approximation, not measured data.
"""
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

PARAMETERS=dict(outlet_diameter_mm=1.,inlet_diameter_mm=.5,inlet_count=3,
    swirl_arm_mm=2.5,chamber_height_mm=.5,inlet_length_mm=1.,outlet_length_mm=.75,
    cone_angle_deg=105.,kinematic_viscosity_mm2_s=1.7)

def reference(mass):
    rho=800.;rc=.0005;ri=.00025;n=3;nu=1.7e-6;arm=.0025
    epsilon=.844 # fig. 24 at r_in/R = .1
    a=arm*rc/(epsilon*n*ri**2);ck=(arm+ri)/rc
    re=mass/rho/(n*math.pi*ri**2)*2*ri/nu;re_total=re*math.sqrt(n)
    lam=24.6/re_total**.75 if re_total<=2300 else 1.22/re_total**.36
    beta=lam*a/2
    # x decreases from chamber wall to outlet. y=M/M_in; s=loss coefficient.
    def derivatives(x,y):return beta*y*y, -lam*a**3*y**3/x**2
    steps=10000;dx=(1-ck)/steps;x=ck;y=1.;loss=0.
    for _ in range(steps):
        k1,s1=derivatives(x,y)
        k2,s2=derivatives(x+dx/2,y+dx*k1/2)
        k3,s3=derivatives(x+dx/2,y+dx*k2/2)
        k4,s4=derivatives(x+dx,y+dx*k3)
        y+=dx*(k1+2*k2+2*k3+k4)/6;loss+=dx*(s1+2*s2+2*s3+s4)/6;x+=dx
    ae=a*y
    # Maximize the original ideal mu^2; avoids the application's phi equation.
    with localcontext() as context:
        context.prec=60;aa=Decimal(str(ae))**2;lo=Decimal('0');hi=Decimal('1')
        def score(p):return p*p*(1-p)/(aa*p*p+1-p)
        for _ in range(140):
            l=(2*lo+hi)/3;r=(lo+2*hi)/3
            if score(l)<score(r):lo=l
            else:hi=r
        phi=float((lo+hi)/2)
    samples=((2.125,2.65),(2.25,2.12),(2.5,1.60),(2.75,1.26),(3.,1.10),
             (3.25,.97),(3.5,.88),(4.,.76),(4.5,.68),(4.7,.66))
    log_re=math.log10(re)
    for (r0,q0),(r1,q1) in zip(samples,samples[1:]):
        if r0<=log_re<=r1:
            xi=q0+(q1-q0)*(log_re-r0)/(r1-r0);break
    inlet_loss=xi*(rc*rc/(n*ri*ri))**2
    outlet_loss=.135/phi**2
    mu=1/math.sqrt(ae*ae/(1-phi)+1/phi**2+inlet_loss+loss+outlet_loss)
    dp=.5*rho*(mass/rho/(math.pi*rc*rc*mu))**2
    return dict(mass_kg_s=mass,pressure_pa=dp,re_inlet=re,re_equivalent=re_total,
        friction=lam,a_active=a,a_equivalent=ae,phi=phi,loss_inlet=inlet_loss,
        loss_chamber=loss,loss_outlet=outlet_loss,mu=mu,
        film_mm=.5*(1-math.sqrt(1-phi)))

if __name__=='__main__':
    data=dict(provenance='Independent RK4 integration and Decimal maximum-flow search; see dityakin_reference.py',
              parameters=PARAMETERS,density_kg_m3=800.,states=[reference(m) for m in (.002,.004,.006)])
    Path(__file__).with_suffix('.json').write_text(json.dumps(data,indent=2)+'\n')
