"""Batch calculation for a saved JSON project; no PyQt5 needed.

Usage: python batch_run.py examples/demo_project_140.json --out results
"""
from pathlib import Path
import sys, argparse, json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from fuelfield.core import project_from_json,calculate_plane,evaluate_continuous_field
from fuelfield.visualization import draw_nozzle_markers, draw_detailed_map, draw_polar_nozzles_map, color_scale
from fuelfield.data_io import write_result_csv


def main():
    parser=argparse.ArgumentParser(description='Calculate 1–50 mass-conservative fuel fields.')
    parser.add_argument('project',help='Path to JSON project saved from GUI')
    parser.add_argument('--out',default='results',help='Output folder')
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=Path(args.project)
    cfg,nozzles=project_from_json(json.loads(p.read_text(encoding='utf-8')))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    planes=[calculate_plane(nozzles,cfg,z) for z in cfg.z_mm]
    fields=[evaluate_continuous_field(nozzles,cfg,z) for z in cfg.z_mm]
    for i,(z,plane,smooth) in enumerate(zip(cfg.z_mm,planes,fields)):
        fig=plt.figure(figsize=(7,7))
        ax=fig.add_subplot(111,projection='polar')
        theta_edge=np.deg2rad(cfg.grid_start_deg+(np.arange(cfg.n_angular+1)-.5)*360/cfg.n_angular)
        th,r=np.meshgrid(theta_edge,plane.radial_edges_mm)
        data,norm,cmap=color_scale(plane.flux_kg_m2_s,cfg,[p.flux_kg_m2_s for p in planes])
        m=ax.pcolormesh(th,r,data,shading='flat',cmap=cmap,norm=norm)
        ax.set_theta_zero_location('E');ax.set_theta_direction(1)
        ax.set_ylim(0,cfg.r_out_mm)
        ax.set_title(f'Section {i+1}, z={z:g} mm')
        draw_nozzle_markers(ax,cfg,nozzles,polar=True)
        fig.colorbar(m,ax=ax,shrink=.8,label='kg/(m² s)')
        fig.savefig(out/f'section_{i+1}_field.png',dpi=160,bbox_inches='tight');plt.close(fig)
        fig=plt.figure(figsize=(8,8),layout='constrained')
        draw_detailed_map(fig,smooth,plane,cfg,nozzles,
                          shared_values=[f.masked_flux_kg_m2_s for f in fields])
        fig.savefig(out/f'section_{i+1}_detailed.png',dpi=160,bbox_inches='tight');plt.close(fig)
        fig=plt.figure(figsize=(8,8),layout='constrained')
        draw_polar_nozzles_map(fig,plane,cfg,nozzles)
        fig.savefig(out/f'section_{i+1}_polar_nozzles.png',dpi=160,bbox_inches='tight');plt.close(fig)
        cols=2 if cfg.n_radial>1 else 1
        rows=int(np.ceil(cfg.n_radial/cols))
        fig,axes=plt.subplots(rows,cols,figsize=(11,max(5,rows*2.5)),squeeze=False)
        order=np.argsort(plane.theta_centers_deg)
        for k in range(cfg.n_radial):
            ax=axes.flat[k]
            x=plane.theta_centers_deg[order]
            y=plane.flux_kg_m2_s[k,order]
            ax.plot(np.r_[x,x[0]+360],np.r_[y,y[0]])
            ax.set_title(f'Level {k+1}, R={plane.radial_centers_mm[k]:g} mm')
            ax.set_xlabel('angle, deg');ax.set_ylabel('kg/(m² s)');ax.grid(alpha=.2)
            ax.set_xlim(0,360)
        for ax in list(axes.flat)[cfg.n_radial:]:ax.axis('off')
        fig.tight_layout();fig.savefig(out/f'section_{i+1}_profiles.png',dpi=140,bbox_inches='tight');plt.close(fig)
        print(f'Section {i+1} z={z:g} mm | total={plane.injected_kg_s*3600:.5f} kg/h | '
              f'in={plane.inside_kg_s*3600:.5f}, inner={plane.inner_kg_s*3600:.5f}, '
              f'outer={plane.outer_kg_s*3600:.5f}; balance error={plane.rel_balance_error:+.3e}')
    write_result_csv(str(out/'fuel_fields.csv'),planes)
    print('Results:',out.resolve())

if __name__=='__main__':main()
