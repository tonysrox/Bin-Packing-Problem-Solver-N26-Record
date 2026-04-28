import numpy as np
from scipy.optimize import minimize, basinhopping
import matplotlib.pyplot as ppt
from numba import njit
from joblib import Parallel, delayed
import argparse
import sys
import os


if sys.platform == 'win32':
    try:
        import psutil
        p = psutil.Process(os.getpid())
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        print("system priority high")
    except ImportError:
        print("system priority normal")


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("N", type=int, help="Number of inner polygons (e.g., 26)")
arg_parser.add_argument("nsi", type=int, help="Sides of inner polygons (3=tri, 4=square)")
arg_parser.add_argument("nsc", type=int, help="Sides of container (3=tri, 4=square)")
arg_parser.add_argument("--attempts", type=int, default=20, help="Parallel attempts")
args = arg_parser.parse_args()

N = args.N
nsi = args.nsi
nsc = args.nsc


unit_angles = np.linspace(0, 2 * np.pi, nsi, endpoint=False)
unit_vertices = np.column_stack((np.cos(unit_angles), np.sin(unit_angles)))
unit_normals = np.column_stack((np.cos(unit_angles + np.pi/nsi), np.sin(unit_angles + np.pi/nsi)))
cont_angles = np.linspace(0, 2 * np.pi, nsc, endpoint=False)
cont_normals = np.column_stack((np.cos(cont_angles + np.pi/nsc), np.sin(cont_angles + np.pi/nsc)))
cont_apothem = np.cos(np.pi / nsc)

@njit(cache=True)
def calculate_penalty(values, S):
    penalty = 0.0
    polys = np.zeros((N, nsi, 2))
    norms = np.zeros((N, nsi, 2))
    limit = cont_apothem * S
    
    for i in range(N):
        x, y, a = values[i*3], values[i*3+1], values[i*3+2]
        c, s = np.cos(a), np.sin(a)
        for v in range(nsi):
            vx, vy = unit_vertices[v, 0], unit_vertices[v, 1]
            tx = x + (vx * c - vy * s)
            ty = y + (vx * s + vy * c)
            polys[i, v, 0], polys[i, v, 1] = tx, ty
            
            for k in range(nsc):
                dist = tx * cont_normals[k, 0] + ty * cont_normals[k, 1]
                if dist > limit:
                    penalty += (dist - limit)**2
        
        for v in range(nsi):
            nx, ny = unit_normals[v, 0], unit_normals[v, 1]
            norms[i, v, 0] = nx * c - ny * s
            norms[i, v, 1] = nx * s + ny * c

    for i in range(N):
        for j in range(i + 1, N):
            dx = values[i*3] - values[j*3]
            dy = values[i*3+1] - values[j*3+1]
            if (dx*dx + dy*dy) > 4.1: 
                continue
            
            min_overlap = 1e18
            collision = True
            for axis_idx in range(nsi * 2):
                if axis_idx < nsi:
                    ax, ay = norms[i, axis_idx, 0], norms[i, axis_idx, 1]
                else:
                    ax, ay = norms[j, axis_idx - nsi, 0], norms[j, axis_idx - nsi, 1]
                
                min1, max1 = 1e18, -1e18
                min2, max2 = 1e18, -1e18
                for v in range(nsi):
                    d1 = polys[i, v, 0] * ax + polys[i, v, 1] * ay
                    if d1 < min1: min1 = d1
                    if d1 > max1: max1 = d1
                    d2 = polys[j, v, 0] * ax + polys[j, v, 1] * ay
                    if d2 < min2: min2 = d2
                    if d2 > max2: max2 = d2
                
                overlap = min(max1, max2) - max(min1, min2)
                if overlap <= 0:
                    collision = False
                    break
                if overlap < min_overlap:
                    min_overlap = overlap
            
            if collision:
                penalty += min_overlap**2
    return penalty

def run_attempt(seed):
    np.random.seed(seed)
    current_S = 5.408
    step_down = 0.99995 
    
    x0 = np.random.uniform(-0.5, 0.5, N * 3)
    x0[0::3] *= (current_S * 0.8)
    x0[1::3] *= (current_S * 0.8)
    x0[2::3] = np.random.uniform(0, 2*np.pi, N)
    
    best_x, best_S = x0.copy(), current_S

    bh = basinhopping(calculate_penalty, x0, niter=100, T=0.1, 
                      minimizer_kwargs={"args": (current_S,), "method": "L-BFGS-B", "tol": 1e-9})
    x0 = bh.x

    failed_attempts = 0
    for i in range(400):
        res = minimize(calculate_penalty, x0, args=(current_S,), 
                       method='L-BFGS-B', tol=1e-12)
        
        if res.fun < 1e-9:
            best_x, best_S = res.x.copy(), current_S
            x0 = res.x
            current_S *= step_down 
            failed_attempts = 0
        else:
            failed_attempts += 1
            x0 = best_x + np.random.normal(0, 0.005, N * 3) 
            
        if failed_attempts > 5:
            bh_refined = basinhopping(calculate_penalty, x0, niter=15, T=0.02,
                                      minimizer_kwargs={"args": (current_S,), "method": "L-BFGS-B", "tol": 1e-10})
            x0 = bh_refined.x
            failed_attempts = 0 
    return best_S, best_x

if __name__ == "__main__":
    print(f"Starting packing problem solver for N={N}...")
    results = Parallel(n_jobs=-1)(delayed(run_attempt)(i) for i in range(args.attempts))
    best_S, best_vals = min(results, key=lambda x: x[0])

    final_s = best_S * np.sin(np.pi / nsc) / np.sin(np.pi / nsi) 

    print("\n" + "="*45)
    print(f"RESULTS FOR N={N}")
    print(f"Best Side Length (s): {final_s:.10f}")
    print("="*45)

    fig, ax = ppt.subplots(figsize=(10,10))
    c_v = unit_vertices * best_S
    ax.plot(np.append(c_v[:,0], c_v[0,0]), np.append(c_v[:,1], c_v[0,1]), 'r-', lw=3)
    for i in range(N):
        x, y, a = best_vals[i*3:i*3+3]
        c, s = np.cos(a), np.sin(a)
        p = np.array([ [x + (vx*c - vy*s), y + (vx*s + vy*c)] for vx, vy in unit_vertices ])
        ax.fill(p[:,0], p[:,1], alpha=0.7, edgecolor='black', facecolor='#1f77b4', linewidth=1.5)
    
    ax.set_aspect('equal')
    ppt.axis('off')
    ppt.savefig(f"N{N}_record_attempt.png", dpi=300, bbox_inches='tight')
    submission_filename = f"N{N}_submission_data.txt"
    with open(submission_filename, "w") as f:
        f.write(f"RECORD SUBMISSION: N={N} Triangles in Triangle\n")
        f.write(f"Found by: [YOUR NAME]\n")
        f.write(f"Method: Basinhopping + L-BFGS-B (Numba Accelerated)\n")
        f.write(f"Container Side Length (s): {final_s:.12f}\n")
        f.write("-" * 50 + "\n")
        f.write("Format: Triangle index | x-center | y-center | rotation(rad)\n")
        f.write("-" * 50 + "\n")
        
        for i in range(N):
            x, y, a = best_vals[i*3 : i*3+3]
            f.write(f"{i+1:2d} | {x:14.10f} | {y:14.10f} | {a:14.10f}\n")

    print(f"\n[SUCCESS]")
    print(f"1. Coordinate data saved to: {submission_filename}")
    print(f"2. Plot image saved to: N{N}_record_attempt.png")
