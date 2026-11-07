"""Faithful Python translation of the *active* numerical MATLAB algorithm.

Original reference: source/matlab_reference_original.m (400 lines).
Default inputs and the final ACTIVE scenario (Kotel'naya, statistical trials)
are exactly the input images/coordinates in that script.

This is an offline experiment with two given ordinary images. No camera
tracking, geographic positioning, or external-device integration is provided.

Important: MATLAB code contains exploratory plotting, overwritten scenarios,
unused 'load radiopoligon.mat', and many unused 4-D intermediate arrays.
The active numerical recurrence is retained including its potentially invalid
probabilities. Memory layout is collapsed only for quantities which are
mathematically equal across candidates; see README_RU.md.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import correlate

try:
    from numba import njit
except ImportError:  # Portable correctness-first fallback; much slower.
    def njit(*args: Any, **kwargs: Any):  # type: ignore
        def wrap(fn):
            return fn
        return wrap


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class OriginalParameters:
    """Last effective assignments of the MATLAB script, NOT manual tuning."""
    N: int = 256
    b: float = 0.4
    bt: float = 0.2
    bb: float = 1.0
    i0e: int = 180  # MATLAB row (1-based), reference
    j0e: int = 200  # MATLAB column (1-based), reference
    i0: int = 160   # MATLAB row (1-based), first candidate
    j0: int = 170   # MATLAB column (1-based), first candidate
    a: int = 50     # inclusive: 51 rows
    c: int = 200    # inclusive: 201 columns
    H: int = 50     # inclusive: 51 candidate rows
    M: int = 50     # inclusive: 51 candidate cols
    p01: float = 0.5
    p02: float = 0.5
    S1cir_max: float = 0.5


def read_image_as_matlab_double(path: Path) -> np.ndarray:
    """MATLAB imread->rgb2gray(uint8)->double, for RGB or grayscale files.

    rgb2gray for uint8 performs an integer-rounded color conversion. We use
    the documented luminance coefficients and NumPy nearest rounding; exact
    agreement with a particular MATLAB internal rgb2gray build is not proven.
    """
    with Image.open(path) as im:
        a = np.array(im)
    if a.ndim == 3:
        if a.shape[2] < 3:
            raise ValueError('Ожидается одноканальное или RGB изображение')
        rgb = a[:, :, :3].astype(np.float64)
        a = np.rint(
            0.29893602129377539 * rgb[:, :, 0]
            + 0.58704307445112136 * rgb[:, :, 1]
            + 0.11402090425510326 * rgb[:, :, 2]
        ).clip(0, 255)
    return np.asarray(a, dtype=np.float64)


def preprocess_matlab(image: np.ndarray, repeats: int) -> dict[str, np.ndarray]:
    """MATLAB imfilter(..., ones(5), 'replicate') and corr Sobel kernels."""
    I = image.astype(np.float64, copy=True)
    lowfreq = np.ones((5, 5), dtype=np.float64)  # no division by 25
    for _ in range(repeats):
        I = correlate(I, lowfreq, mode='nearest')

    sobel1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)
    sobel2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    S1 = correlate(I, sobel1, mode='nearest')
    S2 = correlate(I, sobel2, mode='nearest')
    module = np.sqrt(S1**2 + S2**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        fi = np.arctan(S2 / S1)  # MATLAB atan(S2./S1), not atan2()
        normalized = (module - module.min()) / (module.max() - module.min())
    return {'filtered': I, 'S1': S1, 'S2': S2, 'magnitude': module, 'fi': fi, 'norm': normalized}


def matlab_var(a: np.ndarray) -> float:
    """MATLAB var(A(:)) default unbiased normalization 1/(n-1)."""
    return float(np.var(a, ddof=1))


def beta_parameters(mean: float, variance: float) -> tuple[float, float]:
    common = mean * (1.0 - mean) / variance - 1.0
    return mean * common, (1.0 - mean) * common


def prepare_inputs(
    reference_image: Path,
    current_image: Path,
    cfg: OriginalParameters = OriginalParameters(),
) -> dict[str, Any]:
    # MATLAB: k1 = imread(p0), k2 = imread(p)
    I0 = read_image_as_matlab_double(reference_image)
    I02 = read_image_as_matlab_double(current_image)
    if I0.shape != I02.shape:
        raise ValueError('В MATLAB K(:,:,1:3) требуется одинаковый размер кадров.')
    height, width = I0.shape
    if not (1 <= cfg.i0e <= height - cfg.a and 1 <= cfg.j0e <= width - cfg.c):
        raise ValueError('Эталонный фрагмент выходит за изображение.')
    if not (1 <= cfg.i0 + cfg.H + cfg.a + 1 <= height and 1 <= cfg.j0 + cfg.M + cfg.c + 1 <= width):
        raise ValueError('Сетка кандидатов выходит за изображение для MATLAB индексации.')

    ref = preprocess_matlab(I0, 3)
    curr = preprocess_matlab(I02, 2)
    I_0_1, I_0_12 = ref['norm'], curr['norm']
    with np.errstate(invalid='ignore'):
        mc = float(I_0_1[I_0_1 > cfg.b].mean())
        Rc = matlab_var(I_0_1[I_0_1 > cfg.b])
        mf = float(I_0_1[I_0_1 < cfg.b].mean())
        Rf = matlab_var(I_0_1[I_0_1 < cfg.b])
        mct = float(I_0_12[I_0_12 > cfg.bt].mean())
        Rct = matlab_var(I_0_12[I_0_12 > cfg.bt])
        mft = float(I_0_12[I_0_12 < cfg.bt].mean())
        Rft = matlab_var(I_0_12[I_0_12 < cfg.bt])

    # MATLAB: I_0_12 = I_0_12 + (mc - mct)
    shifted_current = I_0_12 + (mc - mct)
    maskf = np.full_like(I_0_1, mf)
    maskf[I_0_1 < cfg.b] = I_0_1[I_0_1 < cfg.b]
    maskc = np.full_like(I_0_1, mf)
    maskc[I_0_1 > cfg.b] = I_0_1[I_0_1 > cfg.b]

    # MATLAB inclusiveness: maskc(i0e:i0e+a,j0e:j0e+c).
    startrow = cfg.i0e - 1
    startcol = cfg.j0e - 1
    maskc_e = maskc[startrow:startrow + cfg.a + 1, startcol:startcol + cfg.c + 1]
    row0, col0 = np.where(maskc_e > cfg.b)
    # MATLAB find() scans columns first, then rows.
    order = np.lexsort((row0, col0))
    ic = np.asarray(row0[order], dtype=np.int32)
    jc = np.asarray(col0[order], dtype=np.int32)
    xe = maskc_e[ic, jc]
    Q = int(ic.size)
    if Q == 0:
        raise ValueError('В эталоне нет пикселей > b, Q=0.')

    # Unusual but literal histogram density from original: x uses count spread.
    G = np.histogram(I_0_1, bins=cfg.N)[0]
    x = (G.max() - G.min()) / cfg.N
    with np.errstate(divide='ignore', invalid='ignore'):
        f0 = G / x
    mu = float(I_0_1.mean())
    Ru = matlab_var(I_0_1)

    metrics = {
        'shape': [height, width],
        'Q': Q, 'candidate_count': (cfg.H + 1) * (cfg.M + 1),
        'N': cfg.N, 'b': cfg.b, 'bt': cfg.bt,
        'mu': mu, 'Ru': Ru,
        'mc': mc, 'Rc': Rc, 'mf': mf, 'Rf': Rf,
        'mct': mct, 'Rct': Rct, 'mft': mft, 'Rft': Rft,
        'shift_mc_minus_mct': mc-mct,
        'initial_teta1': Rc + mc * mc,
        'initial_teta2': Rft + mft * mft,
        'alpha_beta_full': beta_parameters(mu, Ru),
        'alpha_beta_signal': beta_parameters(mc, Rc),
        'alpha_beta_background': beta_parameters(mf, Rf),
        'histogram_x_from_counts': float(x),
        'warning_f0_definition': 'MATLAB uses spread of bin COUNTS, not intensity bin width.',
    }
    return {
        'cfg': cfg, 'metrics': metrics,
        'ref_raw': I0, 'current_raw': I02,
        'ref': ref, 'curr': curr, 'shifted_current': shifted_current,
        'maskf': maskf, 'maskc': maskc, 'maskc_e': maskc_e,
        'ic': ic, 'jc': jc, 'xe': xe, 'G': G, 'f0': f0,
    }


@njit(cache=True, error_model='numpy')
def original_recursion_step(
    frame: np.ndarray, ic: np.ndarray, jc: np.ndarray,
    mu1: np.ndarray, mu2: np.ndarray,
    p1: np.ndarray, p2: np.ndarray, S1cir: np.ndarray,
    theta1: float, theta2: float, Rct: float, Rft: float,
    i0: int, j0: int, H: int, M: int,
    bb: float, score_max: float,
) -> tuple[float, int, int, int, int, int, int]:
    """Literal loop m -> h -> q, with MATLAB f/X/W and original S1cir.

    `theta1` and `theta2` are the accumulating *teta1/teta2*, NOT W/f.
    MATLAB teta1(q,h,m,k+1)=teta1(q,h,m,k)+Rct for every q,h,m;
    therefore the scalar theta values are exactly identical across q,h,m.
    `mu1/mu2` are xcir1/xcir2 and DO get replaced by X/f each frame.
    """
    Q = ic.size
    total_nonprob = 0
    total_nonfinite = 0
    total_sc1 = 0
    last_changed_h = -1
    last_changed_m = -1
    scir = np.zeros((H+1, M+1), dtype=np.int8)

    for m in range(M+1):
        for h in range(H+1):
            sum_f1 = 0.0
            sum_f2 = 0.0
            for q in range(Q):
                # Original: i=ic(q)+i0+h, j=jc(q)+j0+m; MATLAB 1-based.
                # `ic` is 0-based, so python row = ic + i0 + h.
                z = bb * frame[i0+h+ic[q], j0+m+jc[q]]
                c0 = 0.75 * (1.0 - z*z)
                c1 = 1.5 * bb * z
                c2 = -0.75 * bb*bb

                x1 = mu1[h, m, q]
                x2 = mu2[h, m, q]
                R1 = theta1 - x1*x1
                R2 = theta2 - x2*x2
                gamma1 = 1.0/R1 * x1*(1.0-x1)-1.0
                gamma2 = 1.0/R2 * x2*(1.0-x2)-1.0
                alfa1 = x1 * gamma1
                alfa2 = x2 * gamma2
                # Keep the source's original parentheses/order, not a simplified beta-polynomial.
                f1 = c0+alfa1/gamma1*(c1+c2*(alfa1+1.0)/(gamma1+1.0))
                f2 = c0+alfa2/gamma2*(c1+c2*(alfa2+1.0)/(gamma2+1.0))
                X1 = alfa1/gamma1*(c0+(alfa1+1.0)/(gamma1+1.0)*(c1+c2*(alfa1+2.0)/(gamma1+2.0)))
                X2 = alfa2/gamma2*(c0+(alfa2+1.0)/(gamma2+1.0)*(c1+c2*(alfa2+2.0)/(gamma2+2.0)))
                W1 = alfa1/gamma1*(alfa1+1.0)/(gamma1+1.0)*(c0+(alfa1+2.0)/(gamma1+2.0)*(c1+c2*(alfa1+3.0)/(gamma1+3.0)))
                W2 = alfa2/gamma2*(alfa2+1.0)/(gamma2+1.0)*(c0+(alfa2+2.0)/(gamma2+2.0)*(c1+c2*(alfa2+3.0)/(gamma2+3.0)))
                # MATLAB: xcir1=X1/f1, tetacir1=W1/f1, Rcir=tetacir-xcir^2.
                # Only xcir, NOT tetacir, is read on the next k iteration.
                mu1[h,m,q] = X1/f1
                mu2[h,m,q] = X2/f2
                tetacir1 = W1/f1
                tetacir2 = W2/f2
                unused_Rcir1 = tetacir1 - mu1[h,m,q]*mu1[h,m,q]
                unused_Rcir2 = tetacir2 - mu2[h,m,q]*mu2[h,m,q]
                if not (np.isfinite(mu1[h,m,q]) and np.isfinite(mu2[h,m,q]) and np.isfinite(unused_Rcir1) and np.isfinite(unused_Rcir2)):
                    total_nonfinite += 1
                # MATLAB sums f, does NOT multiply across q.
                sum_f1 += f1
                sum_f2 += f2

            Sf = p1[h,m]*sum_f1 + p2[h,m]*sum_f2
            p1[h,m] = p1[h,m]*sum_f1/Sf
            p2[h,m] = p2[h,m]*sum_f2/Sf
            if not (np.isfinite(p1[h,m]) and np.isfinite(p2[h,m])):
                total_nonfinite += 1
            if p1[h,m]<0.0 or p1[h,m]>1.0 or p2[h,m]<0.0 or p2[h,m]>1.0:
                total_nonprob += 1

            # MATLAB resets S_p1,S_p2 to 0 for each candidate, then sums
            # the single p1/p2 value once: structure=1 iff p1>p2.
            if p1[h,m] > p2[h,m]:
                scir[h,m] = 1
                total_sc1 += 1
                S1cir[h,m] += p1[h,m]
            else:
                scir[h,m] = 2

    # MATLAB scanning order: r=1:M+1 OUTER, t=1:H+1 INNER; ties use >=.
    for r in range(M+1):
        for t in range(H+1):
            if scir[t,r] == 1 and S1cir[t,r] >= score_max:
                score_max = S1cir[t,r]
                last_changed_h = t
                last_changed_m = r
    return (score_max,last_changed_h,last_changed_m,total_nonprob,total_nonfinite,total_sc1,Q*(H+1)*(M+1))


def run_original_experiment(data: dict[str, Any], progress: bool = True) -> dict[str, Any]:
    cfg = data['cfg']
    metrics = data['metrics']
    Q = metrics['Q']
    # MATLAB k=1 is the reference, k=2,3 are the SAME shifted current image.
    # The numerical state with independent dimensions (q,h,m) is retained.
    shape = (cfg.H+1,cfg.M+1,Q)
    mu1 = np.full(shape, metrics['mc'], dtype=np.float64)
    mu2 = np.full(shape, metrics['mft'], dtype=np.float64)
    p1 = np.full((cfg.H+1,cfg.M+1), cfg.p01, dtype=np.float64)
    p2 = np.full((cfg.H+1,cfg.M+1), cfg.p02, dtype=np.float64)
    S1cir = np.zeros_like(p1)
    theta1 = metrics['initial_teta1']
    theta2 = metrics['initial_teta2']
    score_max = cfg.S1cir_max
    history = []
    for k in (2,3):  # MATLAB K(:,:,2), K(:,:,3)
        theta1 += metrics['Rct']
        theta2 += metrics['Rft']
        out = original_recursion_step(
            data['shifted_current'],data['ic'],data['jc'],mu1,mu2,p1,p2,S1cir,
            theta1,theta2,metrics['Rct'],metrics['Rft'],
            cfg.i0,cfg.j0,cfg.H,cfg.M,cfg.bb,score_max,
        )
        score_max, h, m, nonprob, nonfinite, sc1, operations = out
        # If no new maximum at k>2, MATLAB does not assign tcir(k)/rcir(k).
        # Expose that explicitly rather than falsely claiming previous box found.
        valid = h >= 0 and m >= 0
        item = {
            'matlab_k': k, 'new_maximum': valid,
            'h': int(h) if valid else None,
            'm': int(m) if valid else None,
            'matlab_tcir': int(h + 1) if valid else None,
            'matlab_rcir': int(m + 1) if valid else None,
            'score_max': float(score_max),
            'non_probability_candidates': int(nonprob),
            'non_finite_count': int(nonfinite),
            'structure_1_count': int(sc1),
            'evaluated_pixel_steps': int(operations),
            'theta1': float(theta1), 'theta2': float(theta2),
        }
        if valid:
            item['box_pixel_0_based'] = {
                'x': int(cfg.j0 + m), 'y': int(cfg.i0 + h),
                'w': int(cfg.c + 1), 'h': int(cfg.a + 1),
            }
        history.append(item)
        if progress:
            print(f"Шаг k={k}: новый максимум={valid}, t={item['matlab_tcir']} r={item['matlab_rcir']}, "
                  f"S1max={score_max:.9g}; отрицат./>1 p: {nonprob}, нечисловых: {nonfinite}")

    # Keep one p map from each structure and accumulated S1 as raw research output.
    return {
        'params': asdict(cfg), 'metrics': metrics, 'history': history,
        'p1':p1, 'p2':p2, 'S1cir':S1cir,
        'nonfinite_p1':int(np.count_nonzero(~np.isfinite(p1))),
        'out_of_bounds_p1':int(np.count_nonzero(np.isfinite(p1) & ((p1<0)|(p1>1)))),
        'score_max':float(score_max),
    }


def plot_experiment(data:dict[str,Any],result:dict[str,Any],outdir:Path)->None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cfg=data['cfg']
    outdir.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    axes[0,0].imshow(data['ref_raw'],cmap='gray')
    axes[0,0].set_title('Эталон image21.bmp')
    def draw_rect(ax, x, y, w, h, label, style='-'):
        xx=[x,x+w,x+w,x,x]
        yy=[y,y,y+h,y+h,y]
        ax.plot(xx, yy, linestyle=style, linewidth=2, label=label)

    draw_rect(axes[0,0],cfg.j0e-1,cfg.i0e-1,cfg.c+1,cfg.a+1,'Эталон')
    axes[0,1].imshow(data['current_raw'],cmap='gray')
    axes[0,1].set_title('Текущее изображение 01_0_0.png')
    for row in result['history']:
        b=row.get('box_pixel_0_based')
        if b:
            draw_rect(axes[0,1],b['x'],b['y'],b['w'],b['h'],
                      f"После k={row['matlab_k']}",
                      style='-' if row['matlab_k']==2 else '--')
    axes[0,1].legend(loc='upper right', fontsize=9)
    axes[0,2].imshow(data['ref']['norm'],cmap='gray',vmin=0,vmax=1)
    axes[0,2].set_title('Собель эталона (3 прохода 5×5)')
    axes[1,0].imshow(data['curr']['norm'],cmap='gray',vmin=0,vmax=1)
    axes[1,0].set_title('Собель текущего (2 прохода 5×5)')
    axes[1,1].imshow(result['S1cir'],cmap='viridis',origin='lower',aspect='auto')
    axes[1,1].set_title('Накопленная S1cir: h против m')
    axes[1,1].set_xlabel('m');axes[1,1].set_ylabel('h')
    axes[1,2].imshow(result['p1'],cmap='viridis',origin='lower',aspect='auto')
    axes[1,2].set_title('Итоговая p1 (возможны значения вне [0,1])')
    axes[1,2].set_xlabel('m');axes[1,2].set_ylabel('h')
    for ax in axes.flat[:4]:ax.axis('off')
    fig.savefig(outdir/'overview.png',dpi=135)
    plt.close(fig)

    # Distributions shown as in MATLAB; f0 histogram's erroneous normalization is preserved.
    from scipy.stats import beta as beta_dist
    x=np.arange(0,1.001,.01)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for label,key in [('Все','alpha_beta_full'),('Контуры','alpha_beta_signal'),('Фон','alpha_beta_background')]:
        alpha,beta=result['metrics'][key]
        with np.errstate(divide='ignore',invalid='ignore'):
            axes[0].plot(x,beta_dist.pdf(x,alpha,beta),label=label)
    axes[0].set_xlim(0,1);axes[0].set_title('Бета-плотности методом моментов');axes[0].legend()
    axes[1].plot(np.linspace(0,1,len(data['f0'])),data['f0'])
    axes[1].set_title('Исходная f0=G/x (MATLAB, спорная нормировка)')
    fig.savefig(outdir/'distributions.png',dpi=135)
    plt.close(fig)



def plot_matlab_figures(data:dict[str,Any],outdir:Path)->None:
    """Save the 11 ACTIVE MATLAB `figure;` displays (presentational analogue)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import beta as beta_dist
    outdir.mkdir(parents=True,exist_ok=True)
    cfg=data['cfg']
    plots=[
        ('01_filtered_reference','Исходное изображение',data['ref']['filtered'],'image'),
        ('02_sobel_magnitude','Модуль отклика на действие оператора Собеля',data['ref']['magnitude'],'image'),
        ('03_gradient_angle','Угол направления градиента яркости',data['ref']['fi'],'image'),
        ('04_normalized_edges','От 0 до 1',data['ref']['norm'],'image'),
        ('05_histogram','Гистограмма от 0 до 1',data['ref']['norm'],'hist'),
    ]
    x=np.arange(0,1.001,.01)
    for filename, title, arr, kind in plots:
        fig,ax=plt.subplots(figsize=(7,5))
        if kind=='image':ax.imshow(arr,cmap='gray')
        else:
            freqs,edges=np.histogram(arr,bins=cfg.N)
            ax.bar((edges[:-1]+edges[1:])/2, freqs/freqs.sum(),width=np.diff(edges),align='center')
        ax.set_title(title)
        fig.tight_layout();fig.savefig(outdir/(filename+'.png'),dpi=120);plt.close(fig)
    for index,(key,title) in enumerate([('alpha_beta_full','f_и'),('alpha_beta_signal','f_с'),('alpha_beta_background','f_ф')],start=6):
        a,b=data['metrics'][key]
        fig,ax=plt.subplots(figsize=(7,5))
        with np.errstate(invalid='ignore',divide='ignore'):
            ax.plot(x,beta_dist.pdf(x,a,b))
        ax.set_title(title)
        fig.tight_layout();fig.savefig(outdir/(f'{index:02}_beta_{key}.png'),dpi=120);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5));ax.plot(np.linspace(0,1,cfg.N),data['f0']);ax.set_title('f_0')
    fig.tight_layout();fig.savefig(outdir/'09_f0.png',dpi=120);plt.close(fig)
    for index,title,img in [(10,'Фон',data['maskf']),(11,'Сигнал',data['maskc'])]:
        fig,ax=plt.subplots(figsize=(7,5));ax.imshow(img,cmap='gray');ax.set_title(title)
        fig.tight_layout();fig.savefig(outdir/f'{index:02}_{title}.png',dpi=120);plt.close(fig)

def save_results(data:dict[str,Any],result:dict[str,Any],outdir:Path,make_plots:bool=True)->None:
    outdir.mkdir(parents=True,exist_ok=True)
    serial={key:result[key] for key in ('params','metrics','history','nonfinite_p1','out_of_bounds_p1','score_max')}
    (outdir/'results.json').write_text(json.dumps(serial,ensure_ascii=False,indent=2,allow_nan=True),encoding='utf-8')
    np.savez_compressed(outdir/'maps.npz',p1=result['p1'],p2=result['p2'],S1cir=result['S1cir'],
                        ic=data['ic'],jc=data['jc'],xe=data['xe'])
    if make_plots:plot_experiment(data,result,outdir)


def main()->None:
    parser=argparse.ArgumentParser(description='Прямой Python-перенос математической части оригинального MATLAB-скрипта')
    parser.add_argument('--reference',type=Path,default=ROOT/'inputs'/'image21.bmp',help='MATLAB p0 / k1')
    parser.add_argument('--current',type=Path,default=ROOT/'inputs'/'01_0_0.png',help='MATLAB p / k2')
    parser.add_argument('--output',type=Path,default=ROOT/'results')
    parser.add_argument('--no-plots',action='store_true')
    parser.add_argument('--original-figures',action='store_true',help='Создать отдельные аналоги 11 figure из MATLAB')
    args=parser.parse_args()
    data=prepare_inputs(args.reference,args.current)
    print('Размер:',data['metrics']['shape'],'Q:',data['metrics']['Q'],'сетка:',data['metrics']['candidate_count'])
    out=run_original_experiment(data)
    save_results(data,out,args.output,make_plots=not args.no_plots)
    if args.original_figures:
        plot_matlab_figures(data,args.output/'matlab_figures')
    print('Готово:',args.output.resolve())


if __name__=='__main__':
    main()