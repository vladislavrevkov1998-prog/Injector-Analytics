"""Reproducible, explicitly conditional band-overlap teaching diagrams."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[1]
COLORS=['#3481b8','#9b67b3','#dc8a34']


def overlap(before,after):
    """Annular area fractions for the stated constant-flux, straight-ray example."""
    a,b=np.asarray(before[:-1])[:,None],np.asarray(before[1:])[:,None]
    lo=np.maximum(a,after[:-1]);hi=np.minimum(b,after[1:])
    inside=np.maximum(hi**2-lo**2,0)/(b**2-a**2)
    return np.column_stack((inside,1-inside.sum(axis=1)))


def diagram(name,before,after,title):
    fig=plt.figure(figsize=(8.4,5.8),dpi=130,layout='constrained')
    grid=fig.add_gridspec(2,1,height_ratios=[3,1.35]);ax=fig.add_subplot(grid[0])
    z0,z1=10,90
    for j,color in enumerate(COLORS):
        ax.add_patch(Rectangle((4,before[j]),6,before[j+1]-before[j],facecolor=color,alpha=.38,lw=0))
        ax.add_patch(Rectangle((90,after[j]),6,after[j+1]-after[j],facecolor=color,alpha=.18,lw=0))
        ax.text(1,(before[j]+before[j+1])/2,f'A{j+1}',ha='right',va='center',color=color,weight='bold')
        ax.text(99,(after[j]+after[j+1])/2,f'B{j+1}',ha='left',va='center',color=color,weight='bold')
    for k in range(4):
        ax.plot([z0,z1],[before[k],after[k]],color='#8392a5',ls='--',lw=.8)
    for z in np.linspace(z0,z1,5):
        t=(z-z0)/(z1-z0);edges=(1-t)*np.asarray(before)+t*np.asarray(after)
        ax.plot([z,z],[edges[0],edges[-1]],color='#c7d0dd',lw=.8)
    for j,color in enumerate(COLORS):
        for r in np.linspace(before[j]+1,before[j+1]-1,7):
            outside=r<after[0] or r>=after[-1]
            ax.annotate('',xy=(z1,r),xytext=(z0,r),arrowprops=dict(arrowstyle='->',color=color,lw=1.1,alpha=.78))
            if outside:ax.plot(z1,r,'x',color='#b9374f',ms=5)
    if after[-1]>before[-1]:
        for r in (105,155):
            ax.annotate('',xy=(z1,r),xytext=(z0,r),arrowprops=dict(arrowstyle='->',color='#b9374f',lw=1,ls=':'))
        ax.text(50,157.5,'Вне A → внутри B: локальная категория, без истории контакта',fontsize=7,color='#a12b42',ha='center')
    else:ax.text(50,157.5,'× вне стенки в B: wall-bound; стрелка не загибается вдоль стенки',fontsize=8,color='#a12b42',ha='center')
    for k in (0,-1):ax.plot([z0,z1],[before[k],after[k]],color='#263c50',lw=2)
    ax.set(xlim=(-4,106),ylim=(96,163),xlabel='z, условные осевые станции A → B',ylabel='R, мм',title=title)
    ax.set_xticks([z0,30,50,70,z1],['A','z₁','z₂','z₃','B']);ax.grid(axis='y',alpha=.15)
    values=overlap(before,after)*100
    tableax=fig.add_subplot(grid[1]);tableax.axis('off')
    table=tableax.table(cellText=[[f'{v:.1f} %' for v in row] for row in values],rowLabels=['из A1','из A2','из A3'],colLabels=['в B1','в B2','в B3','вне B'],loc='center',cellLoc='center')
    table.auto_set_font_size(False);table.set_fontsize(9);table.scale(.9,1.5)
    for j,color in enumerate(COLORS):table[j+1,-1].set_facecolor(color);table[j+1,-1].get_text().set_color('white')
    fig.suptitle('УЧЕБНАЯ СХЕМА: прямые лучи, q = const; не оператор переноса v0.6',fontsize=10,weight='bold')
    output=ROOT/'docs/images/v0.6.3';output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/name,dpi=100);plt.close(fig)


def current_model():
    fig,ax=plt.subplots(figsize=(8.4,3.8),layout='constrained')
    for z in (20,40,60,80):
        t=(z-20)/60;edges=np.linspace(100+15*t,160-15*t,4)
        for j in range(3):
            ax.add_patch(Rectangle((z-3,edges[j]),6,edges[j+1]-edges[j],facecolor=COLORS[j],alpha=.28))
            ax.text(z,(edges[j]+edges[j+1])/2,f'{j+1}',ha='center',va='center',fontsize=8)
        ax.plot([z,z],[edges[0],edges[-1]],color='#617284',lw=.7)
    for j in range(4):ax.plot([20,80],[100+j*20,115+j*10],color='#99a5b5',ls='--',lw=.8)
    for a,b in ((100,115),(160,145)):ax.plot([20,80],[a,b],color='#263c50',lw=2)
    for z,text in ((20,'K(zA)'),(80,'K(zB)')):
        ax.annotate('',xy=(z,164),xytext=(0,190),arrowprops=dict(arrowstyle='->',color='#3c6e99',lw=1.8,connectionstyle='arc3,rad=-.15'))
        ax.text(z,167,text,ha='center',fontsize=10,color='#3c6e99')
    ax.plot(0,190,'o',color='#3c6e99',ms=5)
    ax.text(0,207,'Исходные форсунки Gⱼ и их эмпирические ядра',fontsize=10,color='#263c50')
    ax.text(50,96,'Матрица B вычисляется заново; матрица A не является входом',fontsize=9,ha='center',color='#a12b42')
    ax.set(xlim=(-7,95),ylim=(92,218),xlabel='z →',ylabel='R, мм',title='ТЕКУЩАЯ v0.6: независимые сечения, равновысотная радиальная сетка')
    ax.set_xticks([20,40,60,80],['A','z₁','z₂','B']);ax.spines[['top','right']].set_visible(False)
    fig.savefig(ROOT/'docs/images/v0.6.3/current-model.png',dpi=100);plt.close(fig)


if __name__=='__main__':
    diagram('narrowing.png',[100,120,140,160],[115,125,135,145],'Сужение: один исходный пояс может дать вклад в несколько новых')
    diagram('widening.png',[115,125,135,145],[100,120,140,160],'Расширение: пояса меняют границы, факел не растягивается по стенкам')
    current_model()
