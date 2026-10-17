"""Cached image updates without redrawing axes or hundreds of nozzle labels."""
import numpy as np
from matplotlib.backends.backend_agg import RendererAgg

class QuasiFieldAnimator:
    def __init__(self,canvas,axes,cfg):
        self.canvas=canvas;self.axes=axes;self.cfg=cfg;self.image=axes.images[0];self.background=None;self.overlay=None
        # With nearest-neighbour display, map the sampled scalar pixels to
        # colours after resizing. RGBA-first expands all 720² samples to four
        # channels even when the on-screen field is only ~300² pixels.
        self.image.set_interpolation_stage('data')
        self.image.set_animated(True);self.draw_count=0;self.update_count=0
        self.cid=canvas.mpl_connect('draw_event',self.drawn)
    def drawn(self,event):
        self.draw_count+=1
        if self.canvas.is_saving():self.background=None;return
        # Start from a clean axes background: masked/Clip pixels must not retain
        # old image values or receive the same translucent marker twice.
        self.axes.patch.draw(self.canvas.get_renderer());self.background=self.canvas.copy_from_bbox(self.axes.bbox)
        renderer=RendererAgg(int(self.canvas.figure.bbox.width),int(self.canvas.figure.bbox.height),self.canvas.figure.dpi)
        for artist in [*self.axes.lines,*self.axes.patches,*self.axes.texts,*self.axes.spines.values()]:
            if artist.get_visible():artist.draw(renderer)
        rgba=np.asarray(renderer.buffer_rgba());height,width=rgba.shape[:2];box=self.axes.bbox
        left=max(0,int(np.floor(box.x0)));right=min(width,int(np.ceil(box.x1)));top=max(0,int(np.floor(height-box.y1)));bottom=min(height,int(np.ceil(height-box.y0)))
        self.overlay=(left,height-bottom,np.ascontiguousarray(rgba[top:bottom,left:right][::-1]))
        # The Qt backend presents this full draw itself. Calling blit from a
        # draw_event re-enters paintEvent while the renderer is still drawing.
        self.paint(blit=False)
    def paint(self,blit=True):
        if self.background is None:self.canvas.draw_idle();return
        self.canvas.restore_region(self.background);self.axes.draw_artist(self.image)
        if self.overlay is not None:
            renderer=self.canvas.get_renderer();gc=renderer.new_gc();gc.set_clip_rectangle(self.axes.bbox);renderer.draw_image(gc,*self.overlay);gc.restore()
        if blit:self.canvas.blit(self.axes.bbox)
    def update(self,field):
        # The norm and colormap are fixed when the view is built. Recreating
        # them (and copying the whole field twice) for every frame is needless.
        data=field.masked_flux_kg_m2_s
        if self.cfg.color_scale_mode=='manual' and self.cfg.color_out_of_range=='clip':data=np.ma.masked_outside(data,self.cfg.color_vmin,self.cfg.color_vmax,copy=False)
        self.image.set_data(data);self.update_count+=1;self.paint()
