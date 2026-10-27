"""Undoable chart views and guides, independent of calculated/source data."""
from contextlib import contextmanager
import weakref
from matplotlib import ticker


class PlotHistory:
    def __init__(self, changed=None, limit=100):
        self.commands=[];self.index=0;self.limit=limit;self.changed=changed
        self.depth=0;self.replaying=False

    @staticmethod
    def capture(controllers):
        return [(weakref.ref(c),c.snapshot()) for c in dict.fromkeys(controllers)]

    @staticmethod
    def valid(command):
        for reference,state in command[1]:
            controller=reference()
            if controller is None or not controller.accepts_snapshot(state):return False
        return True

    def _prune(self):
        old=self.commands[:self.index]
        self.commands=[c for c in self.commands if self.valid(c)]
        self.index=sum(self.valid(c) for c in old)

    def can_undo(self):self._prune();return self.index>0
    def can_redo(self):self._prune();return self.index<len(self.commands)
    def undo_text(self):return self.commands[self.index-1][0] if self.can_undo() else ''
    def redo_text(self):return self.commands[self.index][0] if self.can_redo() else ''

    def record(self,label,before,controllers):
        if self.replaying:return
        after=self.capture(controllers)
        if len(before)!=len(after) or any(a[1]['signature']!=b[1]['signature'] for a,b in zip(before,after)):
            self._prune();self.commands[self.index:]=[(label,before,after)]
            if len(self.commands)>self.limit:self.commands=self.commands[-self.limit:]
            self.index=len(self.commands)
            if self.changed:self.changed()

    @contextmanager
    def action(self,label,controllers):
        outer=self.depth==0 and not self.replaying
        before=self.capture(controllers) if outer else None
        self.depth+=1
        try:yield
        finally:
            self.depth-=1
            if outer:self.record(label,before,controllers)

    def _restore(self,states):
        self.replaying=True
        try:
            for reference,state in states:
                controller=reference()
                if controller is not None:controller.restore_snapshot(state)
        finally:self.replaying=False
        if self.changed:self.changed()

    def undo(self):
        if not self.can_undo():return False
        self.index-=1;self._restore(self.commands[self.index][1]);return True

    def redo(self):
        if not self.can_redo():return False
        command=self.commands[self.index];self.index+=1;self._restore(command[2]);return True


def locator_state(locator):
    """Plain data for the locator types used by chart/axis settings."""
    edge=getattr(locator,'_edge',None)
    if isinstance(locator,ticker.MultipleLocator):return dict(kind='multiple',base=float(edge.step),offset=float(getattr(locator,'_offset',0)))
    if isinstance(locator,ticker.FixedLocator):return dict(kind='fixed',locations=[float(v) for v in locator.locs],nbins=locator.nbins)
    if isinstance(locator,ticker.NullLocator):return dict(kind='null')
    if isinstance(locator,ticker.AutoMinorLocator):return dict(kind='minor',n=getattr(locator,'ndivs',None))
    if isinstance(locator,ticker.MaxNLocator):
        return dict(kind='maxn',nbins=getattr(locator,'_nbins','auto'),steps=[float(v) for v in locator._steps],
                    integer=bool(locator._integer),symmetric=bool(locator._symmetric),prune=locator._prune,min_n_ticks=int(locator._min_n_ticks))
    if isinstance(locator,ticker.LinearLocator):return dict(kind='linear',numticks=int(locator.numticks))
    if isinstance(locator,ticker.LogLocator):
        return dict(kind='log',base=float(locator._base),subs=locator._subs if isinstance(locator._subs,str) else [float(v) for v in locator._subs],numticks=locator.numticks)
    return dict(kind='auto')


def restore_locator(state):
    values=dict(state);kind=values.pop('kind')
    if kind=='multiple':
        try:return ticker.MultipleLocator(**values)
        except TypeError:return ticker.MultipleLocator(values['base'])
    if kind=='fixed':return ticker.FixedLocator(values['locations'],nbins=values['nbins'])
    if kind=='null':return ticker.NullLocator()
    if kind=='minor':return ticker.AutoMinorLocator(values['n'])
    if kind=='maxn':return ticker.MaxNLocator(**values)
    if kind=='linear':return ticker.LinearLocator(**values)
    if kind=='log':return ticker.LogLocator(**values)
    return ticker.AutoLocator()
