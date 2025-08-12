import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap

cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
  0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
 [0.0589714286, 0.6837571429, 0.7253857143], 
 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
 [0.7184095238, 0.7411333333, 0.3904761905], 
 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
  0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
 [0.9763, 0.9831, 0.0538]]

MATMAP = LinearSegmentedColormap.from_list('parula', cm_data)

WAVEMAP_PAL = ['#5e60ce', '#00c49a','#ffca3a','#D81159','#fe7f2d','#7bdff2','#0496ff','#efa6c9','#ced4da', '#1eb43a']
WAVEMAP_PAL2 = ['#D94E72', '#55C9D9', '#62D98F', '#6d8891', '#559FD9', '#D94EC5', '#8DDA62']

MISC6_PAL = ['#1CA5FF', '#E38F19', '#AE1ED6', '#AED61C', '#D63A29', '#1CD676']

cmap = colormaps.get_cmap('Set1')

class MPLCarousel:
    def __init__(self, plot_function, n_items, figsize=(12, 8), subplot_layout=(1, 1), enable_toggle=False, toggle_key=' ', cache=None):
        self.plot_function = plot_function
        self.n_items = n_items
        self.rows, self.cols = subplot_layout
        self.enable_toggle = enable_toggle
        self.cache = cache if cache is not None else {}

        self.index = 0
        self.fig, self.axs = plt.subplots(self.rows, self.cols, squeeze=False, figsize=figsize)
        self.axs = self.axs.flatten()
        self.fig.subplots_adjust(bottom=0.2)

        # Persistent plot handles for speed
        self.plot_objects = [None] * (self.rows * self.cols)

        # Navigation UI
        axprev = plt.axes([0.1, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.21, 0.05, 0.1, 0.075])
        self.bprev = Button(axprev, 'Previous')
        self.bnext = Button(axnext, 'Next')
        self.bprev.on_clicked(lambda event: self.goto(self.index - 1))
        self.bnext.on_clicked(lambda event: self.goto(self.index + 1))
        self.raster_cache = {}  # {index: [LineCollection, ...]}

        # Toggle button
        if self.enable_toggle:
            axtoggle = plt.axes([0.72, 0.05, 0.1, 0.075])
            self.btoggle = Button(axtoggle, 'Select', color='lightgray')
            self.btoggle.on_clicked(self.toggle_selection)
            self.toggle_key = toggle_key.lower()
            self.toggle_states = [False] * self.n_items

        axclose = plt.axes([0.83, 0.05, 0.1, 0.075])
        self.bclose = Button(axclose, 'Close', color='lightgray')
        self.bclose.on_clicked(lambda event: self.close())

        # Slider
        axslider = plt.axes([0.4, 0.05, 0.25, 0.03])
        self.slider = Slider(axslider, 'Index', 0, n_items - 1, valinit=0, valstep=1)
        self.slider.on_changed(lambda val: self.goto(int(val)))

        # Key event throttling
        self.last_key_time = 0
        self.key_delay = 0.08
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        if self.enable_toggle:
            self.counter_text = self.fig.text(
                0.98, 0.02, "", ha='right', va='bottom', fontsize=10, color='black'
            )

        self.draw_item(self.index)

    def draw_item(self, index):
        self.index = index
        for ax in self.axs:
            ax.clear()

        self.plot_function(index, self.axs, (self.rows, self.cols))
        self.slider.set_val(index)

        if self.enable_toggle:
            self.btoggle.ax.set_facecolor(
                'lightgreen' if self.toggle_states[index] else 'lightgray'
            )
            selected_count = sum(self.toggle_states)
            self.counter_text.set_text(f"Selected: {selected_count}/{self.n_items}")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # Restore focus safely (backend-aware)
        try:
            mgr = getattr(self.fig.canvas, "manager", None)
            if hasattr(mgr, "window"):  # QtAgg, WXAgg
                mgr.window.activateWindow()
                mgr.window.raise_()
            elif hasattr(self.fig.canvas, "_tkcanvas"):  # TkAgg
                self.fig.canvas._tkcanvas.focus_set()
        except Exception:
            pass

    def goto(self, index):
        index = max(0, min(self.n_items - 1, index))
        if index != self.index:
            self.draw_item(index)

    def toggle_selection(self, event):
        self.toggle_states[self.index] = not self.toggle_states[self.index]
        self.counter_text.set_text(f"Selected: {sum(self.toggle_states)}/{self.n_items}")
        self.btoggle.ax.set_facecolor('lightgreen' if self.toggle_states[self.index] else 'lightgray')
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        # Force focus back to figure so spacebar continues to work
        try:
            self.fig.canvas.manager.window.activateWindow()
            self.fig.canvas.manager.window.raise_()
        except Exception:
            pass

    def on_key(self, event):
        now = time.time()
        if now - self.last_key_time < self.key_delay:
            return
        self.last_key_time = now
        if event.key in ['right', 'n']:
            self.goto(self.index + 1)
        elif event.key in ['left', 'p']:
            self.goto(self.index - 1)
        elif self.enable_toggle and (event.key.lower() == self.toggle_key):
            self.toggle_selection(event)

    def get_toggle_states(self):
        return self.toggle_states
        
    def show(self, **kwargs):
        plt.show(**kwargs)

    def close(self):
        self.fig.canvas.mpl_disconnect(self.fig.canvas.manager.key_press_handler_id)
        plt.close(self.fig)