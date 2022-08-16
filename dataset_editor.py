import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import sys
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, SpanSelector


class Plotter():

    def __init__(self, data):
        x,y,conf,patt = self._get_data_from_file(data)
        self.file_name = data
        self.x_data = x
        self.y_data = y
        self.conf = conf
        self.patt = patt
        self.fig = None
        self.base = []
        self.seg_x = []
        self.seg_y = []
        self.color = []
        self.ax1 = None
        self.ax2 = None
        self.editing = False
        self.current_span = []
        self.span_pos = []
        self.span_ax1 = None
        self.span_ax2 = None
        self._create_button()


    def _get_data_from_file(self, file_name):
        x, y, conf, patt = [],[],[],[]
        with open(file_name, "r") as f:
            f.readline()
            lines = f.readlines()
            for line in lines:
                data = line.split('\t')
                x.append(float(data[0]))
                y.append(float(data[1]))
                conf.append(float(data[2]))
                patt.append(data[3][:-1])
        return x, y, conf, patt


    def plot(self):
        '''
        red    -> fixation
        blue   -> saccade
        green  -> pursuit 
        yellow -> blink
        '''
        self._segment(self.x_data,self.y_data,self.patt)

        self.fig = plt.figure(1)
        self._connect()
        self.ax1, self.ax2 = self.fig.subplots(2, 1, sharex=True)
        self.ax1.plot(self.base, self.seg_x, alpha=0.5, zorder=1)
        self.ax1.scatter(x=self.base, y=self.seg_x, c=self.color,
                         s=5, zorder=2)
        plt.xlabel('samples')
        plt.ylabel('x-displacement (0-1)')
        plt.legend(handles=self._set_legend())
        plt.grid()

        self.ax2.plot(self.base, self.seg_y, alpha=0.5, zorder=1)
        self.ax2.scatter(x=self.base, y=self.seg_y, c=self.color, 
                         s=5, zorder=2)
        plt.xlabel('samples')
        plt.ylabel('y-displacement (0-1)')
        plt.legend(handles=self._set_legend())
    

        self.span_ax1 = SpanSelector(self.ax1, self._onselect, 
                    'horizontal', useblit=True, button=1,
                    props=dict(alpha=0.5, facecolor='red'))
        self.span_ax2 = SpanSelector(self.ax2, self._onselect, 
                    'horizontal', useblit=True, button=1,
                    props=dict(alpha=0.5, facecolor='red'))
        
        plt.grid()
        plt.show()


    def _connect(self):
        self.cid = self.fig.canvas.mpl_connect('key_press_event',
            self._onpress)


    def _onpress(self, event):
        if event.key == ',':
            self.ax1.get_xaxis().pan(-2)
        elif event.key == '.':
            self.ax1.get_xaxis().pan(2)
        bounds = self.ax1.get_xbound()
        self.ax2.set_xbound(bounds)
        plt.draw()


    def replot(self, i, j):
        self.ax1.scatter(x=self.base[i:j], y=self.seg_x[i:j], 
                         c=self.color[i:j], s=5, zorder=2)
        self.ax2.scatter(x=self.base[i:j], y=self.seg_y[i:j], 
                         c=self.color[i:j], s=5, zorder=2)
        plt.draw()


    def mirror(self, src_file):
        patt = []
        with open(src_file, 'r') as f:
            f.readline()
            lines = f.readlines()
            for line in lines:
                data = line.split('\t')
                patt.append(data[3])
        self.patt = patt

    
    def _graph_to_text(self, event):
        data = "X_coord\tY_coord\tConfidence\tPattern\n"
        for i in range(len(self.base)):
            color = self.color[i]
            pattern = self._get_pattern(color)
            x = str(self.x_data[i]) + '\t'
            y = str(self.y_data[i]) + '\t'
            conf = "%1.3f"%self.conf[i] + '\t'
            data += x + y + conf + pattern + '\n'
        self._save_to_disk(data)


    def _save_to_disk(self, data):
        new_file = self.file_name[:-4] + "_edited.txt"
        with open(new_file, 'w') as f:
            f.write(data)
        print('Saved as ', new_file)


    def _change_label(self, event):
        label = event.inaxes.get_label()
        color = self._get_color(label)
        xdata_0 = int(self.span_pos[0])
        xdata_1 = int(self.span_pos[1])
        i, j = self.base[xdata_0], self.base[xdata_0]
        while self.base[j] <= xdata_1 and j < len(self.base):
            self.color[j] = color
            j += 1
        self.replot(i,j)
        

    def _create_button(self):
        self.b_ax = plt.axes([0.8,0.92,0.07,0.05], facecolor='gray')
        self.bsave = Button(self.b_ax, 'Save')
        self.bsave.on_clicked(self._graph_to_text)
        self.fixation_ax = plt.axes([0.15, 0.9, 0.07, 0.05], label='F')
        self.saccade_ax = plt.axes([0.24, 0.9, 0.07, 0.05], label='S')
        self.pursuit_ax = plt.axes([0.32, 0.9, 0.07, 0.05], label='P')
        self.blink_ax = plt.axes([0.40, 0.9, 0.07, 0.05], label='B')
        self.btn_fixation = Button(self.fixation_ax, 'F', color='red')
        self.btn_fixation.on_clicked(self._change_label)
        self.btn_saccade = Button(self.saccade_ax, 'S', color='blue')
        self.btn_saccade.on_clicked(self._change_label)
        self.btn_pursuit = Button(self.pursuit_ax, 'P', color='green')
        self.btn_pursuit.on_clicked(self._change_label)
        self.btn_blink = Button(self.blink_ax, 'B', color='yellow')
        self.btn_blink.on_clicked(self._change_label)


    def _onselect(self, xmin, xmax):
        sp1 = self.ax1.axvspan(xmin, xmax, facecolor='0.4', alpha=0.3)
        sp2 = self.ax2.axvspan(xmin, xmax, facecolor='0.4', alpha=0.3)
        if self.editing:
            self.current_span[0].remove()
            self.current_span[1].remove()
            plt.draw()
        self.current_span = [sp1, sp2]
        self.span_pos = [xmin, xmax]
        self.editing = True


    def _set_legend(self):
        elements = [
            Line2D([0],[0], marker='o', color='r', label='Fixation'),
            Line2D([0],[0], marker='o', color='b', label='Saccade'),
            Line2D([0],[0], marker='o', color='g', label='Pursuit'),
            Line2D([0],[0], marker='o', color='y', label='Blink')
        ]
        return elements


    def _segment(self, x, y, patt):
        for i in range(len(x)):
            color = self._get_color(patt[i])
            self.seg_x.append(x[i])
            self.seg_y.append(y[i])
            self.base.append(i)
            self.color.append(color)


    def _get_color(self, pattern):
        if pattern == 'F':
            return 'r'
        elif pattern == 'S':
            return 'b'
        elif pattern == 'P':
            return 'g'
        else:
            return 'y'


    def _get_pattern(self, color):
        if color == 'r':
            return 'F'
        elif color == 'b':
            return 'S'
        elif color == 'g':
            return 'P'
        else:
            return 'B'


#----------------------------------------------
if __name__=="__main__":
    if len(sys.argv) == 2:
        pltr = Plotter(sys.argv[1])
        pltr.plot()
    elif len(sys.argv) == 3:
        pltr = Plotter(sys.argv[1])
        pltr.mirror(sys.argv[2])
        pltr.plot()
    else:
        print("usage: program <file_to_plot> [file_to_mirror_from]")
        sys.exit()
