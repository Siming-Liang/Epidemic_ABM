"""
Utilities for creating visualizations
"""
import numpy as np
import solara
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib import animation
from mesa.visualization.utils import update_counter
from mpl_toolkits.axes_grid1 import make_axes_locatable


def good_colorbar(im, fig, ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im, cax=cax)


def region_state_plot(model):
    update_counter.get()
    fig = Figure()
    axes = fig.subplots(1, 3)

    susceptible = model.datacollector.model_vars['Susceptible-local'][-1]
    infected = model.datacollector.model_vars['Infected-local'][-1]
    resistant = model.datacollector.model_vars['Resistant-local'][-1]

    for ax, data, var_name in zip(
            axes,
            (susceptible, infected, resistant),
            ('Susceptible', 'Infected', 'Resistant')
    ):
        im = ax.imshow(
            np.flip(data.T, axis=(0,)),
            interpolation='nearest',
            vmin=0,
            extent=(-0.5, model.state_grid_width - 0.5, -0.5, model.state_grid_height - 0.5)
        )
        ax.set_title(var_name)
        good_colorbar(im, fig, ax)

    fig.tight_layout()

    solara.FigureMatplotlib(
        fig, format='png', bbox_inches="tight", dependencies=None
    )


def properties_plot(model):
    update_counter.get()
    fig = Figure()
    axes = fig.subplots(1, 3)

    spread_chance = model.get_spread_chance()
    recovery_chance = model.get_recovery_chance()
    gain_resistance_chance = model.get_gain_resistance_chance()


    for ax, data, var_name in zip(
            axes,
            (spread_chance, recovery_chance, gain_resistance_chance),
            ('Spread chance', 'Recovery chance', 'Gain resistance chance')
    ):
        im = ax.imshow(
            np.flip(data.T, axis=(0,)),
            interpolation='nearest',
            vmin=0.0,
            vmax=1.0,
            extent=(-0.5, model.state_grid_width - 0.5, -0.5, model.state_grid_height - 0.5)
        )
        ax.set_title(var_name)
        good_colorbar(im, fig, ax)

    fig.tight_layout()

    solara.FigureMatplotlib(
        fig, format='png', bbox_inches="tight", dependencies=None
    )


class DiseaseMovie:
    def __init__(self, models, show_diff=False, step_range=None):
        if step_range is None:
            step_range = (0, models[0].steps + 1)
        self.step_range = step_range

        self.models = models

        self.fig, self.axes = plt.subplots(len(models) + int(show_diff), 3)
        if len(models) == 1:
            self.axes = [self.axes]

        self.axes[0][0].set_title('S')
        self.axes[0][1].set_title('I')
        self.axes[0][2].set_title('R')

        self.ims = []

        for ax_row, model in zip(self.axes, self.models):
            data = model.datacollector.model_vars
            im_row = [
                ax_row[0].imshow(
                    data['Susceptible-local'][self.step_range[0]],
                ),
                ax_row[1].imshow(
                    data['Infected-local'][self.step_range[0]],
                ),
                ax_row[2].imshow(
                    data['Resistant-local'][self.step_range[0]],
                )
            ]
            ax_row[0].set_title('S')
            ax_row[1].set_title('I')
            ax_row[2].set_title('R')
            self.ims.append(im_row)

        self.show_diff = show_diff
        if show_diff:
            ax_row = self.axes[-1]
            data1 = self.models[-1].datacollector.model_vars
            data2 = self.models[-2].datacollector.model_vars
            im_row = [
                ax_row[0].imshow(
                    np.abs(data1['Susceptible-local'][self.step_range[0]]
                           - data2['Susceptible-local'][self.step_range[0]]),
                ),
                ax_row[1].imshow(
                    np.abs(data1['Infected-local'][self.step_range[0]]
                           - data2['Infected-local'][self.step_range[0]]),
                ),
                ax_row[2].imshow(
                    np.abs(data1['Resistant-local'][self.step_range[0]]
                           - data2['Resistant-local'][self.step_range[0]]),
                )
            ]
            ax_row[0].set_title('S')
            ax_row[1].set_title('I')
            ax_row[2].set_title('R')
            self.ims.append(im_row)

        self.animation = animation.FuncAnimation(
            fig=self.fig,
            func=self.update,
            frames=self.step_range[1] - self.step_range[0],
            interval=1000
        )

    def update(self, frame):
        for im_row, model in zip(self.ims, self.models):
            data = model.datacollector.model_vars
            s = data['Susceptible-local'][frame + self.step_range[0]]
            im_row[0].set_data(s)
            im_row[0].set_clim(vmin=0, vmax=s.max())

            i = data['Infected-local'][frame + self.step_range[0]]
            im_row[1].set_data(i)
            im_row[1].set_clim(vmin=0, vmax=i.max())

            r = data['Resistant-local'][frame + self.step_range[0]]
            im_row[2].set_data(r)
            im_row[2].set_clim(vmin=0, vmax=r.max())

        if self.show_diff:
            im_row = self.ims[-1]
            data1 = self.models[-1].datacollector.model_vars
            data2 = self.models[-2].datacollector.model_vars

            ds = np.abs(data1['Susceptible-local'][frame + self.step_range[0]]
                        - data2['Susceptible-local'][frame + self.step_range[0]])
            im_row[0].set_data(ds)
            im_row[0].set_clim(vmin=0, vmax=np.max(ds))

            di = np.abs(data1['Infected-local'][frame + self.step_range[0]]
                        - data2['Infected-local'][frame + self.step_range[0]])
            im_row[1].set_data(di)
            im_row[1].set_clim(vmin=0, vmax=np.max(di))

            dr = np.abs(data1['Resistant-local'][frame + self.step_range[0]]
                        - data2['Resistant-local'][frame + self.step_range[0]])
            im_row[2].set_data(dr)
            im_row[2].set_clim(vmin=0, vmax=np.max(dr))
