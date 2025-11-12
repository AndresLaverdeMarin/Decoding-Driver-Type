"""
Manim Animation: Hilbert-Huang Transform Visualization

This module creates an animated visualization demonstrating the Hilbert-Huang Transform
process for analyzing driving behavior signals. The animation shows the decomposition
of speed signals into Intrinsic Mode Functions (IMFs) and their transformation using
the Hilbert transform to compute instantaneous amplitude.

The visualization is designed for presentations and educational purposes to illustrate
signal processing techniques applied to autonomous vehicle trajectory analysis.

Requirements:
    - manim: Community version of the animation engine
    - pandas: For data loading and manipulation
    - numpy: For numerical operations
"""

from manim import *
import pandas as pd
import numpy as np
from operator import add

import os
from pathlib import Path

# Get the current working directory for data file paths
local_path = Path(os.getcwd())

# Load preprocessed driving data
# Data contains speed, acceleration, and time headway signals with their IMF decompositions
data_ACC = pd.read_csv(local_path.joinpath("data").joinpath("data2plotACC.csv"))
data_HV = pd.read_csv(local_path.joinpath("data").joinpath("data2plotHV.csv"))


class CreateGraph(Scene):
    """
    Manim Scene for visualizing Hilbert-Huang Transform on driving signals.

    This scene creates an animated visualization showing:
    1. Input driving signals (speed, acceleration, time headway)
    2. Decomposition into Intrinsic Mode Functions (IMFs) via Empirical Mode Decomposition
    3. Transformation to Hilbert IMFs showing instantaneous amplitude
    4. Final reconstruction showing amplitude envelope

    The animation demonstrates signal processing techniques used in driver behavior analysis.
    """

    def construct(self):
        """
        Construct the complete animation sequence.

        This method builds and animates the visualization of the Hilbert-Huang Transform
        process, including data extraction, graph creation, and transformation animations.
        """
        # Extract time series data for visualization (samples 1500-5000)
        # This window contains representative driving behavior patterns
        speed = data_ACC["Speed"].values.tolist()[1500:5000]

        # Extract Intrinsic Mode Functions (IMFs) from Empirical Mode Decomposition
        # IMFs represent signal components at different frequency scales (9 levels)
        imf_speed = [
            data_ACC[f"Speed-imf-{n}"].values.tolist()[1500:5000] for n in range(0, 9)
        ]

        # Extract Hilbert-transformed IMFs showing instantaneous amplitude
        himf_speed = [
            data_ACC[f"Speed-himf-{n}"].values.tolist()[1500:5000] for n in range(0, 9)
        ]

        # Extract additional driving signals for context
        acc = data_ACC["Acceleration"].values.tolist()[1500:5000]
        th = data_ACC["TimeHeadway"].values.tolist()[1500:5000]

        # Create time axis (10 Hz sampling rate)
        time = np.arange(0, len(speed) / 10, 0.1).tolist()

        # Create speed signal visualization
        # This graph shows the raw speed data from autonomous vehicle trajectory
        speed_axes = Axes(
            x_range=[np.min(time), np.max(time) + 1, 50],
            y_range=[np.min(speed), np.max(speed) + 1, 5],
            tips=False,
            x_length=3.2,
            y_length=3.2,
            y_axis_config={
                "include_numbers": True,
                "font_size": 17,
            },
            x_axis_config={
                "include_numbers": True,
                "font_size": 17,
            },
        ).add_coordinates()
        speed_axes.to_edge(LEFT)
        speed_axis_labels = speed_axes.get_axis_labels(
            x_label=MathTex("Time [s]", font_size=17),
            y_label=MathTex("Speed [m/s]", font_size=17),
        )
        speed_line = speed_axes.plot_line_graph(
            time, speed, add_vertex_dots=False, line_color=BLUE
        )
        speed_stuff = VGroup(speed_axes, speed_line, speed_axis_labels)

        # Create acceleration signal visualization
        # Acceleration patterns distinguish between human and autonomous driving styles
        acc_axes = Axes(
            x_range=[np.min(time), np.max(time) + 1, 50],
            y_range=[np.min(acc) - 0.3, np.max(acc) + 0.1, 0.3],
            x_length=3.2,
            y_length=3.2,
            tips=False,
            y_axis_config={
                "include_numbers": True,
                "font_size": 17,
            },
            x_axis_config={
                "include_numbers": True,
                "font_size": 17,
            },
        ).add_coordinates()
        acc_axes.to_edge(LEFT)
        acc_axis_labels = acc_axes.get_axis_labels(
            x_label=MathTex("Time [s]", font_size=17),
            y_label=MathTex("Acceleration [m/s^2]", font_size=17),
        )
        acc_line = acc_axes.plot_line_graph(
            time, acc, add_vertex_dots=False, line_color=RED
        )
        acc_stuff = VGroup(acc_axes, acc_line, acc_axis_labels)

        # Create time headway signal visualization
        # Time headway represents the temporal gap between vehicles
        th_axes = Axes(
            x_range=[np.min(time), np.max(time) + 1, 50],
            y_range=[np.min(th), np.max(th) + 0.1, 0.1],
            x_length=3.2,
            y_length=3.2,
            tips=False,
            y_axis_config={
                "include_numbers": True,
                "font_size": 17,
            },
            x_axis_config={
                "include_numbers": True,
                "font_size": 17,
            },
        ).add_coordinates()
        th_axes.to_edge(LEFT)
        th_axis_labels = th_axes.get_axis_labels(
            x_label=MathTex("Time [s]", font_size=17),
            y_label=MathTex("TimeHeadway", font_size=17),
        )
        th_line = th_axes.plot_line_graph(
            time, th, add_vertex_dots=False, line_color=GREEN
        )
        th_stuff = VGroup(th_axes, th_line, th_axis_labels)

        # Group all input signals together for layout
        input_signals_ACC = VGroup(speed_stuff, acc_stuff, th_stuff).arrange(
            direction=RIGHT
        )

        # Create IMF (Intrinsic Mode Function) visualizations
        # IMFs represent signal decomposition at different frequency scales
        # Each IMF captures a specific oscillatory mode from the original signal
        imf_speed_stuff = []

        # Color scheme for distinguishing different IMF levels
        colors = [
            GREEN_D,
            RED_D,
            PURPLE_D,
            YELLOW_D,
            TEAL_D,
            MAROON_D,
            BLUE_D,
            GREY_D,
            GOLD_D,
        ]

        # Y-axis step ranges for each IMF level
        # Lower IMFs have smaller amplitudes requiring finer scales
        step_range = [
            0.01,
            0.01,
            0.01,
            0.05,
            0.2,
            0.35,
            1,
            1,
            2,
        ]

        # Create individual graphs for each IMF level
        for num, imf in enumerate(imf_speed):
            # Last IMF shows x-axis labels, others are stacked above it
            if num == len(imf_speed) - 1:
                imf_speed_axes = Axes(
                    x_range=[np.min(time), np.max(time) + 1, 50],
                    y_range=[np.min(imf), np.max(imf) + 0.005, step_range[num]],
                    x_length=5.5,
                    y_length=0.6,
                    tips=False,
                    y_axis_config={
                        "include_numbers": True,
                        "font_size": 10,
                    },
                    x_axis_config={
                        "include_numbers": True,
                        "font_size": 25,
                        "include_ticks": True,
                    },
                ).add_coordinates()
            else:
                # Intermediate IMFs: hide x-axis for cleaner stacking
                imf_speed_axes = Axes(
                    x_range=[np.min(time), np.max(time), 50],
                    y_range=[np.min(imf), np.max(imf) + 0.005, step_range[num]],
                    x_length=5.5,
                    y_length=0.6,
                    tips=False,
                    y_axis_config={
                        "include_numbers": True,
                        "font_size": 10,
                    },
                    x_axis_config={
                        "include_numbers": False,
                        "font_size": 0.1,
                        "include_ticks": False,
                    },
                ).add_coordinates()

            # Plot the IMF signal
            imf_speed_line = imf_speed_axes.plot_line_graph(
                time, imf, add_vertex_dots=False, line_color=colors[num]
            )
            imf_speed_stuff.append(VGroup(imf_speed_axes, imf_speed_line))

        # Stack all IMF graphs vertically and position on right side
        imf_speed_group = VGroup(
            *[imf_speed_stuff[n] for n in range(0, len(imf_speed_stuff))]
        ).arrange(direction=DOWN)
        imf_speed_group.to_edge(RIGHT)

        # Create Hilbert-transformed IMF (HIMF) visualizations
        # HIMFs show the instantaneous amplitude envelope of each IMF
        # This reveals the time-varying frequency content of the signal
        final_himf = []
        himf_speed_stuff = []

        # Use same color scheme for consistency across IMF and HIMF visualizations
        colors = [
            GREEN_D,
            RED_D,
            PURPLE_D,
            YELLOW_D,
            TEAL_D,
            MAROON_D,
            BLUE_D,
            GREY_D,
            GOLD_D,
        ]

        # Y-axis step ranges matching IMF configuration
        step_range = [
            0.01,
            0.01,
            0.01,
            0.05,
            0.2,
            0.35,
            1,
            1,
            2,
        ]

        # Create individual graphs for each Hilbert-transformed IMF
        for num, himf in enumerate(himf_speed):
            # Configure axes similarly to IMF graphs
            if num == len(imf_speed) - 1:
                # Bottom HIMF shows x-axis labels
                himf_speed_axes = Axes(
                    x_range=[np.min(time), np.max(time) + 1, 50],
                    y_range=[np.min(himf), np.max(himf) + 0.005, step_range[num]],
                    x_length=8,
                    y_length=0.35,
                    tips=False,
                    y_axis_config={
                        "include_numbers": True,
                        "font_size": 10,
                    },
                    x_axis_config={
                        "include_numbers": True,
                        "font_size": 25,
                        "include_ticks": True,
                    },
                ).add_coordinates()
            else:
                # Upper HIMFs: hide x-axis for compact stacking
                himf_speed_axes = Axes(
                    x_range=[np.min(time), np.max(time), 50],
                    y_range=[np.min(himf), np.max(himf) + 0.005, step_range[num]],
                    x_length=8,
                    y_length=0.35,
                    tips=False,
                    y_axis_config={
                        "include_numbers": True,
                        "font_size": 10,
                    },
                    x_axis_config={
                        "include_numbers": False,
                        "font_size": 0.1,
                        "include_ticks": False,
                    },
                ).add_coordinates()

            # Store HIMF data for final summation
            final_himf.append(himf)

            # Plot the Hilbert-transformed IMF
            himf_speed_line = himf_speed_axes.plot_line_graph(
                time, himf, add_vertex_dots=False, line_color=colors[num]
            )
            himf_speed_stuff.append(VGroup(himf_speed_axes, himf_speed_line))

        # Stack all HIMF graphs vertically
        himf_speed_group = VGroup(
            *[himf_speed_stuff[n] for n in range(0, len(himf_speed_stuff))]
        ).arrange(direction=DOWN)

        # Create final reconstruction by summing all Hilbert-transformed IMFs
        # This produces the instantaneous amplitude envelope of the original signal
        # Used as a feature for distinguishing driver types
        sum_himf = [sum(x) for x in zip(*final_himf)]

        himf_sum_speed_axes = Axes(
            x_range=[np.min(time), np.max(time) + 1, 250],
            y_range=[1, 12 + 0.1, 2],
            tips=False,
            x_length=6,
            y_length=4.5,
            y_axis_config={
                "include_numbers": True,
                "font_size": 17,
            },
            x_axis_config={
                "include_numbers": True,
                "font_size": 17,
            },
        ).add_coordinates()

        # Add axis labels with professional formatting
        himf_sum_speed_axis_labels = himf_sum_speed_axes.get_axis_labels(
            x_label=Text("Time [s]", font_size=17, font="Times New Roman"),
            y_label=Text("Speed Amplitude", font_size=17, font="Times New Roman"),
        )

        # Plot the summed amplitude envelope in gold color
        himf_sum_speed_line = himf_sum_speed_axes.plot_line_graph(
            time, sum_himf, add_vertex_dots=False, line_color="#EFC859"
        )

        himf_sum_speed_stuff = VGroup(
            himf_sum_speed_axes, himf_sum_speed_line, himf_sum_speed_axis_labels
        )

        # Create visual elements for annotation

        # Arrow indicating transformation process
        arrow = Arrow(buff=2.2, start=1.8 * LEFT, end=1.85 * RIGHT)
        arrow.move_to(1.5 * LEFT)

        # Text explaining Empirical Mode Decomposition step
        text_1 = Text("Compute Intrinsic Mode Functions (IMF)", font_size=20)
        text_1.to_edge(UL)

        # Text explaining Hilbert Transform step
        text_2 = Text(
            "Compute instantaneous amplitude from a set of IMFs", font_size=20
        )
        text_3 = Text("applying Hilbert transformation", font_size=20)
        ht_text = VGroup(text_2, text_3).arrange(DOWN)
        ht_text.to_edge(UL)

        # Animation sequence
        # Note: Most animations are commented out for faster rendering during development
        # Uncomment these lines to create the full step-by-step visualization

        # Step 1: Display individual input signals
        # self.play(Create(speed_axes, run_time=2))
        # self.wait(1)
        # self.play(Write(speed_axis_labels))
        # self.play(Create(speed_line))
        # self.wait(1)
        # self.play(speed_stuff.animate.scale(0.4))
        # self.play(speed_stuff.animate.shift(UP * 2.5))
        # self.play(speed_stuff.animate.shift(LEFT * 8))
        # self.wait(1)
        # self.play(Create(acc_axes, run_time=2))
        # self.wait(1)
        # self.play(Write(acc_axis_labels))
        # self.play(Create(acc_line))
        # self.play(acc_stuff.animate.scale(0.4))
        # self.play(acc_stuff.animate.shift(DOWN * 0.5))
        # self.play(acc_stuff.animate.shift(LEFT * 8))
        # self.wait(1)
        # self.play(Create(th_axes, run_time=2))
        # self.wait(1)
        # self.play(Write(th_axis_labels))
        # self.play(Create(th_line))
        # self.play(th_stuff.animate.scale(0.4))
        # self.play(th_stuff.animate.shift(DOWN * 2.75))
        # self.play(th_stuff.animate.shift(LEFT * 8))
        # self.wait(1)
        # self.remove(acc_stuff, th_stuff)
        # self.wait(1)
        # self.play(speed_stuff.animate.shift(RIGHT * 1))
        # self.play(speed_stuff.animate.shift(DOWN * 2.5))
        # self.play(speed_stuff.animate.scale(1.5), Create(row))

        # Step 2: Show all input signals together
        # self.play(Create(input_signals_ACC, run_time=2))
        # self.wait(1)
        # self.remove(input_signals_ACC)

        # Step 3: Show EMD decomposition into IMFs
        # self.play(
        #     Create(speed_stuff, run_time=2),
        #     Create(arrow, run_time=2),
        #     Create(text_1, run_time=1),
        # )
        # self.play(Create(imf_speed_group, run_time=2))
        # self.wait(1)

        # Step 4: Transform IMFs to Hilbert-transformed IMFs
        # self.play(
        #     Transform(imf_speed_group, himf_speed_group), Transform(text_1, ht_text)
        # )
        # self.wait(1)

        # Step 5: Show final amplitude envelope (active animation)
        # Display Hilbert-transformed IMFs
        self.play(Create(himf_speed_group, run_time=1))

        # Transform to show summed amplitude envelope
        self.play(Transform(himf_speed_group, himf_sum_speed_stuff))
        self.wait(3)
