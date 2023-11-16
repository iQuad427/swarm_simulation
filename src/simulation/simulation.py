import threading
import time

import dearpygui.dearpygui as dpg
import math
import numpy as np

from src.simulation.agent import Agent
from src.simulation.arena import Arena, RectangleArena

# TODO: could want to use `pause` methods in agents to pause their threads
paused: bool = False


class Simulation:
    def __init__(
            self,
            # SIMULATION PARAMETERS
            dim=5,
            arena: Arena = RectangleArena(xlim=50, ylim=50, width=50, height=50),
            refresh_rate=0.01,  # 10 milliseconds
            # AGENTS PARAMETERS
            agents_speed=0.05,  # 5 centimeters per seconds
            # MODULES PARAMETERS
            triangulation_precision=1.0,  # 1 meter
            communication_frequency=0.5,  # 500 milliseconds
    ):
        self.dim = dim
        self.agents = []
        self.agents_thread = []

        self.arena = arena

        # Parameters
        self.refresh_rate = refresh_rate  # simulation refresh rate in seconds
        self.agents_speed = agents_speed * self.refresh_rate  # to have a speed in meters per refresh_rate
        self.triangulation_precision = triangulation_precision
        self.communication_frequency = communication_frequency

        # Simulation Information
        self.distance_matrix = np.zeros((dim, dim), dtype=float)
        self.connection_matrix = np.zeros((dim, dim), dtype=int)
        self.saved_const = []

        self.main_window = None

    def toggle_pause(self):
        global paused
        paused = not paused

        for agent in self.agents:
            agent.paused = paused

    def launch_agent_threads(self):
        size = len(self.agents)

        launch_threads = [
            threading.Thread(
                target=self.launching_agents_controller,
                args=(i, i + size // 4)
            ) for i in range(0, size, size // 4)
        ]

        for thread in launch_threads:
            thread.start()
            self.agents_thread.append(thread)

    def stop_agent_threads(self):
        for thread in self.agents_thread:
            thread.join()

    def launching_agents_controller(self, start, end):
        for i in range(start, end):
            agent = self.agents[i]
            print("Starting Agent " + str(agent.id))
            agent_thread = threading.Thread(target=self.agent_controller, args=(agent,))
            agent_thread.start()

    def agent_controller(self, thread_agent: Agent):
        global paused

        # Start agent triangulation
        triangulation_thread = threading.Thread(
            target=thread_agent.triangulation_handler,
        )
        triangulation_thread.start()

        # Simulate agen communication
        communication_thread = threading.Thread(
            target=thread_agent.communication_handler,
            args=(self.agents, self.distance_matrix),
        )
        communication_thread.start()

        print("Agent " + str(thread_agent.id) + " started")
        while True:
            if paused:
                # Avoid causing the thread to over-consume CPU
                time.sleep(self.refresh_rate)
                continue

            # Simulate agent movement
            thread_agent.move(thread_agent)

            # Simulate agent collision
            self.arena.collide(thread_agent)
            thread_agent.collide(self.agents)

            time.sleep(self.refresh_rate)

    def update_matrices(self):
        for i in range(self.dim):
            for j in range(i, self.dim):
                if i != j:
                    self.distance_matrix[i, j] = math.sqrt(
                        (self.agents[i].x - self.agents[j].x) ** 2 + (self.agents[i].y - self.agents[j].y) ** 2
                    )
                    dpg.configure_item(
                        int(self.connection_matrix[i, j]),
                        p1=(self.agents[i].x, self.agents[i].y),
                        p2=(self.agents[j].x, self.agents[j].y)
                    )
                else:
                    self.distance_matrix[i, j] = 0

    def render_triangulation(self):
        for agent in self.agents:
            dpg.configure_item(f"triangulation_{str(agent)}", x=agent.tri_x, y=agent.tri_y)

    def launch_gui(self):
        dpg.create_context()
        dpg.create_viewport(title="Simulation", x_pos=0, y_pos=0, width=1100, height=645)

        with dpg.window(
                pos=[0, 0], autosize=True, no_title_bar=True,
                no_collapse=True, no_resize=True, no_close=True, no_move=True,
        ) as main_window:
            with dpg.group():
                with dpg.group(horizontal=True):
                    with dpg.plot(no_menus=False, no_title=True, no_box_select=True, no_mouse_pos=True, width=500,
                                  height=500, equal_aspects=True):
                        default_x = dpg.add_plot_axis(
                            axis=0, no_gridlines=True, no_tick_marks=True, no_tick_labels=True, label="", lock_min=True
                        )
                        dpg.set_axis_limits(axis=default_x, ymin=0, ymax=self.arena.xlim)
                        default_y = dpg.add_plot_axis(
                            axis=1, no_gridlines=True, no_tick_marks=True, no_tick_labels=True, label="", lock_min=True
                        )
                        dpg.set_axis_limits(axis=default_y, ymin=0, ymax=self.arena.ylim)

                        # Draw arena walls
                        if isinstance(self.arena, RectangleArena):
                            dpg.draw_rectangle(pmin=[0, 0], pmax=[self.arena.width, self.arena.height],
                                               color=[33, 33, 33],
                                               fill=[33, 33, 33])

                        # Draw connections between agents
                        for i in range(self.dim):
                            for j in range(i, self.dim):
                                if i != j:
                                    self.connection_matrix[i, j] = dpg.draw_line(
                                        (self.agents[i].x, self.agents[i].y), (self.agents[j].x, self.agents[j].y),
                                        color=[255, 0, 0], thickness=0.1
                                    )

                        # Draw agents
                        drawn_agents = []
                        agents_label = []
                        for agent in self.agents:
                            item = dpg.draw_circle(
                                (agent.x, agent.y), agent.radius,
                                color=agent.color, fill=agent.color,
                            )
                            text = dpg.draw_text(
                                (agent.x - agent.radius / 2, agent.y + agent.radius), str(agent.id),
                                color=[0, 0, 0], size=2
                            )
                            agents_label.append(text)
                            drawn_agents.append(item)

                    triangulation_tab_bar = dpg.tab_bar(label="triangulation_tab_bar")

                    with triangulation_tab_bar:
                        for agent in self.agents:
                            with dpg.tab(label=f"{str(agent)}"):
                                with dpg.plot(
                                        no_menus=False, no_title=True, no_box_select=True, no_mouse_pos=True,
                                        width=500, height=500, equal_aspects=True
                                ):
                                    triangulation_x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="x")
                                    dpg.set_axis_limits(
                                        axis=triangulation_x_axis,
                                        ymin=-(self.arena.xlim * math.sqrt(2)),
                                        ymax=self.arena.xlim * math.sqrt(2)
                                    )
                                    triangulation_y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="y")
                                    dpg.set_axis_limits(
                                        axis=triangulation_y_axis,
                                        ymin=-(self.arena.ylim * math.sqrt(2)),
                                        ymax=self.arena.ylim * math.sqrt(2)
                                    )
                                    dpg.add_scatter_series(
                                        agent.tri_x, agent.tri_y,
                                        parent=dpg.last_item(), tag=f"triangulation_{str(agent)}"
                                    )

                dpg.add_button(label="Pause", callback=self.toggle_pause)

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window(window=main_window, value=True)
        # dpg.start_dearpygui()

        while dpg.is_dearpygui_running():
            if not paused:
                for agent in self.agents:
                    dpg.configure_item(drawn_agents[agent.id], center=(agent.x, agent.y))
                    dpg.configure_item(agents_label[agent.id], pos=(agent.x - agent.radius / 2, agent.y + agent.radius))

                self.update_matrices()
                # self.update_triangulation()

            self.render_triangulation()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()

    def setup(self):
        raise NotImplementedError("Simulation does not implement a specific setup")

    def launch(self):
        self.setup()

        self.launch_agent_threads()
        self.launch_gui()
