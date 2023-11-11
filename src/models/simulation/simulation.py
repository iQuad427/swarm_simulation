import random
import threading
import time

import dearpygui.dearpygui as dpg
import math
import numpy as np

from src.models.simulation.arena import Arena, RectangleArena
from src.modules.communication.model import Communication, FakeCommunication
from src.modules.triangulation.model import Triangulation, FakeTriangulation

arena_width = 50
arena_height = 50

communication_refresh_rate = 0.01  # 10 milliseconds
triangulation_refresh_rate = 0.05  # 500 milliseconds

tri_x = {"triangulation": []}
tri_y = {"triangulation": []}

paused: bool = True


def toggle_pause():
    global paused
    paused = not paused


# Define the Agent class
class Agent:
    def __init__(
            self, agent_id, x, y,
            agents_speed=0.01,
            communication_chances=0,  # TODO: implement communication module and remove this parameter
            triangulation: Triangulation = FakeTriangulation(agent_id=0),
            communication: Communication = FakeCommunication(),
    ):
        global tri_x, tri_y

        self.id = agent_id
        self.x = x
        self.y = y

        self.radius = 1
        self.color = [255, 255, 255]

        self.communication = communication
        self.triangulation = triangulation

        tri_y[str(self)] = []
        tri_x[str(self)] = []

        self.data = dict()
        self.data[self.id] = dict()

        self.communication_chances = communication_chances

        self.speed = agents_speed
        self.angle = random.randint(0, 360)
        self.dx = math.cos(self.angle)
        self.dy = math.sin(self.angle)

    def __str__(self):
        return f"agent_{self.id}"

    def verify_other_agents(self, agents: list):
        for agent in agents:
            if (
                    agent.y - agent.radius < self.y + self.radius < agent.y + agent.radius or agent.y - agent.radius < self.y - self.radius < agent.y + agent.radius
            ) and (
                    agent.x - agent.radius < self.x + self.radius < agent.x + agent.radius or agent.x - agent.radius < self.x - self.radius < agent.x + agent.radius
            ):
                if self.x < agent.x:
                    self.dx = -1
                else:
                    self.dx = 1
                if self.y < agent.y:
                    self.dy *= -1
                else:
                    self.dy *= 1

    def verify_wall_collide(self):
        if self.x - self.radius < 0:
            self.dx = 1
        elif self.x + self.radius > arena_width:
            self.dx = -1

        if self.y - self.radius < 0:
            self.dy = 1
        elif self.y + self.radius > arena_height:
            self.dy = -1

    def collide(self, agents: list):
        self.verify_wall_collide()
        self.verify_other_agents(agents)

    def walk_forward(self, agents: list):
        # Simulate random movement
        self.x += self.speed * self.dx
        self.y += self.speed * self.dy

        self.collide(agents)

    def receive_information(self, other_agents: list, context: np.ndarray):
        index = round(random.uniform(0, 1) * (len(other_agents) - 1))
        other_agent = other_agents[index]

        if other_agent.id == self.id:
            return

        information = other_agent.send_information()
        if self.id < other_agent.id:
            distance = context[self.id, other_agent.id]
        else:
            distance = context[other_agent.id, self.id]

        self.data[self.id][other_agent.id] = distance
        self.data[other_agent.id] = information

        # TODO: add distance computation errors, noise, etc.
        self.triangulation.update_distance_matrix(other_agent.id, information, distance)

    def send_information(self):
        return self.data[self.id]

    def triangulation_handler(self):
        global triangulation_refresh_rate

        while True:
            if paused:
                # Avoid causing the thread to over-consume CPU
                time.sleep(triangulation_refresh_rate)
                continue

            x, y = self.triangulation.update_triangulation()

            if x is not None and y is not None:
                tri_x[f"agent_{self.id}"] = x
                tri_y[f"agent_{self.id}"] = y

            time.sleep(triangulation_refresh_rate)

    def communication_handler(self, agents, context):
        global communication_refresh_rate

        while True:
            if paused:
                # Avoid causing the thread to over-consume CPU
                time.sleep(communication_refresh_rate)
                continue

            if random.random() < self.communication_chances:
                self.receive_information(agents, context)

            time.sleep(communication_refresh_rate)


class Simulation:
    def __init__(
            self, dim=5, arena: Arena = RectangleArena(width=arena_width, height=arena_height),
            refresh_rate=0.01,  # 10 milliseconds
            agents_speed=0.05,  # 5 centimeters per seconds
            triangulation_precision=1.0,  # 1 meter
            communication_frequency=0.5,  # 500 milliseconds
    ):
        self.dim = dim
        self.agents = []

        self.arena = arena

        # Parameters
        self.refresh_rate = refresh_rate
        self.agents_speed = agents_speed * self.refresh_rate  # to have a speed in meters per refresh_rate
        self.triangulation_precision = triangulation_precision
        self.communication_chances = communication_refresh_rate / communication_frequency

        self.distance_matrix = np.zeros((dim, dim), dtype=float)
        self.connection_matrix = np.zeros((dim, dim), dtype=int)
        self.saved_const = []

        self.main_window = None

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

    def launching_agents_controller(self, start, end):
        for i in range(start, end):
            agent = self.agents[i]
            print("Starting Agent " + str(agent.id))
            agent_thread = threading.Thread(target=self.agent_controller, args=(agent,))
            agent_thread.start()

    def agent_controller(self, thread_agent: Agent):
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
            thread_agent.walk_forward(self.agents)

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
        global tri_x, tri_y

        for agent in self.agents:
            dpg.configure_item(f"triangulation_{str(agent)}", x=tri_x[str(agent)], y=tri_y[str(agent)])

    def launch_gui(self):
        dpg.create_context()
        dpg.create_viewport(title="Simulation", x_pos=0, y_pos=0, width=1100, height=645)

        with dpg.window(pos=[0, 0], autosize=True, no_collapse=True, no_resize=True, no_close=True,
                        no_move=True,
                        no_title_bar=True) as main_window:
            with dpg.group():
                with dpg.group(horizontal=True):
                    with dpg.plot(no_menus=False, no_title=True, no_box_select=True, no_mouse_pos=True, width=500,
                                  height=500, equal_aspects=True):
                        default_x = dpg.add_plot_axis(axis=0, no_gridlines=True, no_tick_marks=True,
                                                      no_tick_labels=True,
                                                      label="", lock_min=True)
                        dpg.set_axis_limits(axis=default_x, ymin=0, ymax=arena_width)
                        default_y = dpg.add_plot_axis(axis=1, no_gridlines=True, no_tick_marks=True,
                                                      no_tick_labels=True,
                                                      label="", lock_min=True)
                        dpg.set_axis_limits(axis=default_y, ymin=0, ymax=arena_height)

                        # Draw arena walls
                        dpg.draw_rectangle(pmin=[0, 0], pmax=[arena_width, arena_height], color=[33, 33, 33],
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
                        for agent in self.agents:
                            item = dpg.draw_circle(
                                (agent.x, agent.y), agent.radius,
                                color=agent.color, fill=agent.color,
                            )
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
                                    dpg.set_axis_limits(axis=triangulation_x_axis, ymin=-(arena_height * math.sqrt(2)),
                                                        ymax=arena_height * math.sqrt(2))
                                    triangulation_y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="y")
                                    dpg.set_axis_limits(axis=triangulation_y_axis, ymin=-(arena_width * math.sqrt(2)),
                                                        ymax=arena_width * math.sqrt(2))
                                    dpg.add_scatter_series(
                                        tri_x[str(agent)], tri_y[str(agent)],
                                        parent=dpg.last_item(), tag=f"triangulation_{str(agent)}"
                                    )

                        # with triangulation_plot:
                        #     self.update_matrices()
                        #     self.update_triangulation()
                        #
                        #     dpg.add_plot_axis(dpg.mvXAxis, label="x")
                        #     dpg.add_plot_axis(dpg.mvYAxis, label="y")
                        #     dpg.add_scatter_series(
                        #         tri_x["simulation"], tri_y["simulation"],
                        #         parent=dpg.last_item(), tag="triangulation"
                        #     )
                        #
                        #     self.render_triangulation()

                dpg.add_button(label="Pause", callback=toggle_pause)

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window(window=main_window, value=True)
        # dpg.start_dearpygui()

        while dpg.is_dearpygui_running():
            if not paused:
                for agent in self.agents:
                    dpg.configure_item(drawn_agents[agent.id], center=(agent.x, agent.y))

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
